"""Feature vectors + labels per (query, candidate).

`FeatureBuilder.build_commit` is called at the leakage boundary of the pipeline: the snapshot,
graph, token / content indexes and both miners all still describe parent(t); the only
post-commit information used is the seed block's own new body.

12 model features in three families (history / graph / text)
plus two locality features that are stored but not used by the methods by default
(`model.features` in the config selects the columns a method uses).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tfimpact.config import FeaturesConfig
from tfimpact.constants import ADDED, MODIFIED, REMOVED
from tfimpact.features.association_rules import AssociationRuleMiner, changed_key
from tfimpact.features.candidates import generate
from tfimpact.features.graph import RefGraph, adamic_adar_weights
from tfimpact.features.propagation import IN, OUT, PropagationMiner, edge_direction
from tfimpact.features.similarity import (
    ContentIndex,
    TfIdfVector,
    TokenIndex,
    cosine,
    jaccard,
    name_tokens,
    type_tokens,
)
from tfimpact.processing.diff import ChangeItem
from tfimpact.processing.snapshot import Snapshot

FEATURES = [
    # history: mined from earlier commits
    "hist_cochange_conf",  # N(seed, cand) / N(seed)
    "hist_cochange_count",  # log1p(N(seed, cand)) / log1p(max in query)
    "hist_cochange_recency",  # sum exp(-age / tau) over co-changes, / max in query
    "hist_cand_recent_changes",  # candidate changes in the last churn_window commits, / max
    "hist_file_cochange_conf",  # file-level rule file(seed) => file(cand)
    "hist_type_cochange_conf",  # type rule  action+type(seed) => *type(cand)  (or +T)
    # graph: reference structure at parent(t)
    "graph_proximity",  # 1 / (1 + d), d <= graph_max_dist
    "graph_shared_neighbors",  # Adamic-Adar over shared neighbors, normalized by the seed's
    "graph_cand_degree",  # candidate degree / max degree in the snapshot
    "graph_propagation_conf",  # mined rule (seed action, edge direction) => cand changes
    # text
    "text_name_similarity",  # Jaccard of name tokens (add-type: vs type name)
    "text_content_similarity",  # TF-IDF cosine, seed new body vs candidate parent body
    # locality: stored, not used by the methods by default
    "loc_same_file",
    "loc_same_dir",
]
N_FEATURES = len(FEATURES)
COL = {name: i for i, name in enumerate(FEATURES)}
FAMILIES = ("hist", "graph", "text", "loc")
EXISTING = "existing"
ADD_TYPE = "add_type"
ADD_TYPE_PREFIX = "+type:"
MAX_SEEDS_PER_COMMIT = 1000  # query_id = idx * 1000 + seed position (max_items <= 30)


def feature_family(name: str) -> str:
    return name.split("_", 1)[0]


def validate_feature_names(names: Sequence[str]) -> list[str]:
    """Check a feature selection (e.g. `model.features`) against the stored columns."""
    unknown = [n for n in names if n not in COL]
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}; known: {FEATURES}")
    return list(names)


_C_CONF = COL["hist_cochange_conf"]
_C_COUNT = COL["hist_cochange_count"]
_C_RECENCY = COL["hist_cochange_recency"]
_C_CHURN = COL["hist_cand_recent_changes"]
_C_FILE = COL["hist_file_cochange_conf"]
_C_TYPE = COL["hist_type_cochange_conf"]
_C_PROX = COL["graph_proximity"]
_C_SHARED = COL["graph_shared_neighbors"]
_C_DEGREE = COL["graph_cand_degree"]
_C_PROP = COL["graph_propagation_conf"]
_C_NAME = COL["text_name_similarity"]
_C_CONTENT = COL["text_content_similarity"]
_C_SAME_FILE = COL["loc_same_file"]
_C_SAME_DIR = COL["loc_same_dir"]


@dataclass
class CommitFeatures:
    queries: list[dict[str, Any]] = field(default_factory=list)
    query_ids: list[np.ndarray] = field(default_factory=list)
    cand_ids: list[list[str]] = field(default_factory=list)
    cand_kinds: list[list[str]] = field(default_factory=list)
    cand_files: list[list[str]] = field(default_factory=list)
    X: list[np.ndarray] = field(default_factory=list)
    y: list[np.ndarray] = field(default_factory=list)


def _max_norm(v: np.ndarray) -> np.ndarray:
    m = float(v.max()) if v.size else 0.0
    return v / m if m > 0 else np.zeros_like(v)


class FeatureBuilder:
    def __init__(
        self,
        cfg: FeaturesConfig,
        snap: Snapshot,
        graph: RefGraph,
        tokens: TokenIndex,
        miner: AssociationRuleMiner,
        propagation: PropagationMiner,
        content: ContentIndex,
    ) -> None:
        self.cfg, self.snap, self.graph, self.tokens = cfg, snap, graph, tokens
        self.miner, self.propagation, self.content = miner, propagation, content

    def build_commit(self, idx: int, sha: str, items: list[ChangeItem]) -> CommitFeatures:
        out = CommitFeatures()
        blocks = self.snap.blocks
        degree = self.graph.degrees()  # parent snapshot, once per commit
        max_deg = max(degree.values(), default=0)
        by_degree = sorted(degree, key=lambda a: (-degree[a], a))
        changed = {it.address for it in items if it.action in (MODIFIED, REMOVED)}
        added_by: dict[str, set[str]] = {}
        for it in items:
            if it.action == ADDED:
                added_by.setdefault(it.type, set()).add(it.address)
        content_cache: dict[str, TfIdfVector] = {}  # parent bodies; IDF fixed within a commit

        for k, q in enumerate(items):
            seed = q.new if q.new is not None else q.old  # removed seed: its last body
            assert seed is not None
            qid = idx * MAX_SEEDS_PER_COMMIT + k
            label_existing = changed - {q.address}
            label_types = {t for t, addrs in added_by.items() if addrs - {q.address}}

            dist = self.graph.distances(seed, self.cfg.graph_max_dist)
            cs = generate(
                seed,
                q.type_item,
                blocks,
                dist,
                self.snap.dir_addresses(seed.directory),
                self.miner,
                self.tokens,
                degree,
                by_degree,
                self.cfg.max_candidates,
                self.cfg.prefilter_max_dist,
            )
            X_e = self._existing_features(
                idx, seed, q, cs.existing, dist, degree, max_deg, content_cache
            )
            X_a = self._add_type_features(seed, q.type_item, cs.add_types)
            ids = cs.existing + [ADD_TYPE_PREFIX + t for t in cs.add_types]
            y = np.array(
                [c in label_existing for c in cs.existing]
                + [t in label_types for t in cs.add_types],
                dtype=np.int8,
            )
            out.X.append(np.vstack([X_e, X_a]).astype(np.float32))
            out.y.append(y)
            out.cand_ids.append(ids)
            out.cand_kinds.append([EXISTING] * len(cs.existing) + [ADD_TYPE] * len(cs.add_types))
            out.cand_files.append([blocks[c].file for c in cs.existing] + [""] * len(cs.add_types))
            out.query_ids.append(np.full(len(ids), qid, dtype=np.int64))
            out.queries.append(
                {
                    "query_id": qid,
                    "idx": idx,
                    "commit_sha": sha,
                    "seed_address": q.address,
                    "seed_type": q.type,
                    "seed_kind": q.kind,
                    "seed_action": q.action,
                    "seed_is_new": q.action == ADDED,
                    "n_labels": len(label_existing) + len(label_types),
                    "n_labels_existing": len(label_existing),
                    "n_labels_add_type": len(label_types),
                    "n_candidates": len(ids),
                    "n_prefilter_union": cs.n_union,
                    "n_labels_in_candidates": int(y.sum()),
                }
            )
        return out

    # --- feature families ---
    def _existing_features(
        self,
        idx: int,
        seed: Any,
        item: ChangeItem,
        cands: list[str],
        dist: dict[str, int],
        degree: dict[str, int],
        max_deg: int,
        content_cache: dict[str, TfIdfVector],
    ) -> np.ndarray:
        n = len(cands)
        X = np.zeros((n, N_FEATURES), dtype=np.float64)
        if n == 0:
            return X
        m, g, blocks, cfg = self.miner, self.graph, self.snap.blocks, self.cfg
        q = seed.address
        n_q = m.n_item.get(q, 0)
        partners = m.partners(q)
        incoming = g.in_neighbors(q)
        outgoing = g.out_neighbors(seed)  # seed's new body
        seed_nbrs = incoming | outgoing
        aa_w = adamic_adar_weights(seed_nbrs, degree)
        aa_norm = sum(aa_w.values())
        prop = {
            IN: self.propagation.confidence(item.action, IN),
            OUT: self.propagation.confidence(item.action, OUT),
        }
        seed_tok = name_tokens(seed.name)
        seed_vec = self.content.vector(seed)
        file_conf: dict[str, float] = {}
        sup = np.zeros(n)
        rec = np.zeros(n)
        churn = np.zeros(n)
        for i, c in enumerate(cands):
            b = blocks[c]
            # history
            st = partners.get(c)
            if st:
                sup[i] = st[0]
                rec[i] = st[1] * math.exp(-(idx - st[2]) / cfg.recency_tau)
                X[i, _C_CONF] = st[0] / n_q if n_q else 0.0
            churn[i] = m.churn(c, idx, cfg.churn_window)
            if b.file not in file_conf:
                file_conf[b.file] = m.file_confidence(seed.file, b.file)
            X[i, _C_FILE] = file_conf[b.file]
            X[i, _C_TYPE] = m.type_confidence(item.type_item, changed_key(b.type))
            # graph
            d = dist.get(c)
            X[i, _C_PROX] = 1.0 / (1.0 + d) if d is not None else 0.0
            if aa_norm:
                shared = (seed_nbrs - {c}) & (g.neighbors(c) - {q})
                X[i, _C_SHARED] = sum(aa_w[z] for z in shared) / aa_norm
            X[i, _C_DEGREE] = degree.get(c, 0) / max_deg if max_deg else 0.0
            direction = edge_direction(c, incoming, outgoing)
            if direction is not None:
                X[i, _C_PROP] = prop[direction]
            # text
            X[i, _C_NAME] = jaccard(seed_tok, name_tokens(b.name))
            cv = content_cache.get(c)
            if cv is None:
                cv = content_cache[c] = self.content.vector(b)
            X[i, _C_CONTENT] = cosine(seed_vec, cv)
            # locality
            X[i, _C_SAME_FILE] = float(b.file == seed.file)
            X[i, _C_SAME_DIR] = float(b.directory == seed.directory)
        X[:, _C_COUNT] = np.log1p(sup) / math.log1p(sup.max()) if sup.max() > 0 else 0.0
        X[:, _C_RECENCY] = _max_norm(rec)
        X[:, _C_CHURN] = _max_norm(churn)
        return X

    def _add_type_features(self, seed: Any, seed_ti: str, types: list[str]) -> np.ndarray:
        X = np.zeros((len(types), N_FEATURES), dtype=np.float64)
        seed_tok = name_tokens(seed.name)
        for i, t in enumerate(types):
            X[i, _C_TYPE] = self.miner.type_confidence(seed_ti, ADDED + t)
            X[i, _C_NAME] = jaccard(seed_tok, type_tokens(t))
        return X
