"""Recommend co-changes for blocks of a local Terraform repository with the trained model.

1. Mine the repo's own history up to HEAD with the training pipeline (same rule miners, reference
   graph and indexes; last `mining.max_commits` commits; no queries). Nothing is cloned or sent
   anywhere, and no Terraform tooling is needed.
2. Seeds: one block (`--block`), or every block that differs between the working tree and HEAD
   (`--diff`), with the edited bodies, exactly like "the seed's new body" during training.
3. Candidates and features are built by the training code (`FeatureBuilder`); each candidate is
   scored with the model (w · x, plus the offset for add-type candidates). With several seeds a
   candidate keeps its best score (`recommend.combine: max`) or the sum (`sum`).
4. Two lists: existing blocks to modify, and block types to add.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from tfimpact.config import Config, FeaturesConfig
from tfimpact.constants import ADDR_SEP, MODIFIED
from tfimpact.features.vector import (
    ADD_TYPE,
    ADD_TYPE_PREFIX,
    COL,
    FeatureBuilder,
    validate_feature_names,
)
from tfimpact.processing.diff import ChangeItem, diff_blocks
from tfimpact.processing.gitio import worktree_tf_changes
from tfimpact.processing.hcl_parser import module_dir, parse_blocks
from tfimpact.processing.pipeline import RepoState, process_repo


class RecommendError(ValueError):
    """User-facing problem (unknown block, no edits, incompatible model, …)."""


@dataclass
class Model:
    features: list[str]
    weights: np.ndarray
    offset: float
    features_config: dict
    meta: dict

    @classmethod
    def load(cls, path: Path) -> Model:
        if not path.exists():
            raise RecommendError(f"model not found: {path} (run `tfimpact train`)")
        raw = json.loads(path.read_text())
        names = list(raw.get("features") or raw["weights"])
        try:
            validate_feature_names(names)
        except ValueError as e:
            raise RecommendError(f"model incompatible with this code version: {e}") from e
        w = np.array([raw["weights"][n] for n in names], dtype=np.float64)
        return cls(names, w, float(raw["offset"]), raw.get("features_config", {}), raw)


@dataclass
class Suggestion:
    candidate: str  # block address, or the block type for add-type suggestions
    kind: str  # "existing" | "add_type"
    file: str
    score: float
    contributions: list[tuple[str, float]]  # (feature, weight * value), largest first
    seed: str  # the seed this score came from


@dataclass
class Recommendation:
    repo: str
    head_sha: str
    n_commits: int
    seeds: list[tuple[str, str]]  # (address, action)
    modify: list[Suggestion] = field(default_factory=list)
    add_types: list[Suggestion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def s(x: Suggestion) -> dict:
            return {
                "candidate": x.candidate,
                "file": x.file,
                "score": round(x.score, 6),
                "seed": x.seed,
                "contributions": {k: round(v, 6) for k, v in x.contributions},
            }

        return {
            "repo": self.repo,
            "head_sha": self.head_sha,
            "n_commits": self.n_commits,
            "seeds": [{"address": a, "action": act} for a, act in self.seeds],
            "modify": [s(x) for x in self.modify],
            "add_types": [s(x) for x in self.add_types],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------- state and seeds


def mine_state(repo: Path, cfg: Config) -> RepoState:
    res = process_repo(repo, repo.name, cfg, blobless=False, rev="HEAD", compute_features=False)
    assert res.state is not None
    return res.state


def resolve_block(state: RepoState, spec: str) -> str:
    """Full address (`dir::resource.aws_s3_bucket.logs`) or a short form (`aws_s3_bucket.logs`,
    `resource.aws_s3_bucket.logs`, `module.vpc`, `var.region`) that matches exactly one block."""
    blocks = state.snap.blocks
    if spec in blocks:
        return spec
    suffixes = (f"{ADDR_SEP}{spec}", f"{ADDR_SEP}resource.{spec}")
    matches = sorted(a for a in blocks if a.endswith(suffixes))
    if not matches:
        raise RecommendError(f"no block matches {spec!r} at HEAD")
    if len(matches) > 1:
        listed = "\n  ".join(matches[:20])
        raise RecommendError(f"{spec!r} is ambiguous; use the full address:\n  {listed}")
    return matches[0]


def block_seed(state: RepoState, address: str) -> ChangeItem:
    """A block of HEAD as a seed (treated as modified, with its current body)."""
    b = state.snap.blocks[address]
    return ChangeItem(address, b.kind, b.type, b.name, MODIFIED, b.file, b, b)


def worktree_seeds(repo: Path, state: RepoState) -> tuple[list[ChangeItem], list[str]]:
    """Blocks added / modified / removed in the working tree compared with HEAD."""
    changed = worktree_tf_changes(repo)
    by_dir: dict[str, list[str]] = {}
    for path in changed:
        by_dir.setdefault(module_dir(path), []).append(path)
    items: list[ChangeItem] = []
    errors: list[str] = []
    for directory, paths in sorted(by_dir.items()):
        before = state.snap.dir_blocks(directory)
        edited = set(paths)
        after = {a: b for a, b in before.items() if b.file not in edited}
        for p in paths:
            f = repo / p
            if not f.exists():
                continue  # deleted file: its blocks are removed
            try:
                after.update(parse_blocks(f.read_text(encoding="utf-8", errors="replace"), p))
            except Exception as e:  # noqa: BLE001 - report and keep the HEAD version
                errors.append(f"{p}: cannot parse ({type(e).__name__}); file ignored")
                after.update({a: b for a, b in before.items() if b.file == p})
        items += diff_blocks(before, after)
    return items, errors


# --- scoring ---


def score_seeds(
    state: RepoState,
    items: list[ChangeItem],
    model: Model,
    features_cfg: FeaturesConfig,
    combine: str,
) -> tuple[list[Suggestion], list[Suggestion]]:
    builder = FeatureBuilder(
        features_cfg,
        state.snap,
        state.graph,
        state.tokens,
        state.miner,
        state.propagation,
        state.content,
    )
    cf = builder.build_commit(state.last_idx + 1, "worktree", items)
    cols = [COL[n] for n in model.features]
    seeds = {it.address for it in items}
    best: dict[str, Suggestion] = {}
    for q, X, ids, kinds, files in zip(
        cf.queries, cf.X, cf.cand_ids, cf.cand_kinds, cf.cand_files, strict=True
    ):
        Xm = X[:, cols].astype(np.float64)
        is_add = np.array([k == ADD_TYPE for k in kinds])
        contrib = Xm * model.weights
        score = contrib.sum(axis=1) + is_add * model.offset
        for i, cid in enumerate(ids):
            if cid in seeds:
                continue
            parts = sorted(
                ((model.features[j], float(contrib[i, j])) for j in np.flatnonzero(contrib[i])),
                key=lambda kv: -kv[1],
            )
            if is_add[i]:
                parts.append(("add_type_offset", model.offset))
            s = float(score[i])
            prev = best.get(cid)
            if prev is None or (combine == "max" and s > prev.score):
                name = cid[len(ADD_TYPE_PREFIX) :] if is_add[i] else cid
                kind = ADD_TYPE if is_add[i] else "existing"
                best[cid] = Suggestion(name, kind, files[i], s, parts, q["seed_address"])
            elif combine == "sum":
                prev.score += s
    ranked = sorted(best.values(), key=lambda x: (-x.score, x.candidate))
    return [x for x in ranked if x.kind != ADD_TYPE], [x for x in ranked if x.kind == ADD_TYPE]


def recommend(
    repo: Path,
    cfg: Config,
    model: Model,
    block: str | None = None,
    diff: bool = False,
    top: int | None = None,
) -> Recommendation:
    if bool(block) == bool(diff):
        raise RecommendError("choose exactly one of --block or --diff")
    if cfg.recommend.combine not in ("max", "sum"):
        raise RecommendError(f"recommend.combine must be max or sum, not {cfg.recommend.combine!r}")
    features_cfg = (
        replace(cfg.features, **model.features_config) if model.features_config else cfg.features
    )
    state = mine_state(repo, cfg)
    errors: list[str] = []
    if block:
        items = [block_seed(state, resolve_block(state, block))]
    else:
        items, errors = worktree_seeds(repo, state)
        if not items:
            raise RecommendError("no block differs between the working tree and HEAD")
    modify, add_types = score_seeds(state, items, model, features_cfg, cfg.recommend.combine)
    n = cfg.recommend.top if top is None else top
    return Recommendation(
        repo=str(repo),
        head_sha=state.head_sha,
        n_commits=state.last_idx + 1,
        seeds=[(it.address, it.action) for it in items],
        modify=modify[:n],
        add_types=add_types[:n],
        errors=errors,
    )
