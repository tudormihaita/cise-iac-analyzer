"""Load the mined per-repo tables for training and evaluation.

Only what the methods and metrics need is kept in memory: the selected feature columns
(float32), label, query id, candidate kind and file. Warm-up queries are dropped by default:
they are neither trained on nor evaluated (they still fed the rule miners during mining).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tfimpact.config import Config
from tfimpact.features.vector import validate_feature_names

ACCRETION_REASON = "add_only_frac>0.4"
_BASE_COLS = ["query_id", "cand_kind", "cand_file", "label"]
_QUERY_COLS = [
    "query_id",
    "idx",
    "seed_address",
    "seed_action",
    "seed_is_new",
    "n_labels",
    "n_labels_existing",
    "n_labels_add_type",
    "n_labels_in_candidates",
    "in_warmup",
]


@dataclass
class RepoData:
    repo: str
    kind: str
    features: pd.DataFrame  # one row per (query, candidate)
    queries: pd.DataFrame  # one row per query
    feature_names: list[str]

    def X(self, names: Sequence[str] | None = None) -> np.ndarray:
        return self.features[list(names or self.feature_names)].to_numpy(np.float32)

    @property
    def n_queries(self) -> int:
        return len(self.queries)


CORE = "core"
ADD_ONLY = "add_only"


def dataset_repos(cfg: Config) -> pd.DataFrame:
    """Rows of dataset.csv to evaluate on, with a `subset` column:

    * `core`: the included repos (cross-validated: trained on and tested on)
    * `add_only`: repos excluded *only* for their add-only fraction; never trained on, always
      evaluated as a separate held-out group
    """
    ds = pd.read_csv(cfg.paths.dataset_csv)
    core = ds.included.astype(bool)
    add_only = ~core & ds.exclude_reason.fillna("").eq(ACCRETION_REASON)
    out = ds[core | add_only].copy()
    out["subset"] = np.where(core[core | add_only], CORE, ADD_ONLY)
    return out.reset_index(drop=True)


def load_repo(
    cfg: Config, slug: str, feature_names: Sequence[str], kind: str = "", drop_warmup: bool = True
) -> RepoData:
    names = validate_feature_names(feature_names)
    d = cfg.paths.processed / slug
    q = pd.read_parquet(d / "queries.parquet", columns=_QUERY_COLS)
    if drop_warmup:
        q = q[~q.in_warmup.astype(bool)]
    q = q.reset_index(drop=True)
    q.insert(0, "repo", slug)
    f = pd.read_parquet(d / "features.parquet", columns=_BASE_COLS + names)
    f = f[f.query_id.isin(q.query_id)].reset_index(drop=True)
    f.insert(0, "repo", slug)
    f["cand_kind"] = f.cand_kind.astype(str)
    return RepoData(slug, kind, f, q, names)


def query_counts(cfg: Config, slugs: Sequence[str]) -> dict[str, int]:
    """Non-warm-up queries per repo (for balancing folds), without loading the features."""
    out = {}
    for s in slugs:
        q = pd.read_parquet(cfg.paths.processed / s / "queries.parquet", columns=["in_warmup"])
        out[s] = int((~q.in_warmup.astype(bool)).sum())
    return out
