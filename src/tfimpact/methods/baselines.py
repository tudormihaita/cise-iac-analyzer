"""Non-learned comparison methods.

* `arm`: ROSE (Zimmermann et al., ICSE 2004) navigation settings: rank existing blocks by the
  confidence of the rule seed => candidate (ties by support), recommend only rules with
  support >= 1 and confidence >= 0.1, at most the top 10. When the seed is a new block (ROSE
  has no rule for it) the type-level rule confidence is used instead, with the same thresholds.
* `fixed_weights`: the plain average of the model features, i.e. the model without training
  (GA-IaCRec combines its heuristics with fixed equal weights).
* `random`: chance-level reference ranking.
"""

from __future__ import annotations

import zlib

import numpy as np

from tfimpact.config import ArmConfig, Config
from tfimpact.evaluation.loader import RepoData
from tfimpact.features.vector import EXISTING
from tfimpact.methods.base import Method, Ranking

ARM_FEATURES = ["hist_cochange_conf", "hist_cochange_count", "hist_type_cochange_conf"]


def lexicographic_score(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Dense rank of (primary, secondary) pairs: orders by primary, then secondary; equal pairs
    get equal scores (left to the metrics' seeded tie-break)."""
    if len(primary) == 0:
        return np.zeros(0)
    keys = np.rec.fromarrays(
        [primary.astype(np.float64), secondary.astype(np.float64)], names="a,b"
    )
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.float64)


class Arm(Method):
    name = "arm"

    def __init__(self, cfg: ArmConfig) -> None:
        if cfg.min_support != 1:
            # stored features keep confidence and a per-query normalised support, so only the
            # ROSE setting "support >= 1" (= any co-change, conf > 0) can be applied exactly
            raise ValueError("arm supports only min_support = 1")
        self.cfg = cfg

    def rank(self, data: RepoData) -> Ranking:
        f = data.features
        conf = f.hist_cochange_conf.to_numpy(np.float64)
        count = f.hist_cochange_count.to_numpy(np.float64)
        tconf = f.hist_type_cochange_conf.to_numpy(np.float64)
        q = data.queries
        new_seed = f.query_id.map(dict(zip(q.query_id, q.seed_is_new, strict=True)))
        new_seed = new_seed.to_numpy(bool)
        existing = (f.cand_kind == EXISTING).to_numpy()
        min_conf = self.cfg.min_confidence
        inst_ok = existing & ~new_seed & (conf > 0) & (conf >= min_conf)
        type_ok = new_seed & (tconf > 0) & (tconf >= min_conf)
        primary = np.where(new_seed, tconf, conf)
        secondary = np.where(new_seed, 0.0, count)
        recommended = inst_ok | type_ok
        scores = np.where(recommended, lexicographic_score(primary, secondary), -1.0)
        return Ranking(scores, recommended, self.cfg.top_k)

    def params(self) -> dict:
        return {
            "min_support": self.cfg.min_support,
            "min_confidence": self.cfg.min_confidence,
            "top_k": self.cfg.top_k,
        }


class FixedWeights(Method):
    name = "fixed_weights"

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)

    def rank(self, data: RepoData) -> Ranking:
        return Ranking(data.X(self.feature_names).mean(axis=1, dtype=np.float64))

    def params(self) -> dict:
        w = 1.0 / len(self.feature_names)
        return {"weights": dict.fromkeys(self.feature_names, w)}


class RandomRanking(Method):
    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def rank(self, data: RepoData) -> Ranking:
        rng = np.random.default_rng([self.seed, zlib.crc32(data.repo.encode())])
        return Ranking(rng.random(len(data.features)))

    def params(self) -> dict:
        return {"seed": self.seed}


def required_features(cfg: Config, methods: list[str]) -> list[str]:
    names = list(cfg.model.features)
    if "arm" in methods:
        names += [n for n in ARM_FEATURES if n not in names]
    return names


def make_baseline(name: str, cfg: Config) -> Method:
    if name == "arm":
        return Arm(cfg.baselines.arm)
    if name == "fixed_weights":
        return FixedWeights(cfg.model.features)
    if name == "random":
        return RandomRanking(cfg.seed)
    raise ValueError(f"unknown method {name!r} (known: arm, fixed_weights, random)")
