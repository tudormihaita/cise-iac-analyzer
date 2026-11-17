"""The GA as an evaluation method (fit per fold, rank test repos).

Variants (same code):
    ga             the method; with `ga.tune` it first picks its configuration per fold by the
                   nested grid search (`grid.tune`), then trains `evaluation.seeds` runs with it
    random_search  same number of fitness evaluations, chromosomes drawn at random
    ga_permuted    labels shuffled within each training query (leakage check)
    ga_negative    negative weights allowed (suppressor check)

Options used by the experiments (ablation, learning curve), each changing the settings hash:
    features    learn weights over a subset of the model features only
    fixed       {fold: configuration id}: use that grid configuration, no tuning
    train_frac  run s trains on a random share of the fold's training repos; the repos come
                from one fixed permutation per (fold, s), so smaller shares are subsets of
                larger ones
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from tfimpact.config import Config
from tfimpact.evaluation.loader import RepoData
from tfimpact.features.vector import ADD_TYPE
from tfimpact.methods.base import Method, Ranking
from tfimpact.methods.ga.data import TrainSet
from tfimpact.methods.ga.fitness import scores
from tfimpact.methods.ga.grid import config_by_id, config_id, tune
from tfimpact.methods.ga.runs import load_run, run_rng, save_run, settings_hash
from tfimpact.methods.ga.search import evolve, random_search

VARIANTS = ("ga", "random_search", "ga_permuted", "ga_negative")
CHECK_VARIANTS = ("ga_permuted", "ga_negative")


class GAMethod(Method):
    learns = True

    def __init__(
        self,
        cfg: Config,
        variant: str = "ga",
        *,
        name: str | None = None,
        features: Sequence[str] | None = None,
        fixed: dict[int, str] | None = None,
        train_frac: float | None = None,
        n_seeds: int | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown GA variant {variant!r}")
        if train_frac is not None and not 0 < train_frac <= 1:
            raise ValueError(f"train_frac must be in (0, 1], not {train_frac}")

        self.variant = variant
        self.name = name or variant
        self.cfg = cfg
        self.names = list(cfg.model.features if features is None else features)
        if not self.names:
            raise ValueError("no features to learn weights for")

        default_seeds = cfg.ga.check_seeds if variant in CHECK_VARIANTS else cfg.evaluation.seeds
        self.n_seeds = n_seeds or default_seeds
        self.fixed = fixed
        self.train_frac = train_frac
        self.runs: list[dict] = []
        self.run_dir = cfg.paths.outputs / "ga_runs"
        self.ga_cfg = cfg.ga  # configuration of the final runs (the tuned one for `ga`)
        self.tuning: list[dict] = []
        self.chosen: str | None = None

    def _payload(self, train_repos: list[str]) -> dict:
        p = {
            "variant": self.variant,
            "ga": asdict(self.ga_cfg),
            "features": self.names,
            "base_seed": self.cfg.seed,
            "train_repos": sorted(train_repos),
        }
        if self.train_frac is not None:
            p["train_frac"] = self.train_frac
        return p

    def train_subset(self, train: Sequence[RepoData], fold: int, seed: int) -> list[RepoData]:
        """The training repos of run `seed` (all of them unless `train_frac` is set)."""
        if self.train_frac is None:
            return list(train)
        ordered = sorted(train, key=lambda d: d.repo)
        perm = run_rng(self.cfg.seed, "train_subset", fold, seed).permutation(len(ordered))
        k = max(1, round(self.train_frac * len(ordered)))
        return [ordered[i] for i in perm[:k]]

    def fit(self, train: Sequence[RepoData], fold: int = -1) -> GAMethod:
        self.ga_cfg, self.tuning, self.chosen = self.cfg.ga, [], None

        if self.fixed is not None:
            self.ga_cfg = config_by_id(self.cfg.ga, self.fixed[fold])
            self.chosen = self.fixed[fold]
        elif self.variant == "ga" and self.cfg.ga.tune:
            self.ga_cfg, self.tuning = tune(
                train, self.names, self.cfg.ga, self.cfg.seed, fold, self.run_dir
            )
            self.chosen = config_id(self.ga_cfg)

        h = settings_hash(self._payload([d.repo for d in train]))
        out_dir = self.run_dir / f"{self.name}_{h}"
        self.runs, ts = [], None

        for s in range(self.n_seeds):
            path = out_dir / f"fold{fold}_seed{s}.json"
            rec = load_run(path, h)
            if rec is None:
                subset = self.train_subset(train, fold, s)
                if ts is None or self.train_frac is not None:
                    perm = run_rng(self.cfg.seed, "permute", fold, 0)
                    ts = TrainSet.build(
                        subset, self.names, perm if self.variant == "ga_permuted" else None
                    )
                rec = self._run(ts, fold, s, h)
                rec["n_train_repos"] = len(subset)
                save_run(path, rec)
            self.runs.append(rec)
        return self

    def _run(self, ts: TrainSet, fold: int, seed: int, h: str) -> dict:
        rng = run_rng(self.cfg.seed, self.name, fold, seed)
        batch = ts.sample(self.ga_cfg.fitness_sample, rng)
        F = len(self.names)
        if self.variant == "random_search":
            res = random_search(batch, self.ga_cfg, rng, F)
        else:
            res = evolve(batch, self.ga_cfg, rng, F, negative=self.variant == "ga_negative")
        return {
            "variant": self.name,
            "fold": fold,
            "seed": seed,
            "settings_hash": h,
            "weights": dict(zip(self.names, map(float, res.best[:-1]), strict=True)),
            "offset": float(res.best[-1]),
            "fitness": res.best_fitness,
            "evaluations": res.evaluations,
            "config": config_id(self.ga_cfg),
            "n_train_queries": int(len(batch.starts)),
            "history": res.history,
        }

    def chromosome(self, rec: dict) -> np.ndarray:
        return np.r_[[rec["weights"][n] for n in self.names], rec["offset"]]

    def rank_all(self, data: RepoData) -> list[Ranking]:
        if not self.runs:
            raise RuntimeError("fit() first")
        X = data.X(self.names)
        is_add = (data.features.cand_kind == ADD_TYPE).to_numpy(np.float32)
        S = scores(X, is_add, np.array([self.chromosome(r) for r in self.runs]))
        return [Ranking(S[:, i].astype(np.float64)) for i in range(S.shape[1])]

    def rank(self, data: RepoData) -> Ranking:
        return self.rank_all(data)[0]

    def params(self) -> dict:
        keys = ("fold", "seed", "weights", "offset", "fitness", "evaluations", "n_train_repos")
        out: dict = {
            "config": config_id(self.ga_cfg),
            "runs": [{k: r[k] for k in keys if k in r} for r in self.runs],
        }
        if self.chosen is not None:
            out["chosen"] = self.chosen
        if self.tuning:
            out["tuning"] = self.tuning
        return out
