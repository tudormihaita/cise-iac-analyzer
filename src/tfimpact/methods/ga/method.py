"""The GA as an evaluation method (fit per fold, rank test repos).

Variants (same code):
    ga             the method; with `ga.tune` it first picks its configuration per fold by the
                   nested grid search (`grid.tune`), then trains `evaluation.seeds` runs with it
    random_search  same number of fitness evaluations, chromosomes drawn at random
    ga_permuted    labels shuffled within each training query (leakage check)
    ga_negative    negative weights allowed (suppressor check)
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
from tfimpact.methods.ga.grid import config_id, tune
from tfimpact.methods.ga.runs import load_run, run_rng, save_run, settings_hash
from tfimpact.methods.ga.search import evolve, random_search

VARIANTS = ("ga", "random_search", "ga_permuted", "ga_negative")
CHECK_VARIANTS = ("ga_permuted", "ga_negative")


class GAMethod(Method):
    learns = True

    def __init__(self, cfg: Config, variant: str = "ga") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown GA variant {variant!r}")
        self.name = variant
        self.cfg = cfg
        self.names = list(cfg.model.features)
        self.n_seeds = cfg.ga.check_seeds if variant in CHECK_VARIANTS else cfg.evaluation.seeds
        self.runs: list[dict] = []
        self.run_dir = cfg.paths.outputs / "ga_runs"
        self.ga_cfg = cfg.ga  # configuration of the final runs (the tuned one for `ga`)
        self.tuning: list[dict] = []
        self.chosen: str | None = None

    def _payload(self, train_repos: list[str]) -> dict:
        return {
            "variant": self.name,
            "ga": asdict(self.ga_cfg),
            "features": self.names,
            "base_seed": self.cfg.seed,
            "train_repos": sorted(train_repos),
        }

    def fit(self, train: Sequence[RepoData], fold: int = -1) -> GAMethod:
        self.ga_cfg, self.tuning, self.chosen = self.cfg.ga, [], None
        if self.name == "ga" and self.cfg.ga.tune:
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
                if ts is None:
                    perm = run_rng(self.cfg.seed, "permute", fold, 0)
                    ts = TrainSet.build(
                        train, self.names, perm if self.name == "ga_permuted" else None
                    )
                rec = self._run(ts, fold, s, h)
                save_run(path, rec)
            self.runs.append(rec)
        return self

    def _run(self, ts: TrainSet, fold: int, seed: int, h: str) -> dict:
        rng = run_rng(self.cfg.seed, self.name, fold, seed)
        batch = ts.sample(self.ga_cfg.fitness_sample, rng)
        F = len(self.names)
        if self.name == "random_search":
            res = random_search(batch, self.ga_cfg, rng, F)
        else:
            res = evolve(batch, self.ga_cfg, rng, F, negative=self.name == "ga_negative")
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
        out: dict = {
            "config": config_id(self.ga_cfg),
            "runs": [
                {k: r[k] for k in ("fold", "seed", "weights", "offset", "fitness", "evaluations")}
                for r in self.runs
            ],
        }
        if self.tuning:
            out["chosen"] = self.chosen
            out["tuning"] = self.tuning
        return out
