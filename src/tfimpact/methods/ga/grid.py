"""Grid search over GA settings (nested inside the cross-validation).

For one fold: the fold's training repos are split into inner-train (75 %) and validation (25 %)
repos; every configuration of `ga.grid` is trained `grid.seeds` times on the inner-train repos
and scored on all validation queries; the configuration with the best mean validation MRR wins
(the first in grid order on ties). Test repos are never involved.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from tfimpact.config import GAConfig
from tfimpact.evaluation.loader import RepoData
from tfimpact.evaluation.splits import inner_split
from tfimpact.methods.ga.data import TrainSet
from tfimpact.methods.ga.fitness import fitness
from tfimpact.methods.ga.runs import load_run, run_rng, save_run, settings_hash
from tfimpact.methods.ga.search import evolve

log = logging.getLogger(__name__)


def config_id(c: GAConfig) -> str:
    return f"p{c.population}g{c.generations}_m{c.mutation_rate}_c{c.crossover_rate}"


def grid_configs(base: GAConfig) -> list[GAConfig]:
    """All grid configurations; every other setting is taken from `base`."""
    g = base.grid
    return [
        replace(base, population=p, generations=n, mutation_rate=m, crossover_rate=c)
        for p, n in g.pop_gen
        for m in g.mutation_rate
        for c in g.crossover_rate
    ]


def config_by_id(base: GAConfig, cid: str) -> GAConfig:
    """The grid configuration with this id (as written in `grid.csv`)."""
    for c in grid_configs(base):
        if config_id(c) == cid:
            return c
    raise ValueError(f"{cid!r} is not a configuration of the current grid")


def tune(
    train: Sequence[RepoData],
    names: list[str],
    base: GAConfig,
    base_seed: int,
    fold: int,
    run_dir: Path,
) -> tuple[GAConfig, list[dict]]:
    """Best configuration for this fold, and one record per grid run."""
    sizes = {d.repo: d.n_queries for d in train}
    inner, valid = inner_split(sorted(sizes), sizes, base.valid_frac, base_seed + fold)
    by_repo = {d.repo: d for d in train}
    h = settings_hash(
        {
            "variant": "grid",
            "ga": asdict(base),
            "features": names,
            "base_seed": base_seed,
            "inner": inner,
            "valid": valid,
        }
    )
    out_dir = run_dir / f"grid_{h}"
    F = len(names)
    ts = vbatch = None
    configs = grid_configs(base)
    records: list[dict] = []
    for i, gc in enumerate(configs):
        cid = config_id(gc)
        for s in range(base.grid.seeds):
            path = out_dir / f"fold{fold}_{cid}_seed{s}.json"
            rec = load_run(path, h)
            if rec is None:
                if ts is None:
                    ts = TrainSet.build([by_repo[r] for r in inner], names)
                    vts = TrainSet.build([by_repo[r] for r in valid], names)
                    vbatch = vts.sample(None, np.random.default_rng(0))
                log.info("fold %d: grid %d/%d %s seed %d", fold, i + 1, len(configs), cid, s)
                rng = run_rng(base_seed, f"grid/{cid}", fold, s)
                res = evolve(ts.sample(gc.fitness_sample, rng), gc, rng, F)
                rec = {
                    "settings_hash": h,
                    "fold": fold,
                    "config": cid,
                    "seed": s,
                    "train_fitness": res.best_fitness,
                    "valid": float(fitness(res.best[None], vbatch)[0]),
                    "weights": dict(zip(names, map(float, res.best[:-1]), strict=True)),
                    "offset": float(res.best[-1]),
                }
                save_run(path, rec)
            records.append(
                {k: rec[k] for k in ("fold", "config", "seed", "train_fitness", "valid")}
            )
    means: dict[str, list[float]] = {}
    for r in records:
        means.setdefault(r["config"], []).append(r["valid"])
    best = max(configs, key=lambda c: float(np.mean(means[config_id(c)])))
    log.info("fold %d: chosen configuration %s", fold, config_id(best))
    return best, records
