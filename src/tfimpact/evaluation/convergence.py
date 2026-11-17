"""Convergence check: how many generations does the GA need?

Per fold, the training repos are split into inner-train / validation repos
(`splits.inner_split`). The GA runs on the inner-train repos with a generous budget
(`ga.convergence_generations`); after every generation the best fitness on its training sample
and the fitness of that best individual on *all* validation queries are recorded. Training
fitness that keeps rising while validation stays flat means extra generations only fit the
sample. Runs are saved like GA runs and reused when the command is repeated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from tfimpact.config import Config
from tfimpact.evaluation.loader import CORE, dataset_repos, load_repo, query_counts
from tfimpact.evaluation.splits import inner_split, load_or_make_folds, train_test
from tfimpact.methods.ga import (
    TrainSet,
    evolve,
    fitness,
    load_run,
    run_rng,
    save_run,
    settings_hash,
)


def run_convergence(
    cfg: Config, progress: Callable[[str], None] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ga, ev = cfg.ga, cfg.evaluation
    names = list(cfg.model.features)
    repos = dataset_repos(cfg)
    core = repos[repos.subset == CORE].slug.tolist()
    sizes = query_counts(cfg, core)
    folds = load_or_make_folds(cfg.paths.outputs / "folds.csv", sizes, ev.folds, cfg.seed)
    rows = []
    for fold in range(ev.folds):
        train, _ = train_test(folds, fold)
        inner, valid = inner_split(train, sizes, ga.valid_frac, cfg.seed + fold)
        h = settings_hash(
            {
                "variant": "convergence",
                "ga": asdict(ga),
                "features": names,
                "base_seed": cfg.seed,
                "inner": inner,
                "valid": valid,
            }
        )
        ts = vbatch = None
        for s in range(ga.convergence_seeds):
            path = cfg.paths.outputs / "ga_runs" / f"convergence_{h}" / f"fold{fold}_seed{s}.json"
            rec = load_run(path, h)
            if rec is None:
                if ts is None:
                    if progress:
                        progress(f"fold {fold + 1}/{ev.folds}: loading {len(train)} repos")
                    ts = TrainSet.build([load_repo(cfg, r, names) for r in inner], names)
                    vts = TrainSet.build([load_repo(cfg, r, names) for r in valid], names)
                    vbatch = vts.sample(None, np.random.default_rng(0))
                if progress:
                    progress(f"fold {fold + 1}/{ev.folds}: seed {s + 1}/{ga.convergence_seeds}")
                rng = run_rng(cfg.seed, "convergence", fold, s)
                batch = ts.sample(ga.fitness_sample, rng)
                vb = vbatch

                def on_gen(_g: int, best: np.ndarray, vb=vb) -> dict:
                    return {"valid": float(fitness(best[None], vb)[0])}

                res = evolve(
                    batch,
                    ga,
                    rng,
                    len(names),
                    generations=ga.convergence_generations,
                    on_generation=on_gen,
                )
                rec = {"settings_hash": h, "fold": fold, "seed": s, "history": res.history}
                save_run(path, rec)
            for hrow in rec["history"]:
                rows.append({"fold": fold, "seed": s, **hrow})
    curves = pd.DataFrame(rows)
    return curves, summarize(curves, ga.generations, ga.convergence_tol)


def summarize(curves: pd.DataFrame, budget: int, tol: float) -> pd.DataFrame:
    """Per run: generation where training fitness is within `tol` of its final value
    (plateau), the generation of best validation fitness, and the validation fitness at the
    configured budget vs at the end of the long run."""
    out = []
    for (fold, seed), g in curves.groupby(["fold", "seed"]):
        g = g.sort_values("gen")
        final = g.best.iloc[-1]
        plateau = int(g.gen[g.best >= final - tol].iloc[0])
        at_budget = g[g.gen == min(budget, int(g.gen.max()))]
        out.append(
            {
                "fold": fold,
                "seed": seed,
                "train_final": final,
                "plateau_gen": plateau,
                "valid_best_gen": int(g.gen.iloc[int(np.argmax(g.valid.to_numpy()))]),
                "valid_at_budget": float(at_budget.valid.iloc[0]),
                "valid_final": float(g.valid.iloc[-1]),
                "valid_max": float(g.valid.max()),
            }
        )
    return pd.DataFrame(out)


def plot_convergence(curves: pd.DataFrame, budget: int, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    agg = curves.groupby("gen")[["best", "valid"]].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label in (("best", "training sample (best)"), ("valid", "validation repos")):
        ax.plot(agg.index, agg[col]["mean"], label=label)
        ax.fill_between(agg.index, agg[col]["min"], agg[col]["max"], alpha=0.2)
    ax.axvline(budget, color="grey", ls="--", lw=1, label=f"configured budget ({budget})")
    ax.set_xlabel("generation")
    ax.set_ylabel("MRR (fitness)")
    ax.set_title("GA convergence (mean and range over folds and seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
