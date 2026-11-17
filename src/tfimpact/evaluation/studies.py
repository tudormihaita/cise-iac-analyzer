"""Ablation and learning curve: GA variants evaluated by the cross-validation runner.

Every variant uses, in each fold, the configuration the grid search chose for that fold in the
main run (no re-tuning) and `experiments.seeds` runs. The full GA (`ga`) has exactly the
settings of the main run, so its runs are loaded from the main run's files, not re-trained.

* ablation: one GA per feature family (features whose name starts with `<family>_`)
* learning curve: GA trained on a share of each fold's training repos (`train_frac`); the
  full GA is the 100 % point
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from tfimpact.config import Config
from tfimpact.evaluation.experiments import EvalResult, learned_weights
from tfimpact.evaluation.loader import CORE
from tfimpact.methods.base import Method
from tfimpact.methods.ga import GAMethod

FULL = "ga"
REPORTED = ["rr", "ap", "hit@1", "hit@5", "recall@10"]


def family_features(names: Sequence[str], family: str) -> list[str]:
    return [n for n in names if n.startswith(f"{family}_")]


def ablation_specs(cfg: Config) -> dict[str, dict]:
    """Method name -> GAMethod options; the full GA first."""
    specs: dict[str, dict] = {FULL: {}}
    for fam in cfg.experiments.ablation:
        feats = family_features(cfg.model.features, fam)
        if not feats:
            raise ValueError(f"no model feature starts with {fam}_")
        specs[f"ga_{fam}"] = {"features": feats}
    return specs


def lc_name(frac: float) -> str:
    return f"ga_lc{round(frac * 100):d}"


def learning_curve_specs(cfg: Config) -> dict[str, dict]:
    """Method name -> GAMethod options, smallest share first; the full GA last."""
    fracs = sorted(set(cfg.experiments.learning_curve))
    if any(not 0 < f < 1 for f in fracs):
        raise ValueError("learning-curve shares must be in (0, 1); 1.0 is the full GA")
    specs: dict[str, dict] = {lc_name(f): {"train_frac": f} for f in fracs}
    specs[FULL] = {}
    return specs


def factory(
    specs: dict[str, dict], chosen: dict[int, str], seeds: int
) -> Callable[[str, Config], Method]:
    def make(name: str, cfg: Config) -> Method:
        return GAMethod(cfg, "ga", name=name, fixed=chosen, n_seeds=seeds, **specs[name])

    return make


def chosen_configs(grid: pd.DataFrame) -> dict[int, str]:
    """{fold: configuration id} chosen by the grid search (rows of `grid.csv`)."""
    rows = grid[grid.chosen.astype(bool)].drop_duplicates("fold")
    return {int(f): str(c) for f, c in zip(rows.fold, rows.config, strict=True)}


# --- result tables ---
def _core_metrics(res: EvalResult) -> pd.DataFrame:
    s = res.summary[(res.summary.subset == CORE) & res.summary.metric.isin(REPORTED)]
    wide = s.pivot_table(index=["method", "target"], columns="metric", values="mean")
    rr = s[s.metric == "rr"].set_index(["method", "target"])
    wide["ci_low"], wide["ci_high"] = rr.ci_low, rr.ci_high
    wide["sd_repos"], wide["sd_runs"] = rr.sd_repos, rr.sd_runs
    return wide.rename(columns={"rr": "mrr", "ap": "map"}).reset_index()


def ablation_table(res: EvalResult, specs: dict[str, dict], names: Sequence[str]) -> pd.DataFrame:
    """One row per (variant, target): core metrics and the paired comparison full vs variant
    on per-repo MRR (`ri_full` = relative improvement of the full GA over the variant)."""
    m = _core_metrics(res)
    m.insert(1, "n_features", m.method.map(lambda k: len(specs[k].get("features", names))))
    st = res.stats[(res.stats.metric == "rr") & (res.stats.method_a == FULL)]
    st = st.rename(
        columns={
            "method_b": "method",
            "diff": "full_minus_variant",
            "ri": "ri_full",
            "repos_a_better": "repos_full_better",
            "repos_b_better": "repos_variant_better",
        }
    )
    keep = [
        "method",
        "target",
        "full_minus_variant",
        "ri_full",
        "repos_full_better",
        "repos_variant_better",
        "p_holm",
        "a12",
        "a12_magnitude",
    ]
    out = m.merge(st[keep], on=["method", "target"], how="left")
    out["_order"] = out.method.map({k: i for i, k in enumerate(specs)})
    out = out.sort_values(["target", "_order"]).drop(columns="_order")
    return out.rename(columns={"method": "variant"}).reset_index(drop=True)


def learning_curve_runs(res: EvalResult, specs: dict[str, dict]) -> pd.DataFrame:
    """One row per (share, fold, run, target): test MRR over the fold's core test repos,
    training-fitness MRR and the number of training repos of that run."""
    pq = res.per_query[res.per_query.subset == CORE]
    rep = pq.groupby(["method", "target", "fold", "run", "repo"]).rr.mean()
    runs = rep.groupby(["method", "target", "fold", "run"]).mean().rename("test_mrr").reset_index()
    lw = learned_weights(res.meta.get("method_params", {}))
    cols = ["method", "fold", "seed", "fitness"] + (
        ["n_train_repos"] if "n_train_repos" in lw.columns else []
    )
    lw = lw[cols].rename(columns={"seed": "run", "fitness": "train_fitness"})
    out = runs.merge(lw, on=["method", "fold", "run"], how="left")
    if "n_train_repos" not in out.columns:
        out["n_train_repos"] = np.nan
    folds = pd.Series(res.meta["folds"])
    full_n = out.fold.map(lambda f: int((folds != f).sum()))
    out["n_train_repos"] = out.n_train_repos.fillna(full_n)
    out.insert(1, "train_frac", out.method.map(lambda k: specs[k].get("train_frac", 1.0)))
    return out.sort_values(["target", "train_frac", "fold", "run"]).reset_index(drop=True)


def learning_curve_table(res: EvalResult, specs: dict[str, dict]) -> pd.DataFrame:
    """One row per (share, target): core test metrics (mean over repos, CI, SD over runs and
    folds) and the mean training-fitness MRR / number of training repos."""
    m = _core_metrics(res)
    runs = learning_curve_runs(res, specs)
    tr = runs.groupby(["method", "target"]).agg(
        train_fitness=("train_fitness", "mean"),
        train_fitness_sd=("train_fitness", "std"),
        n_train_repos=("n_train_repos", "mean"),
    )
    by_fold = runs.groupby(["method", "target", "fold"]).test_mrr.mean()
    tr["sd_folds"] = by_fold.groupby(["method", "target"]).std(ddof=1)
    out = m.merge(tr.reset_index(), on=["method", "target"], how="left")
    out.insert(1, "train_frac", out.method.map(lambda k: specs[k].get("train_frac", 1.0)))
    return out.sort_values(["target", "train_frac"]).reset_index(drop=True)


def study_markdown(title: str, note: str, table: pd.DataFrame, cols: dict[str, str]) -> str:
    """A markdown table of the selected columns (`cols`: column -> header). Floats: 3 decimals;
    `p_*` in scientific notation, `ri*` and `train_frac` as percentages, `repos_*` counts as
    integers, `n_*` with one decimal."""
    lines = [f"# {title}", "", note, "", "| " + " | ".join(cols.values()) + " |"]
    lines.append("|" + "---|" * len(cols))
    for r in table.to_dict("records"):
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                cells.append("")
            elif c.startswith("p_"):
                cells.append(f"{v:.2g}")
            elif c.startswith("ri"):
                cells.append(f"{v:+.1%}")
            elif c == "train_frac":
                cells.append(f"{v:.0%}")
            elif c.startswith("repos_"):
                cells.append(f"{v:.0f}")
            elif c.startswith("n_") and isinstance(v, float):
                cells.append(f"{v:.1f}")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
