"""Figures of an evaluation run (written to `<outputs>/figures/`).

* learned_weights_<method>.png  mean ± SD of each learned weight (and the add-type offset)
* per_repo_mrr.png              distribution of per-repo MRR per method (core repos)
* mrr_by_seed_action.png        MRR per seed action (added / modified / removed) per method
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIGURES = "figures"


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_learned_weights(lw: pd.DataFrame, method: str, path: Path) -> Path | None:
    plt = _plt()
    sub = lw[lw.method == method] if len(lw) else lw
    if plt is None or sub.empty:
        return None
    cols = [c for c in sub.columns if c not in ("method", "fold", "seed", "fitness")]
    stats = sub[cols].agg(["mean", "std"]).T
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = range(len(stats))
    ax.barh(list(y), stats["mean"], xerr=stats["std"], color="#4C72B0", alpha=0.85)
    ax.set_yticks(list(y), stats.index)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"weight (mean ± SD over {len(sub)} runs)")
    ax.set_title(f"Learned weights ({method})")
    return _save(fig, plt, path)


def plot_per_repo(per_repo: pd.DataFrame, methods: list[str], path: Path) -> Path | None:
    plt = _plt()
    core = per_repo[per_repo.subset == "core"]
    if plt is None or core.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, target in zip(axes, ("all", "existing"), strict=True):
        data = [core[(core.method == m) & (core.target == target)].rr.to_numpy() for m in methods]
        ax.boxplot(data, tick_labels=methods, showmeans=True)
        ax.set_title(f"targets: {target}")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("MRR per repo (core repos)")
    fig.suptitle("Per-repo MRR distribution")
    return _save(fig, plt, path)


def plot_seed_action(breakdowns: pd.DataFrame, methods: list[str], path: Path) -> Path | None:
    plt = _plt()
    b = breakdowns[(breakdowns.by == "seed_action") & (breakdowns.target == "all")]
    if plt is None or b.empty:
        return None
    groups = ["added", "modified", "removed"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [
            float(b[(b.method == m) & (b.group == g)].rr.iloc[0])
            if len(b[(b.method == m) & (b.group == g)])
            else 0.0
            for g in groups
        ]
        ax.bar([x + i * width for x in range(len(groups))], vals, width, label=m)
    ax.set_xticks([x + width * (len(methods) - 1) / 2 for x in range(len(groups))], groups)
    ax.set_ylabel("MRR (all targets, core repos)")
    ax.set_title("MRR by seed action")
    ax.legend(fontsize=8)
    return _save(fig, plt, path)


def _save(fig, plt, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figures(
    out_dir: Path,
    per_repo: pd.DataFrame,
    breakdowns: pd.DataFrame,
    learned: pd.DataFrame,
    methods: list[str],
) -> list[Path]:
    d = out_dir / FIGURES
    made = [
        plot_per_repo(per_repo, methods, d / "per_repo_mrr.png"),
        plot_seed_action(breakdowns, methods, d / "mrr_by_seed_action.png"),
    ]
    if len(learned):
        for m in learned.method.unique():
            made.append(plot_learned_weights(learned, m, d / f"learned_weights_{m}.png"))
    return [p for p in made if p is not None]
