"""Cross-validated evaluation of ranking methods (grouped k-fold by repository).

Repos come in two subsets (see `loader.dataset_repos`):
  * core: grouped k-fold; for every fold, learned methods are fitted on the core training
    repos and every method ranks the candidates of the fold's core test repos
  * add_only: never trained on; evaluated with every fold's model (learned methods: averaged
    over the folds) and reported as a separate held-out group

Metrics are computed per query, averaged per repo, and repos are weighted equally. Reported
per method, target ("all" = existing blocks + add-type candidates, "existing" = existing blocks
only) and subset (core / add_only / all): mean, SD over repos, SD over folds (core) and a
bootstrap 95% CI. Breakdowns, file-level metrics and the paired statistical comparisons use
the core repos (the population the methods are trained and cross-validated on).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tfimpact.config import Config, config_to_dict
from tfimpact.evaluation import stats as S
from tfimpact.evaluation.loader import ADD_ONLY, CORE, dataset_repos, load_repo, query_counts
from tfimpact.evaluation.metrics import (
    bootstrap_ci,
    compute_file_level_metrics,
    compute_query_metrics,
    metric_columns,
)
from tfimpact.evaluation.splits import load_or_make_folds, train_test
from tfimpact.methods.base import Method
from tfimpact.methods.baselines import make_baseline, required_features

TARGETS: dict[str, str | None] = {"all": None, "existing": "existing"}
SEED_ACTIONS = {"+": "added", "~": "modified", "-": "removed"}
STAT_METRICS = ("rr", "ap")
SUBSETS = (CORE, ADD_ONLY, "all")


@dataclass
class EvalResult:
    per_query: pd.DataFrame
    per_repo: pd.DataFrame
    summary: pd.DataFrame
    breakdowns: pd.DataFrame
    file_level: pd.DataFrame
    stats: pd.DataFrame
    meta: dict = field(default_factory=dict)


def run_evaluation(
    cfg: Config,
    methods: list[str],
    factory: Callable[[str, Config], Method] = make_baseline,
    progress: Callable[[str], None] | None = None,
) -> EvalResult:
    ev = cfg.evaluation
    ks = list(ev.k)
    repos = dataset_repos(cfg)
    kinds = dict(zip(repos.slug, repos.repo_kind.fillna(""), strict=True))
    subset = dict(zip(repos.slug, repos.subset, strict=True))
    core = [r for r in repos.slug if subset[r] == CORE]
    held_out = [r for r in repos.slug if subset[r] == ADD_ONLY]
    sizes = query_counts(cfg, core)
    folds = load_or_make_folds(cfg.paths.results / "folds.csv", sizes, ev.folds, cfg.seed)
    names = required_features(cfg, methods)
    cols = metric_columns(ks)

    per_query, file_rows, ceilings, params = [], [], {}, {}
    for fold in range(ev.folds):
        train, test = train_test(folds, fold)
        fitted = {m: factory(m, cfg) for m in methods}
        if any(obj.learns for obj in fitted.values()):
            train_data = [load_repo(cfg, r, names, kinds[r]) for r in train]
            for obj in fitted.values():
                if obj.learns:
                    obj.fit(train_data)
            del train_data
        params[fold] = {m: obj.params() for m, obj in fitted.items()}
        # held-out repos: learned methods in every fold (averaged later), others once
        jobs = [(r, list(fitted)) for r in test]
        jobs += [(r, [m for m, o in fitted.items() if o.learns or fold == 0]) for r in held_out]
        for r, ms in jobs:
            if not ms:
                continue
            if progress:
                progress(f"fold {fold + 1}/{ev.folds}  {r}")
            data = load_repo(cfg, r, names, kinds[r])
            q = data.queries
            ceilings[r] = float(q.n_labels_in_candidates.sum() / max(q.n_labels.sum(), 1))
            for m in ms:
                rk = fitted[m].rank(data)
                common = dict(
                    ks=ks, seed=cfg.seed, top_k_cut=rk.top_k, drop_warmup=False,
                    recommended=rk.recommended,
                )  # fmt: skip
                tags = dict(method=m, fold=fold, repo_kind=kinds[r], subset=subset[r])
                for target, kind in TARGETS.items():
                    pq = compute_query_metrics(data.features, q, rk.scores, kind=kind, **common)
                    per_query.append(pq.assign(target=target, **tags))
                if subset[r] == CORE:
                    fl = compute_file_level_metrics(data.features, q, rk.scores, **common)
                    file_rows.append(fl.assign(method=m, fold=fold))

    pq = pd.concat(per_query, ignore_index=True)
    per_repo = _per_repo(pq, cols, ceilings)
    core_pq = pq[pq.subset == CORE]
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "methods": methods,
        "n_repos": {CORE: len(core), ADD_ONLY: len(held_out)},
        "folds": {r: int(f) for r, f in folds.items()},
        "held_out": held_out,
        "method_params": params,
        "features": names,
        "config": config_to_dict(cfg),
    }
    return EvalResult(
        per_query=pq,
        per_repo=per_repo,
        summary=_summary(per_repo, cols, ev.bootstrap, cfg.seed),
        breakdowns=_breakdowns(core_pq, cols),
        file_level=_file_level(pd.concat(file_rows, ignore_index=True), cols),
        stats=_stats(per_repo[per_repo.subset == CORE], methods),
        meta=meta,
    )


# --- aggregation ---
def _per_repo(pq: pd.DataFrame, cols: list[str], ceilings: dict[str, float]) -> pd.DataFrame:
    """Per-repo means. A held-out repo evaluated by several fold models: mean over those
    evaluations (same queries each time); its fold is -1."""
    g = pq.groupby(["method", "target", "repo"], sort=True)
    out = g[cols].mean()
    out.insert(0, "n_queries", g.query_id.nunique())
    out.insert(0, "fold", g.fold.first())
    out.insert(0, "repo_kind", g.repo_kind.first())
    out.insert(0, "subset", g.subset.first())
    out = out.reset_index()
    out.loc[out.subset != CORE, "fold"] = -1
    out["recall_ceiling"] = out.repo.map(ceilings)
    return out


def _summary(per_repo: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for (m, t), g_all in per_repo.groupby(["method", "target"], sort=False):
        for sub in SUBSETS:
            g = g_all if sub == "all" else g_all[g_all.subset == sub]
            if g.empty:
                continue
            by_fold = g[g.fold >= 0].groupby("fold")[cols].mean() if sub == CORE else None
            for c in cols:
                lo, hi = bootstrap_ci(g[c].to_numpy(), n_boot, seed)
                rows.append(
                    {
                        "method": m,
                        "target": t,
                        "subset": sub,
                        "metric": c,
                        "mean": g[c].mean(),
                        "sd_repos": g[c].std(ddof=1),
                        "sd_folds": by_fold[c].std(ddof=1) if by_fold is not None else np.nan,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_repos": len(g),
                        "n_queries": int(g.n_queries.sum()),
                    }
                )
    return pd.DataFrame(rows)


def _breakdowns(pq: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Same metrics on query subsets; each subset is averaged per repo, then over repos."""
    pq = pq.assign(seed_action=pq.seed_action.map(SEED_ACTIONS).fillna(pq.seed_action))
    rows = []
    for by in ("seed_action", "repo_kind"):
        for (m, t, grp), g in pq.groupby(["method", "target", by], sort=True):
            rep = g.groupby("repo")[cols].mean()
            rows.append(
                {
                    "method": m,
                    "target": t,
                    "by": by,
                    "group": grp,
                    "n_repos": len(rep),
                    "n_queries": len(g),
                    **rep.mean().to_dict(),
                }
            )
    return pd.DataFrame(rows)


def _file_level(fl: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rep = fl.groupby(["method", "repo"])[cols].mean()
    out = rep.groupby("method").mean()
    out.insert(0, "n_repos", rep.groupby("method").size())
    return out.reset_index()


def _stats(per_repo: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Pairwise per-repo comparisons; Holm correction within each (target, metric) family."""
    rows = []
    for t in TARGETS:
        for metric in STAT_METRICS:
            wide = per_repo[per_repo.target == t].pivot(
                index="repo", columns="method", values=metric
            )
            fam = []
            for a, b in itertools.combinations(methods, 2):
                x, y = wide[a].to_numpy(), wide[b].to_numpy()
                v = S.a12(x, y)
                fam.append(
                    {
                        "target": t,
                        "metric": metric,
                        "method_a": a,
                        "method_b": b,
                        "mean_a": float(np.mean(x)),
                        "mean_b": float(np.mean(y)),
                        "diff": float(np.mean(x - y)),
                        "repos_a_better": int((x > y).sum()),
                        "repos_b_better": int((x < y).sum()),
                        "p_wilcoxon": S.wilcoxon_p(x, y),
                        "a12": v,
                        "a12_magnitude": S.a12_magnitude(v),
                    }
                )
            adj = S.holm([r["p_wilcoxon"] for r in fam])
            for r, p in zip(fam, adj, strict=True):
                r["p_holm"] = float(p)
            rows += fam
    return pd.DataFrame(rows)


# --- output ---
def save_results(res: EvalResult, out_dir: Path, code_version: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out_dir / "metrics.csv",
        "per_repo": out_dir / "per_repo.csv",
        "breakdowns": out_dir / "breakdowns.csv",
        "file_level": out_dir / "file_level.csv",
        "stats": out_dir / "stats.csv",
    }
    for attr, p in paths.items():
        getattr(res, attr).to_csv(p, index=False)
    pq_path = out_dir / "per_query.parquet"
    res.per_query.to_parquet(pq_path, index=False)
    meta_path = out_dir / "run.json"
    meta_path.write_text(json.dumps({**res.meta, "code_version": code_version}, indent=2))
    md_path = out_dir / "summary.md"
    md_path.write_text(summary_markdown(res))
    return [*paths.values(), pq_path, meta_path, md_path]


def summary_markdown(res: EvalResult) -> str:
    s = res.summary
    m = res.meta
    n = m["n_repos"]
    lines = [
        f"# Evaluation ({n[CORE]} core repos, grouped {len(set(m['folds'].values()))}-fold CV; "
        f"{n[ADD_ONLY]} add-only repos held out)",
        "",
        "Mean over repos (each repo weighted equally); MRR as mean ± SD over repos "
        "[bootstrap 95% CI]. Subsets: core = cross-validated repos (headline), add_only = never "
        "trained on, all = both. Other metrics' SDs/CIs and SD over folds: metrics.csv.",
        "",
    ]
    main = ["ap", "hit@1", "hit@5", "recall@10", "precision@5"]
    for t in TARGETS:
        lines += [
            f"## Targets: {t}",
            "",
            "| method | subset | MRR ± SD [95% CI] | MAP | Hit@1 | Hit@5 | Recall@10 | P@5 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for meth in m["methods"]:
            for sub in SUBSETS:
                g = s[(s.method == meth) & (s.target == t) & (s.subset == sub)]
                if g.empty:
                    continue
                g = g.set_index("metric")
                rr = g.loc["rr"]
                cells = [
                    f"{rr['mean']:.3f} ± {rr.sd_repos:.3f} [{rr.ci_low:.3f}, {rr.ci_high:.3f}]"
                ]
                cells += [f"{g.loc[c, 'mean']:.3f}" for c in main]
                lines.append(f"| {meth} | {sub} | " + " | ".join(cells) + " |")
        lines.append("")
    rep = res.per_repo.drop_duplicates("repo")
    lines += [
        "Candidate recall ceiling (mean over repos): "
        + ", ".join(
            f"{sub} {(rep if sub == 'all' else rep[rep.subset == sub]).recall_ceiling.mean():.3f}"
            for sub in SUBSETS
        ),
        "",
    ]
    if len(res.stats):
        lines += [
            "## Paired comparisons (core repos, per-repo MRR, all targets)",
            "",
            "| A vs B | mean A | mean B | A better / B better | p (Holm) | A12 |",
            "|---|---|---|---|---|---|",
        ]
        st = res.stats[(res.stats.target == "all") & (res.stats.metric == "rr")]
        for r in st.itertuples():
            lines.append(
                f"| {r.method_a} vs {r.method_b} | {r.mean_a:.3f} | {r.mean_b:.3f} | "
                f"{r.repos_a_better} / {r.repos_b_better} | {r.p_holm:.2g} | "
                f"{r.a12:.2f} ({r.a12_magnitude}) |"
            )
        lines.append("")
    return "\n".join(lines)
