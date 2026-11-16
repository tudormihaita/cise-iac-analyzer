"""Ranking metrics for the change-impact recommender.

Everything is vectorised: candidates of all queries are ranked in one
``np.lexsort`` (group, -score, random tiebreak) and the metrics are derived
from per-row ranks with ``np.bincount`` / ``np.minimum.at``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
_KEY_SHIFT = 40  # query_id < 2**40 is packed together with the repo code


def metric_columns(ks: Sequence[int] = DEFAULT_K_VALUES) -> list[str]:
    """Names of the per-query metric columns, in output order."""
    cols = ["rr", "ap"]
    for name in ("recall", "precision", "hit"):
        cols += [f"{name}@{k}" for k in ks]
    return cols


def _query_keys(repo: pd.Series, query_id: pd.Series, repos: pd.Index) -> np.ndarray:
    """Pack (repo, query_id) into one int64 key; unknown repos map to -1."""
    codes = repos.get_indexer(repo).astype(np.int64)
    key = (codes << _KEY_SHIFT) | query_id.to_numpy(np.int64)
    key[codes < 0] = -1
    return key


def rank_within_groups(gid: np.ndarray, scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """1-based rank of every row inside its group (higher score = better rank).

    Ties are broken by a random permutation drawn from ``default_rng(seed)``,
    so the result is reproducible for fixed inputs and seed. Returned array
    is aligned with the input rows.
    """
    n = len(gid)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tiebreak = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((tiebreak, -scores.astype(np.float64), gid))
    g_sorted = gid[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = g_sorted[1:] != g_sorted[:-1]
    start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n) - start + 1
    return ranks


def _metrics_from_ranks(
    gid: np.ndarray,
    rank: np.ndarray,
    pos: np.ndarray,
    denom: np.ndarray,
    ks: Sequence[int],
    top_k_cut: int | None,
    recommended: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-group metrics from per-row group ids, ranks and positive flags.

    A positive counts as found only if it is ranked within `top_k_cut` (if set) and, when
    `recommended` is given, the method actually recommends that row.
    """
    n_groups = len(denom)
    keep = pos if top_k_cut is None else pos & (rank <= top_k_cut)
    if recommended is not None:
        keep = keep & recommended
    g_pos, r_pos = gid[keep], rank[keep]
    first = np.full(n_groups, np.inf)
    np.minimum.at(first, g_pos, r_pos.astype(np.float64))
    out: dict[str, np.ndarray] = {"rr": np.where(np.isfinite(first), 1.0 / first, 0.0)}
    safe = np.maximum(denom, 1).astype(np.float64)
    out["ap"] = _average_precision(g_pos, r_pos, n_groups) / safe
    hits = {k: np.bincount(g_pos[r_pos <= k], minlength=n_groups) for k in ks}
    out.update({f"recall@{k}": hits[k] / safe for k in ks})
    out.update({f"precision@{k}": hits[k] / k for k in ks})
    out.update({f"hit@{k}": (hits[k] > 0).astype(np.float64) for k in ks})
    return out


def _average_precision(g_pos: np.ndarray, r_pos: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum over found positives of precision at their rank (divide by #targets for AP)."""
    if len(g_pos) == 0:
        return np.zeros(n_groups)
    order = np.lexsort((r_pos, g_pos))
    g_s, r_s = g_pos[order], r_pos[order].astype(np.float64)
    new = np.empty(len(g_s), dtype=bool)
    new[0] = True
    new[1:] = g_s[1:] != g_s[:-1]
    start = np.maximum.accumulate(np.where(new, np.arange(len(g_s)), 0))
    hits_so_far = np.arange(len(g_s)) - start + 1
    return np.bincount(g_s, weights=hits_so_far / r_s, minlength=n_groups)


def _prepare(
    features: pd.DataFrame,
    queries: pd.DataFrame,
    scores: np.ndarray,
    drop_warmup: bool,
    denom_col: str,
    kind: str | None,
    recommended: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    """Filter queries (warm-up, zero denominator) and map rows to query indices."""
    if len(scores) != len(features):
        raise ValueError("scores must be aligned with features rows")
    if recommended is not None and len(recommended) != len(features):
        raise ValueError("recommended must be aligned with features rows")
    q = queries
    if drop_warmup and "in_warmup" in q:
        q = q[~q["in_warmup"].astype(bool)]
    q = q[q[denom_col] > 0].reset_index(drop=True)
    repos = pd.Index(q["repo"].unique())
    qkeys = _query_keys(q["repo"], q["query_id"], repos)
    fkeys = _query_keys(features["repo"], features["query_id"], repos)
    gid = pd.Index(qkeys).get_indexer(fkeys).astype(np.int64)
    mask = gid >= 0
    if kind is not None:
        mask &= (features["cand_kind"] == kind).to_numpy()
    rec = None if recommended is None else np.asarray(recommended, dtype=bool)[mask]
    return q, features.loc[mask], gid[mask], np.asarray(scores)[mask], rec


def _frame(q: pd.DataFrame, metrics: dict[str, np.ndarray], denom: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"repo": q["repo"].to_numpy(), "query_id": q["query_id"].to_numpy()})
    for name, vals in metrics.items():
        df[name] = vals
    df["n_labels"] = denom
    df["seed_is_new"] = q["seed_is_new"].to_numpy(bool)
    if "seed_action" in q:
        df["seed_action"] = q["seed_action"].to_numpy()
    return df


def compute_query_metrics(
    features: pd.DataFrame,
    queries: pd.DataFrame,
    scores: np.ndarray,
    ks: Sequence[int] = DEFAULT_K_VALUES,
    seed: int = 0,
    top_k_cut: int | None = None,
    drop_warmup: bool = True,
    kind: str | None = None,
    recommended: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-query RR, AP, Recall@k, Precision@k and Hit@k.

    Args:
        features: one row per (query, candidate) with ``repo, query_id,
            cand_kind, label``.
        queries: one row per query (denominators come from here).
        scores: array aligned with ``features`` rows, higher = better.
        ks: cut-offs.
        seed: seed for the random tie-break.
        top_k_cut: if set, candidates ranked beyond it count as not recommended.
        recommended: optional boolean array aligned with ``features``; rows the method does
            not recommend (e.g. rules below a threshold) never count as found.
        drop_warmup: drop queries with ``in_warmup == True``.
        kind: ``None`` (all candidates, denominator ``n_labels``), ``"existing"``
            (existing candidates only, denominator ``n_labels_existing``) or
            ``"add_type"`` (denominator ``n_labels_add_type``). Queries whose
            denominator is 0 are dropped.

    Recall uses *all* true targets as denominator, so targets the candidate
    pre-filter missed count as misses. Queries without candidate rows get 0.
    """
    denom_col = {
        None: "n_labels",
        "existing": "n_labels_existing",
        "add_type": "n_labels_add_type",
    }[kind]
    q, f, gid, sc, rec = _prepare(
        features, queries, scores, drop_warmup, denom_col, kind, recommended
    )
    rank = rank_within_groups(gid, sc, seed)
    denom = q[denom_col].to_numpy(np.int64)
    m = _metrics_from_ranks(gid, rank, f["label"].to_numpy() == 1, denom, ks, top_k_cut, rec)
    return _frame(q, m, denom)


def compute_file_level_metrics(
    features: pd.DataFrame,
    queries: pd.DataFrame,
    scores: np.ndarray,
    ks: Sequence[int] = DEFAULT_K_VALUES,
    seed: int = 0,
    top_k_cut: int | None = None,
    drop_warmup: bool = True,
    recommended: np.ndarray | None = None,
) -> pd.DataFrame:
    """File-level metrics (for comparison with GA-IaCRec's granularity).

    Existing candidates are ranked, mapped to ``cand_file`` and only the first
    occurrence of each file is kept (add_type rows are dropped). A file is a
    true file if it contains at least one positive existing candidate.

    Limitation: the denominator is the number of distinct true files *among
    candidates*; the files of targets missed by the pre-filter are unknown
    here, so file-level recall is optimistic with respect to the ceiling.
    Queries with no true file among candidates are dropped.
    """
    q, f, gid, sc, rec = _prepare(
        features, queries, scores, drop_warmup, "n_labels_existing", "existing", recommended
    )
    rank = rank_within_groups(gid, sc, seed)
    df = pd.DataFrame(
        {
            "gid": gid,
            "rank": rank,
            "file": f["cand_file"].to_numpy(),
            "pos": f["label"].to_numpy() == 1,
            "rec": np.ones(len(gid), dtype=bool) if rec is None else rec,
        }
    ).sort_values(["gid", "rank"], kind="stable")
    df["file_pos"] = df.groupby(["gid", "file"])["pos"].transform("max")
    df = df[~df.duplicated(["gid", "file"])]
    true_files = df[df["file_pos"]].groupby("gid").size()
    denom_all = np.zeros(len(q), dtype=np.int64)
    denom_all[true_files.index.to_numpy()] = true_files.to_numpy()
    new_rank = df.groupby("gid").cumcount().to_numpy() + 1
    m = _metrics_from_ranks(
        df["gid"].to_numpy(),
        new_rank,
        df["file_pos"].to_numpy(),
        denom_all,
        ks,
        top_k_cut,
        df["rec"].to_numpy(),
    )
    out = _frame(q, m, denom_all)
    return out[denom_all > 0].reset_index(drop=True)


def aggregate(per_query: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per-repo mean over queries and overall mean over repos (equal weight)."""
    cols = [
        c
        for c in per_query.columns
        if c in ("rr", "ap") or c.split("@")[0] in ("recall", "precision", "hit")
    ]
    grp = per_query.groupby("repo", sort=True)
    per_repo = grp[cols].mean()
    per_repo.insert(0, "n_queries", grp.size())
    overall = per_repo[cols].mean()
    overall["n_repos"] = len(per_repo)
    overall["n_queries"] = per_repo["n_queries"].sum()
    return per_repo, overall


def breakdowns(
    features: pd.DataFrame,
    queries: pd.DataFrame,
    scores: np.ndarray,
    ks: Sequence[int] = DEFAULT_K_VALUES,
    seed: int = 0,
    top_k_cut: int | None = None,
    drop_warmup: bool = True,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Aggregated metrics for seed new / seed existing and existing / add_type targets.

    Keys: ``seed_new``, ``seed_existing``, ``targets_existing``, ``targets_add_type``.
    For the target breakdowns both candidate rows and the recall denominator
    are restricted to the given kind.
    """
    kw = {"ks": ks, "seed": seed, "top_k_cut": top_k_cut, "drop_warmup": drop_warmup}
    pq = compute_query_metrics(features, queries, scores, **kw)
    out: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    out["seed_new"] = aggregate(pq[pq["seed_is_new"]])
    out["seed_existing"] = aggregate(pq[~pq["seed_is_new"]])
    for kind in ("existing", "add_type"):
        out[f"targets_{kind}"] = aggregate(
            compute_query_metrics(features, queries, scores, kind=kind, **kw)
        )
    return out


def candidate_recall_ceiling(
    queries: pd.DataFrame, drop_warmup: bool = True
) -> tuple[pd.Series, float]:
    """Upper bound on recall: sum(n_labels_in_candidates) / sum(n_labels).

    Returns the per-repo ratio and the overall value (mean over repos).
    """
    q = queries
    if drop_warmup and "in_warmup" in q:
        q = q[~q["in_warmup"].astype(bool)]
    sums = q.groupby("repo")[["n_labels_in_candidates", "n_labels"]].sum()
    per_repo = sums["n_labels_in_candidates"] / sums["n_labels"].where(sums["n_labels"] > 0)
    return per_repo, float(per_repo.mean())


def bootstrap_ci(
    values: Sequence[float] | np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values` (e.g. per-repo MRR)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
