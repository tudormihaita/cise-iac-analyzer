"""Training data for the GA: candidate rows grouped by query, and query sampling.

Only queries with at least one true answer among their candidates are kept: for the others the
reciprocal rank is 0 whatever the weights, so they cannot influence the search.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tfimpact.evaluation.loader import RepoData
from tfimpact.features.vector import ADD_TYPE


@dataclass
class Batch:
    """Rows of a set of queries, contiguous per query."""

    X: np.ndarray  # rows x F, float32
    is_add: np.ndarray  # rows, float32 (1 = add-type candidate)
    y: np.ndarray  # rows, bool
    starts: np.ndarray  # first row of each query
    gid: np.ndarray  # rows -> query index
    qw: np.ndarray  # per-query weight (sums to 1; repos weighted equally)


@dataclass
class TrainSet:
    """All training queries of a set of repos (rows contiguous per query)."""

    X: np.ndarray
    is_add: np.ndarray
    y: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    q_repo: np.ndarray  # repo index of every query
    n_repos: int

    @classmethod
    def build(
        cls,
        repos: Sequence[RepoData],
        names: Sequence[str],
        permute_rng: np.random.Generator | None = None,
    ) -> TrainSet:
        """`permute_rng` shuffles the labels within each query (label-permutation check)."""
        Xs, adds, ys, qids, rep = [], [], [], [], []
        for i, d in enumerate(repos):
            f = d.features
            pos_q = f.query_id[f.label == 1].unique()
            f = f[f.query_id.isin(pos_q)].sort_values("query_id", kind="stable")
            Xs.append(f[list(names)].to_numpy(np.float32))
            adds.append((f.cand_kind == ADD_TYPE).to_numpy(np.float32))
            ys.append(f.label.to_numpy() == 1)
            qids.append(f.query_id.to_numpy(np.int64))
            rep.append(np.full(len(f), i, dtype=np.int64))
        X = np.vstack(Xs) if Xs else np.zeros((0, len(names)), np.float32)
        y = np.concatenate(ys) if ys else np.zeros(0, bool)
        is_add = np.concatenate(adds) if adds else np.zeros(0, np.float32)
        qid = np.concatenate(qids) if qids else np.zeros(0, np.int64)
        rr = np.concatenate(rep) if rep else np.zeros(0, np.int64)
        new = np.ones(len(qid), dtype=bool)
        new[1:] = (qid[1:] != qid[:-1]) | (rr[1:] != rr[:-1])
        starts = np.flatnonzero(new)
        ends = np.r_[starts[1:], len(qid)]
        if permute_rng is not None:
            gid = np.repeat(np.arange(len(starts)), ends - starts)
            order = np.lexsort((permute_rng.random(len(gid)), gid))
            y = y[order]
        return cls(X, is_add, y, starts, ends, rr[starts], len(repos))

    @property
    def n_queries(self) -> int:
        return len(self.starts)

    def sample(self, size: int | None, rng: np.random.Generator) -> Batch:
        """Equal number of queries per repo (all of a repo's queries if it has fewer);
        `size=None` takes every query."""
        if size is None:
            qs = np.arange(self.n_queries)
        else:
            per = max(1, size // max(self.n_repos, 1))
            parts = [
                rng.permutation(np.flatnonzero(self.q_repo == k))[:per] for k in range(self.n_repos)
            ]
            qs = np.sort(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        return self.batch(qs)

    def batch(self, qs: np.ndarray) -> Batch:
        lens = self.ends[qs] - self.starts[qs]
        rows = (
            np.concatenate(
                [np.arange(s, e) for s, e in zip(self.starts[qs], self.ends[qs], strict=True)]
            )
            if len(qs)
            else np.zeros(0, np.int64)
        )
        starts = (
            np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64) if len(qs) else np.zeros(0, np.int64)
        )
        counts = np.bincount(self.q_repo[qs], minlength=self.n_repos)
        qw = 1.0 / counts[self.q_repo[qs]]
        qw = qw / qw.sum() if len(qw) else qw
        gid = np.repeat(np.arange(len(qs)), lens)
        return Batch(self.X[rows], self.is_add[rows], self.y[rows], starts, gid, qw)
