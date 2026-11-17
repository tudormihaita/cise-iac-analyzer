"""Scoring and fitness: the GA's objective, vectorized over the whole population.

Chromosome = F weights + one add-type offset b (last gene):
    existing-block candidate:  score = w · x
    add-type candidate:        score = w · x + b
Fitness = mean reciprocal rank of the first true answer per query, weighted so that every repo
counts equally; ties use the expected rank under random tie-breaking.
"""

from __future__ import annotations

import numpy as np

from tfimpact.methods.ga.data import Batch


def scores(X: np.ndarray, is_add: np.ndarray, P: np.ndarray) -> np.ndarray:
    """rows x individuals; P = population (individuals x genes, last gene = offset)."""
    P = np.atleast_2d(P).astype(np.float32)
    return X @ P[:, :-1].T + is_add[:, None] * P[:, -1]


def fitness(P: np.ndarray, batch: Batch, chunk: int = 16) -> np.ndarray:
    """Weighted mean reciprocal rank of each individual (expected rank under random ties)."""
    P = np.atleast_2d(P)
    out = np.empty(len(P))
    if len(batch.starts) == 0:
        out[:] = 0.0
        return out
    for c in range(0, len(P), chunk):
        S = scores(batch.X, batch.is_add, P[c : c + chunk])
        best = np.maximum.reduceat(np.where(batch.y[:, None], S, -np.inf), batch.starts, axis=0)
        b = best[batch.gid]  # best true answer's score, per row
        greater = np.add.reduceat(np.greater(S, b).astype(np.int32), batch.starts, axis=0)
        ties = np.add.reduceat(np.equal(S, b).astype(np.int32), batch.starts, axis=0) - 1
        rank = 1.0 + greater + ties / 2.0
        out[c : c + chunk] = (batch.qw[:, None] / rank).sum(axis=0)
    return out
