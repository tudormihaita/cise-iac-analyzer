"""Genetic operators on a chromosome (F weights + add-type offset as the last gene).

Weights are non-negative and normalized to sum 1 (each weight = share of trust in a signal);
with `negative=True` they may lie in [-1, 1] and are normalized by sum |w| (suppressor check).
"""

from __future__ import annotations

import numpy as np


def repair(ind: np.ndarray, offset_range: float, negative: bool) -> np.ndarray:
    """Clip genes to their ranges and renormalize the weights."""
    w = np.clip(ind[:-1], -1.0 if negative else 0.0, 1.0)
    norm = np.abs(w).sum()
    w = w / norm if norm > 0 else np.full(len(w), 1.0 / len(w))
    return np.r_[w, np.clip(ind[-1], -offset_range, offset_range)]


def random_individual(
    rng: np.random.Generator, n_features: int, offset_range: float, negative: bool
) -> np.ndarray:
    """Dirichlet weights (non-negative, sum 1) and a uniform offset."""
    w = rng.dirichlet(np.ones(n_features))
    if negative:
        w = w * rng.choice([-1.0, 1.0], n_features)
    return repair(np.r_[w, rng.uniform(-offset_range, offset_range)], offset_range, negative)


def tournament(fit: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Index of the fittest of `k` individuals drawn without replacement."""
    idx = rng.choice(len(fit), size=k, replace=False)
    return int(idx[np.argmax(fit[idx])])


def blend_crossover(
    a: np.ndarray, b: np.ndarray, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """BLX-alpha: each child gene is uniform in the parents' range widened by alpha on each side."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ext = alpha * (hi - lo)
    return rng.uniform(lo - ext, hi + ext)


def gaussian_mutation(
    x: np.ndarray, rate: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Each gene gets N(0, sigma) noise with probability `rate`."""
    mask = rng.random(len(x)) < rate
    return x + mask * rng.normal(0.0, sigma, len(x))


def linear_sigma(start: float, end: float, generation: int, generations: int) -> float:
    """Mutation strength for generation 1..generations, decaying linearly from start to end."""
    return start + (end - start) * (generation - 1) / max(generations - 1, 1)
