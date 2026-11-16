"""Paired comparison of two methods over repositories.

* Wilcoxon signed-rank test on per-repo values (paired): is one method better across repos
  more consistently than chance would explain?
* Vargha-Delaney A12: probability that a random repo value of method A exceeds one of method B
  (0.5 = no difference); magnitude thresholds 0.56 / 0.64 / 0.71 (Vargha & Delaney, 2000).
* Holm step-down correction of p-values for multiple comparisons.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when all paired differences are zero."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("paired samples must have equal length")
    if len(x) == 0 or np.allclose(x - y, 0):
        return 1.0
    return float(sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)


def a12(a: Sequence[float], b: Sequence[float]) -> float:
    """P(A > B) + 0.5 * P(A = B) over all pairs of values."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def a12_magnitude(v: float) -> str:
    d = abs(v - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj
