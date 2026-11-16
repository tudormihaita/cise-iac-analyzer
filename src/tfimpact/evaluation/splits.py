"""Train/test splits by repository (grouped k-fold cross-validation).

Every repository is in exactly one test fold, so a model is always tested on projects it never
saw during training. Folds are balanced by query count: repos are shuffled with a fixed seed,
then assigned largest-first to the fold with the fewest queries so far.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def make_folds(sizes: Mapping[str, int], k: int, seed: int) -> dict[str, int]:
    """repo -> fold index in [0, k), balanced by `sizes` (number of queries per repo)."""
    if k < 2 or k > len(sizes):
        raise ValueError(f"need 2 <= k <= number of repos ({len(sizes)}), got {k}")
    repos = sorted(sizes)
    rng = np.random.default_rng(seed)
    shuffled = [repos[i] for i in rng.permutation(len(repos))]
    order = sorted(shuffled, key=lambda r: -sizes[r])  # stable: ties keep the shuffled order
    totals = [0] * k
    counts = [0] * k
    folds: dict[str, int] = {}
    for r in order:
        f = min(range(k), key=lambda i: (totals[i], counts[i], i))
        folds[r] = f
        totals[f] += sizes[r]
        counts[f] += 1
    return folds


def train_test(folds: Mapping[str, int], fold: int) -> tuple[list[str], list[str]]:
    train = sorted(r for r, f in folds.items() if f != fold)
    test = sorted(r for r, f in folds.items() if f == fold)
    return train, test


def inner_split(
    repos: Sequence[str], sizes: Mapping[str, int], valid_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split training repos into (inner train, validation) with ~`valid_frac` of the queries
    in validation; used to tune settings without touching the test fold."""
    rng = np.random.default_rng(seed)
    order = [sorted(repos)[i] for i in rng.permutation(len(repos))]
    total = sum(sizes[r] for r in repos)
    valid: list[str] = []
    acc = 0
    for r in order:
        if acc >= valid_frac * total or len(valid) >= len(repos) - 1:
            break
        valid.append(r)
        acc += sizes[r]
    return sorted(set(repos) - set(valid)), sorted(valid)


def load_or_make_folds(path: Path, sizes: Mapping[str, int], k: int, seed: int) -> dict[str, int]:
    """Reuse the saved folds if they were made for the same repos, k and seed, so that every
    method is evaluated on identical splits; otherwise create and save them."""
    if path.exists():
        df = pd.read_csv(path)
        same = (
            set(df.repo) == set(sizes)
            and int(df.k.iloc[0]) == k
            and int(df.seed.iloc[0]) == seed
            and dict(zip(df.repo, df.n_queries, strict=True)) == dict(sizes)
        )
        if same:
            return dict(zip(df.repo, df.fold.astype(int), strict=True))
    folds = make_folds(sizes, k, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"repo": r, "fold": f, "n_queries": sizes[r], "k": k, "seed": seed}
            for r, f in folds.items()
        ]
    ).sort_values(["fold", "repo"]).to_csv(path, index=False)
    return folds
