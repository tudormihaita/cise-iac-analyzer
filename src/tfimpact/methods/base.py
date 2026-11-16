"""Common interface of all ranking methods (baselines, GA)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tfimpact.evaluation.loader import RepoData


@dataclass
class Ranking:
    """Scores aligned with `RepoData.features` rows (higher = better).

    `recommended` marks rows the method actually suggests (None = all); `top_k` limits the
    suggestions per query (None = the full ranking).
    """

    scores: np.ndarray
    recommended: np.ndarray | None = None
    top_k: int | None = None


class Method(ABC):
    name: str = ""
    learns: bool = False

    def fit(self, train: Sequence[RepoData]) -> Method:
        """Learn parameters from training repos (no-op for non-learned methods)."""
        return self

    @abstractmethod
    def rank(self, data: RepoData) -> Ranking: ...

    def params(self) -> dict:
        """Learned / configured parameters, recorded with the results."""
        return {}
