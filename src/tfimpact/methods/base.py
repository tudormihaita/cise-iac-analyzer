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

    def fit(self, train: Sequence[RepoData], fold: int = -1) -> Method:
        """Learn parameters from training repos (no-op for non-learned methods).
        `fold` identifies the cross-validation fold (-1 = all data)."""
        return self

    @abstractmethod
    def rank(self, data: RepoData) -> Ranking: ...

    def rank_all(self, data: RepoData) -> list[Ranking]:
        """One ranking per trained model (e.g. one per GA seed); evaluated separately and
        averaged per repo."""
        return [self.rank(data)]

    def params(self) -> dict:
        """Learned / configured parameters, recorded with the results."""
        return {}
