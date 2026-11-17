from __future__ import annotations

from tfimpact.config import Config
from tfimpact.methods.base import Method


def make_method(name: str, cfg: Config) -> Method:
    """All methods by name: baselines and the GA variants."""
    from tfimpact.methods.baselines import make_baseline
    from tfimpact.methods.ga import VARIANTS, GAMethod

    if name in VARIANTS:
        return GAMethod(cfg, name)
    return make_baseline(name, cfg)
