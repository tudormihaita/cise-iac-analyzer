from __future__ import annotations

from tfimpact.constants import (
    SKIP_MERGE,
    SKIP_NO_TF,
    SKIP_START,
    SKIP_TOO_FEW,
    SKIP_TOO_MANY,
)


def classify_commit(
    *,
    is_start: bool,
    is_merge: bool,
    n_items: int,
    min_items: int,
    max_items: int,
    use_merge_commits: bool,
) -> tuple[bool, str | None]:
    """(usable, skip_reason)."""
    if is_start:
        return False, SKIP_START
    if is_merge and not use_merge_commits:
        return False, SKIP_MERGE
    if n_items == 0:
        return False, SKIP_NO_TF
    if n_items < min_items:
        return False, SKIP_TOO_FEW
    if n_items > max_items:
        return False, SKIP_TOO_MANY
    return True, None


def warmup_count(n_usable: int, warmup_frac: float) -> int:
    """Number of leading usable commits that only feed the miner (no queries)."""
    return int(n_usable * warmup_frac)
