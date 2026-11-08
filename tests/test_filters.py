from __future__ import annotations

import pytest

from tfimpact.constants import SKIP_MERGE, SKIP_NO_TF, SKIP_START, SKIP_TOO_FEW, SKIP_TOO_MANY
from tfimpact.processing.filters import classify_commit, warmup_count


def cc(**kw):
    args = dict(
        is_start=False, is_merge=False, n_items=5, min_items=2, max_items=30, use_merge_commits=True
    )
    args.update(kw)
    return classify_commit(**args)


def test_usable():
    assert cc() == (True, None)


def test_start_wins_over_everything():
    assert cc(is_start=True, is_merge=True, n_items=0) == (False, SKIP_START)


def test_merge_skipped_when_disabled():
    assert cc(is_merge=True, use_merge_commits=False) == (False, SKIP_MERGE)


def test_merge_usable_when_enabled():
    assert cc(is_merge=True, use_merge_commits=True) == (True, None)


def test_merge_disabled_takes_priority_over_item_count():
    assert cc(is_merge=True, use_merge_commits=False, n_items=0) == (False, SKIP_MERGE)


def test_merge_enabled_still_subject_to_item_bounds():
    assert cc(is_merge=True, n_items=0) == (False, SKIP_NO_TF)
    assert cc(is_merge=True, n_items=1) == (False, SKIP_TOO_FEW)
    assert cc(is_merge=True, n_items=31) == (False, SKIP_TOO_MANY)


def test_no_tf():
    assert cc(n_items=0) == (False, SKIP_NO_TF)


def test_too_few():
    assert cc(n_items=1) == (False, SKIP_TOO_FEW)


def test_too_many():
    assert cc(n_items=31) == (False, SKIP_TOO_MANY)


@pytest.mark.parametrize("n,ok", [(2, True), (30, True), (1, False), (31, False)])
def test_bounds_inclusive(n: int, ok: bool):
    assert cc(n_items=n)[0] is ok


@pytest.mark.parametrize(
    "n,frac,expected",
    [(10, 0.2, 2), (9, 0.2, 1), (4, 0.2, 0), (0, 0.2, 0), (10, 0.0, 0), (7, 0.5, 3)],
)
def test_warmup_count(n: int, frac: float, expected: int):
    assert warmup_count(n, frac) == expected
