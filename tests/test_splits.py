from __future__ import annotations

import pytest

from tfimpact.evaluation.splits import inner_split, load_or_make_folds, make_folds, train_test

SIZES = {
    f"r{i}": s for i, s in enumerate([5000, 3000, 2500, 900, 800, 700, 400, 300, 200, 100, 90])
}


def test_every_repo_tested_exactly_once():
    folds = make_folds(SIZES, 5, seed=42)
    assert set(folds) == set(SIZES)
    tested = [r for f in range(5) for r in train_test(folds, f)[1]]
    assert sorted(tested) == sorted(SIZES)
    for f in range(5):
        train, test = train_test(folds, f)
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(SIZES)


def test_folds_balanced_by_query_count():
    folds = make_folds(SIZES, 3, seed=1)
    totals = [sum(SIZES[r] for r, f in folds.items() if f == i) for i in range(3)]
    # greedy largest-first: no fold exceeds the ideal share by more than the largest repo
    assert max(totals) - min(totals) <= max(SIZES.values())
    assert max(totals) < 0.5 * sum(SIZES.values())


def test_folds_deterministic_for_seed():
    assert make_folds(SIZES, 5, 7) == make_folds(SIZES, 5, 7)


def test_bad_k_rejected():
    with pytest.raises(ValueError):
        make_folds({"a": 1, "b": 2}, 3, 0)


def test_inner_split_holds_out_about_a_quarter_of_queries():
    train = [r for r in SIZES if r != "r0"]
    inner, valid = inner_split(train, SIZES, 0.25, seed=3)
    assert set(inner) | set(valid) == set(train) and not set(inner) & set(valid)
    assert valid and inner
    share = sum(SIZES[r] for r in valid) / sum(SIZES[r] for r in train)
    assert share >= 0.25


def test_saved_folds_reused_and_invalidated(tmp_path):
    p = tmp_path / "folds.csv"
    a = load_or_make_folds(p, SIZES, 5, 42)
    p.write_text(p.read_text())  # unchanged file is reused
    assert load_or_make_folds(p, SIZES, 5, 42) == a
    other = {**SIZES, "new": 10}
    b = load_or_make_folds(p, other, 5, 42)  # different repo set -> regenerated
    assert set(b) == set(other)
