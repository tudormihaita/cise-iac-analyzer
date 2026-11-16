from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfimpact.evaluation import metrics as M


def _feat(rows: list[tuple]) -> pd.DataFrame:
    """rows: (repo, query_id, cand_id, kind, file, label)."""
    df = pd.DataFrame(
        rows, columns=["repo", "query_id", "cand_id", "cand_kind", "cand_file", "label"]
    )
    df["query_id"] = df["query_id"].astype("int64")
    return df


def _queries(rows: list[dict]) -> pd.DataFrame:
    base = {
        "n_labels_existing": 0,
        "n_labels_add_type": 0,
        "seed_is_new": False,
        "in_warmup": False,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["n_labels_in_candidates"] = df.get("n_labels_in_candidates", df["n_labels"])
    return df


def _single() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """One query, 5 candidates scored 5..1; positives at ranks 3 and 5; 4 true labels."""
    f = _feat([("r", 1, f"c{i}", "existing", "a.tf", int(i in (3, 5))) for i in range(1, 6)])
    q = _queries([{"repo": "r", "query_id": 1, "n_labels": 4, "n_labels_existing": 4}])
    return f, q, np.array([5.0, 4, 3, 2, 1])


def test_rr_recall_precision_hit() -> None:
    f, q, s = _single()
    pq = M.compute_query_metrics(f, q, s)
    row = pq.iloc[0]
    assert row["rr"] == pytest.approx(1 / 3)
    assert row["recall@1"] == 0 and row["recall@5"] == pytest.approx(2 / 4)  # denominator 4, not 2
    assert row["recall@10"] == pytest.approx(0.5)
    assert row["precision@1"] == 0 and row["precision@5"] == pytest.approx(2 / 5)
    assert row["precision@10"] == pytest.approx(2 / 10)
    assert (row["hit@1"], row["hit@5"], row["hit@10"]) == (0, 1, 1)
    assert row["n_labels"] == 4


def test_top_k_cut_zeroes_rr() -> None:
    f, q, s = _single()
    row = M.compute_query_metrics(f, q, s, top_k_cut=2).iloc[0]
    assert row["rr"] == 0 and row["hit@5"] == 0 and row["recall@5"] == 0
    row = M.compute_query_metrics(f, q, s, top_k_cut=3).iloc[0]
    assert row["rr"] == pytest.approx(1 / 3) and row["recall@5"] == pytest.approx(1 / 4)


def test_no_positive_in_candidates_and_missing_rows() -> None:
    f = _feat([("r", 1, "c1", "existing", "a.tf", 0)])
    q = _queries(
        [
            {"repo": "r", "query_id": 1, "n_labels": 2, "n_labels_existing": 2},
            {"repo": "r", "query_id": 2, "n_labels": 1, "n_labels_existing": 1},  # no rows at all
        ]
    )
    pq = M.compute_query_metrics(f, q, np.array([1.0]))
    assert len(pq) == 2 and (pq["rr"] == 0).all() and (pq["recall@10"] == 0).all()


def test_ties_are_seeded() -> None:
    n = 30
    f = _feat([("r", 1, f"c{i}", "existing", "a.tf", int(i == 0)) for i in range(n)])
    q = _queries([{"repo": "r", "query_id": 1, "n_labels": 1, "n_labels_existing": 1}])
    s = np.zeros(n)
    rr = lambda seed: M.compute_query_metrics(f, q, s, seed=seed)["rr"].iloc[0]  # noqa: E731
    assert rr(0) == rr(0) and rr(5) == rr(5)
    assert len({rr(seed) for seed in range(6)}) > 1  # a different seed changes the rank
    ranks_a = M.rank_within_groups(np.zeros(n, dtype=np.int64), s, seed=1)
    assert sorted(ranks_a) == list(range(1, n + 1))


def test_ranks_respect_scores_per_group() -> None:
    gid = np.array([1, 0, 1, 0])
    r = M.rank_within_groups(gid, np.array([0.1, 0.9, 0.5, 0.2]))
    assert r.tolist() == [2, 1, 1, 2]


def test_equal_repo_weighting() -> None:
    f = _feat(
        [("A", 1, "x", "existing", "f", 1), ("A", 1, "y", "existing", "f", 0)]
        + [(r, i, "x", "existing", "f", 0) for r, i in (("B", 1), ("B", 2), ("B", 3))]
        + [(r, i, "y", "existing", "f", 1) for r, i in (("B", 1), ("B", 2), ("B", 3))]
    )
    qs = _queries(
        [{"repo": "A", "query_id": 1, "n_labels": 1, "n_labels_existing": 1}]
        + [{"repo": "B", "query_id": i, "n_labels": 1, "n_labels_existing": 1} for i in (1, 2, 3)]
    )
    s = np.where(f["cand_id"] == "x", 2.0, 1.0)
    pq = M.compute_query_metrics(f, qs, s)
    per_repo, overall = M.aggregate(pq)
    assert per_repo.loc["A", "rr"] == 1.0 and per_repo.loc["B", "rr"] == 0.5
    assert per_repo["n_queries"].to_dict() == {"A": 1, "B": 3}
    assert overall["rr"] == pytest.approx(0.75)  # not (1 + 3*0.5)/4
    assert overall["n_queries"] == 4 and overall["n_repos"] == 2


def test_existing_vs_add_type_breakdown() -> None:
    f = _feat(
        [
            ("r", 1, "e1", "existing", "a.tf", 0),
            ("r", 1, "e2", "existing", "a.tf", 1),
            ("r", 1, "+type:t", "add_type", "", 1),
            ("r", 2, "e1", "existing", "a.tf", 1),
            ("r", 2, "+type:t", "add_type", "", 0),
        ]
    )
    qs = _queries(
        [
            {
                "repo": "r",
                "query_id": 1,
                "n_labels": 4,
                "n_labels_existing": 2,
                "n_labels_add_type": 2,
                "seed_is_new": True,
            },
            {
                "repo": "r",
                "query_id": 2,
                "n_labels": 1,
                "n_labels_existing": 1,
                "n_labels_add_type": 0,
            },
        ]
    )
    s = np.array([3.0, 2.0, 1.0, 3.0, 1.0])
    bd = M.breakdowns(f, qs, s)
    ex, _ = bd["targets_existing"]
    assert ex.loc["r", "n_queries"] == 2
    # q1: existing only -> e1 (neg), e2 (pos): recall@5 = 1/2; q2: 1/1 -> mean 0.75
    assert ex.loc["r", "recall@5"] == pytest.approx(0.75)
    ad, _ = bd["targets_add_type"]
    assert ad.loc["r", "n_queries"] == 1  # q2 has zero add-type labels
    assert ad.loc["r", "recall@5"] == pytest.approx(1 / 2)  # denominator n_labels_add_type=2
    assert ad.loc["r", "rr"] == 1.0
    assert bd["seed_new"][0].loc["r", "n_queries"] == 1
    assert bd["seed_existing"][0].loc["r", "n_queries"] == 1
    assert bd["seed_new"][0].loc["r", "recall@5"] == pytest.approx(2 / 4)


def test_file_level_dedup_keeps_first_occurrence() -> None:
    f = _feat(
        [
            ("r", 1, "e1", "existing", "a.tf", 0),
            ("r", 1, "e2", "existing", "b.tf", 0),
            ("r", 1, "e3", "existing", "a.tf", 1),  # a.tf already seen at rank 1
            ("r", 1, "e4", "existing", "c.tf", 1),
            ("r", 1, "+type:t", "add_type", "", 1),  # dropped
        ]
    )
    qs = _queries([{"repo": "r", "query_id": 1, "n_labels": 5, "n_labels_existing": 3}])
    s = np.array([5.0, 4, 3, 2, 9])
    row = M.compute_file_level_metrics(f, qs, s).iloc[0]
    # file ranking: a.tf (true), b.tf, c.tf (true); 2 true files
    assert row["n_labels"] == 2
    assert row["rr"] == 1.0
    assert row["recall@1"] == pytest.approx(1 / 2)
    assert row["recall@5"] == 1.0
    assert row["precision@5"] == pytest.approx(2 / 5)


def test_warmup_dropped() -> None:
    f = _feat([("r", 1, "x", "existing", "f", 1), ("r", 2, "x", "existing", "f", 1)])
    qs = _queries(
        [
            {"repo": "r", "query_id": 1, "n_labels": 1, "n_labels_existing": 1, "in_warmup": True},
            {"repo": "r", "query_id": 2, "n_labels": 1, "n_labels_existing": 1},
        ]
    )
    s = np.ones(2)
    assert M.compute_query_metrics(f, qs, s)["query_id"].tolist() == [2]
    assert len(M.compute_query_metrics(f, qs, s, drop_warmup=False)) == 2


def test_candidate_recall_ceiling() -> None:
    qs = _queries(
        [
            {"repo": "A", "query_id": 1, "n_labels": 4, "n_labels_in_candidates": 2},
            {"repo": "A", "query_id": 2, "n_labels": 4, "n_labels_in_candidates": 4},
            {"repo": "B", "query_id": 1, "n_labels": 2, "n_labels_in_candidates": 2},
        ]
    )
    per_repo, overall = M.candidate_recall_ceiling(qs)
    assert per_repo["A"] == pytest.approx(6 / 8) and per_repo["B"] == 1.0
    assert overall == pytest.approx((0.75 + 1.0) / 2)
