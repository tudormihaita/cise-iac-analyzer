from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfimpact.config import ArmConfig, load_config
from tfimpact.evaluation.experiments import run_evaluation, save_results, summary_markdown
from tfimpact.evaluation.loader import RepoData, dataset_repos, load_repo
from tfimpact.features.vector import FEATURES
from tfimpact.methods.baselines import (
    Arm,
    FixedWeights,
    RandomRanking,
    lexicographic_score,
    make_baseline,
)


def _repo_data(rows: list[dict], queries: list[dict], repo: str = "r") -> RepoData:
    f = pd.DataFrame([{**dict.fromkeys(FEATURES, 0.0), "cand_file": "a.tf", **r} for r in rows])
    f.insert(0, "repo", repo)
    q = pd.DataFrame(queries)
    q.insert(0, "repo", repo)
    return RepoData(repo, "", f, q, list(load_config().model.features))


def test_lexicographic_score_orders_by_primary_then_secondary():
    s = lexicographic_score(np.array([0.5, 0.5, 0.9, 0.5]), np.array([0.1, 0.7, 0.0, 0.1]))
    assert s[2] > s[1] > s[0] == s[3]


def test_arm_thresholds_ties_and_new_seed_fallback():
    rows = [
        # query 1: existing seed -> instance rules only
        {
            "query_id": 1,
            "cand_kind": "existing",
            "hist_cochange_conf": 0.5,
            "hist_cochange_count": 0.2,
        },
        {
            "query_id": 1,
            "cand_kind": "existing",
            "hist_cochange_conf": 0.5,
            "hist_cochange_count": 0.9,
        },
        {"query_id": 1, "cand_kind": "existing", "hist_cochange_conf": 0.05},  # below 0.1
        {"query_id": 1, "cand_kind": "existing", "hist_type_cochange_conf": 0.9},  # no rule
        {"query_id": 1, "cand_kind": "add_type", "hist_type_cochange_conf": 0.9},
        # query 2: new seed -> type rules, for existing and add-type candidates
        {"query_id": 2, "cand_kind": "existing", "hist_type_cochange_conf": 0.4},
        {"query_id": 2, "cand_kind": "add_type", "hist_type_cochange_conf": 0.8},
        {"query_id": 2, "cand_kind": "add_type", "hist_type_cochange_conf": 0.05},
    ]
    q = [{"query_id": 1, "seed_is_new": False}, {"query_id": 2, "seed_is_new": True}]
    rk = Arm(ArmConfig()).rank(_repo_data(rows, q))
    assert rk.recommended.tolist() == [True, True, False, False, False, True, True, False]
    assert rk.top_k == 10
    s = rk.scores
    assert s[1] > s[0]  # same confidence, higher support first
    assert s[6] > s[5]
    assert (s[~rk.recommended] < s[rk.recommended].min()).all()
    with pytest.raises(ValueError):
        Arm(ArmConfig(min_support=2))


def test_fixed_weights_is_plain_average_and_random_is_reproducible():
    names = list(load_config().model.features)
    rows = [{"query_id": 1, "cand_kind": "existing", names[0]: 1.0, names[1]: 0.5}]
    d = _repo_data(rows, [{"query_id": 1, "seed_is_new": False}])
    assert FixedWeights(names).rank(d).scores[0] == pytest.approx(1.5 / len(names))
    a, b = RandomRanking(1).rank(d).scores, RandomRanking(1).rank(d).scores
    assert np.array_equal(a, b)


# ------------------------------------------------------------------ end-to-end on fake repos


def _write_fake_repo(root, slug: str, n_queries: int, rng: np.random.Generator) -> None:
    d = root / slug
    d.mkdir(parents=True)
    qrows, frows = [], []
    for i in range(n_queries):
        qid = 1000 * (i + 1)
        n_pos = int(rng.integers(1, 3))
        qrows.append(
            {
                "repo": slug,
                "query_id": qid,
                "idx": i + 1,
                "seed_address": f".::resource.x.s{i}",
                "seed_action": "~+-"[i % 3],
                "seed_is_new": i % 3 == 1,
                "n_labels": n_pos,
                "n_labels_existing": n_pos,
                "n_labels_add_type": 0,
                "n_labels_in_candidates": n_pos,
                "in_warmup": i < 2,
            }
        )
        for c in range(10):
            label = int(c < n_pos)
            feats = {f: float(rng.random() * 0.3) for f in FEATURES}
            # planted signal: true answers have high co-change confidence / type confidence
            feats["hist_cochange_conf"] = 0.8 if label else feats["hist_cochange_conf"] * 0.3
            feats["hist_type_cochange_conf"] = 0.7 if label else 0.2
            frows.append(
                {
                    "repo": slug,
                    "query_id": qid,
                    "cand_id": f"c{c}",
                    "cand_kind": "existing",
                    "cand_file": f"f{c % 3}.tf",
                    **feats,
                    "label": label,
                }
            )
    pd.DataFrame(qrows).to_parquet(d / "queries.parquet")
    pd.DataFrame(frows).to_parquet(d / "features.parquet")


@pytest.fixture
def fake_cfg(tmp_path):
    rng = np.random.default_rng(0)
    cfg = load_config()
    cfg.paths.processed = tmp_path / "processed"
    cfg.paths.outputs = tmp_path / "results"
    cfg.paths.dataset_csv = tmp_path / "dataset.csv"
    cfg.evaluation.folds = 2
    cfg.evaluation.bootstrap = 200
    repos = {"a": 12, "b": 8, "c": 15, "d": 6, "e": 9}
    for slug, n in repos.items():
        _write_fake_repo(cfg.paths.processed, slug, n, rng)
    pd.DataFrame(
        {
            "slug": list(repos),
            "repo_kind": ["deployment", "module_library", "deployment", "module_library", "x"],
            "included": [True, True, True, True, False],
            "exclude_reason": [None, None, None, None, "add_only_frac>0.4"],
        }
    ).to_csv(cfg.paths.dataset_csv, index=False)
    return cfg


def test_loader_drops_warmup_and_selects_repos(fake_cfg):
    repos = dataset_repos(fake_cfg)
    assert dict(zip(repos.slug, repos.subset, strict=True)) == {
        "a": "core",
        "b": "core",
        "c": "core",
        "d": "core",
        "e": "add_only",
    }
    d = load_repo(fake_cfg, "a", fake_cfg.model.features)
    assert d.n_queries == 10 and not d.queries.in_warmup.any()
    assert set(d.features.query_id) == set(d.queries.query_id)
    assert d.X().shape == (100, 12) and d.X().dtype == np.float32


def test_run_evaluation_end_to_end(fake_cfg):
    methods = ["arm", "fixed_weights", "random"]
    res = run_evaluation(fake_cfg, methods)
    # every repo (4 core + 1 held out) evaluated once per method and target
    assert len(res.per_repo) == 5 * len(methods) * 2
    assert res.per_repo.groupby(["method", "target"]).repo.nunique().eq(5).all()
    held = res.per_repo[res.per_repo.repo == "e"]
    assert (held.subset == "add_only").all() and (held.fold == -1).all()
    assert (held.n_queries == 7).all()  # non-learned methods: evaluated once, no duplicates
    s = res.summary.set_index(["method", "target", "subset", "metric"])
    fw = ("fixed_weights", "all", "core", "rr")
    assert s.loc[fw, "mean"] > s.loc[("random", "all", "core", "rr"), "mean"]
    lo, hi = s.loc[fw, ["ci_low", "ci_high"]]
    assert lo <= s.loc[fw, "mean"] <= hi
    assert s.loc[fw, "n_repos"] == 4
    assert s.loc[("fixed_weights", "all", "add_only", "rr"), "n_repos"] == 1
    assert s.loc[("fixed_weights", "all", "all", "rr"), "n_repos"] == 5
    assert np.isnan(s.loc[("fixed_weights", "all", "all", "rr"), "sd_folds"])
    assert len(res.stats) == 3 * 2 * 2  # pairs x targets x metrics, core repos only
    assert set(res.per_query[res.per_query.repo == "e"].subset) == {"add_only"}
    assert set(res.breakdowns.by) == {"seed_action", "repo_kind"}
    assert {"added", "modified", "removed"} <= set(res.breakdowns.group)
    assert len(res.file_level) == len(methods)
    paths = save_results(res, fake_cfg.paths.outputs, "test")
    assert all(p.exists() for p in paths)
    assert "fixed_weights" in summary_markdown(res)
    # folds are saved and reused: a second run gives identical results
    res2 = run_evaluation(fake_cfg, methods)
    pd.testing.assert_frame_equal(res.per_repo, res2.per_repo)


class _Learner(FixedWeights):
    """A fake learned method recording what it was trained on."""

    name = "learner"
    learns = True
    seen: list[list[str]] = []

    def fit(self, train, fold=-1):
        _Learner.seen.append(sorted(d.repo for d in train))
        return self


def test_held_out_repos_never_trained_on_and_scored_by_every_fold(fake_cfg):
    _Learner.seen = []

    def factory(name, cfg):
        return _Learner(cfg.model.features) if name == "learner" else make_baseline(name, cfg)

    res = run_evaluation(fake_cfg, ["learner", "random"], factory=factory)
    assert len(_Learner.seen) == fake_cfg.evaluation.folds
    assert all("e" not in s for s in _Learner.seen)
    e = res.per_query[(res.per_query.repo == "e") & (res.per_query.target == "all")]
    assert e[e.method == "learner"].fold.nunique() == fake_cfg.evaluation.folds
    assert e[e.method == "random"].fold.nunique() == 1


def test_run_evaluation_with_ga_variants(fake_cfg):
    fake_cfg.ga.population, fake_cfg.ga.generations, fake_cfg.ga.fitness_sample = 10, 4, 100
    fake_cfg.ga.grid.pop_gen = [[6, 3], [3, 6]]  # small nested grid search for `ga`
    fake_cfg.ga.grid.mutation_rate = [0.3]
    fake_cfg.ga.grid.crossover_rate = [0.8]
    fake_cfg.ga.grid.seeds = 1
    fake_cfg.evaluation.seeds = 2
    res = run_evaluation(fake_cfg, ["ga", "random_search", "fixed_weights"])
    s = res.summary.set_index(["method", "target", "subset", "metric"])
    assert not np.isnan(s.loc[("ga", "all", "core", "rr"), "sd_runs"])
    assert np.isnan(s.loc[("fixed_weights", "all", "core", "rr"), "sd_runs"])
    ga_runs = res.per_query[(res.per_query.method == "ga") & (res.per_query.subset == "core")]
    assert ga_runs.run.nunique() == 2
    paths = save_results(res, fake_cfg.paths.outputs / "out", "test")
    lw = pd.read_csv(fake_cfg.paths.outputs / "out" / "learned_weights.csv")
    # 2 folds x 2 seeds for each learned method
    assert lw.groupby("method").size().to_dict() == {"ga": 4, "random_search": 4}
    assert any(p.name == "learned_weights.csv" for p in paths)
    grid = pd.read_csv(fake_cfg.paths.outputs / "out" / "grid.csv")
    assert set(grid.method) == {"ga"} and grid.groupby("fold").chosen.any().all()
    assert (fake_cfg.paths.outputs / "out" / "figures" / "per_repo_mrr.png").exists()
    assert "chosen by the grid search" in summary_markdown(res)
    # a second run reuses the saved GA runs and gives identical results
    res2 = run_evaluation(fake_cfg, ["ga", "random_search", "fixed_weights"])
    pd.testing.assert_frame_equal(res.per_repo, res2.per_repo)


def test_export_results_copies_tables_and_figures(fake_cfg, tmp_path):
    from tfimpact.evaluation.export import export_results

    res = run_evaluation(fake_cfg, ["fixed_weights", "random"])
    out = fake_cfg.paths.outputs
    save_results(res, out, "test")
    save_results(res, out / "checks", "test")
    copied = export_results(out, tmp_path / "report")
    names = {p.relative_to(tmp_path / "report").as_posix() for p in copied}
    assert {"summary.md", "metrics.csv", "stats.csv", "checks_summary.md"} <= names
    assert "figures/per_repo_mrr.png" in names and "figures/checks_per_repo_mrr.png" in names
    assert not any(n.endswith(".parquet") for n in names)  # large files stay in data/results
