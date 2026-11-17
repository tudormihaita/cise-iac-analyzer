from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfimpact.config import load_config
from tfimpact.evaluation.loader import RepoData
from tfimpact.evaluation.metrics import aggregate, compute_query_metrics
from tfimpact.features.vector import FEATURES
from tfimpact.methods import ga as G

NAMES = list(load_config().model.features)
F = len(NAMES)


def _repo(name: str, n_queries: int, seed: int, signal: str = "graph_propagation_conf") -> RepoData:
    """Random features; true answers have a high value of `signal` (a learnable pattern)."""
    rng = np.random.default_rng(seed)
    rows, qrows = [], []
    for q in range(n_queries):
        qid = 1000 * (q + 1)
        n_pos = int(rng.integers(0, 3))  # some queries have no true answer among candidates
        for c in range(12):
            label = int(c < n_pos)
            feats = {f: float(rng.random()) for f in FEATURES}
            feats[signal] = float(0.6 + 0.4 * rng.random()) if label else float(0.5 * rng.random())
            kind = "add_type" if c >= 10 else "existing"
            rows.append(
                {
                    "repo": name,
                    "query_id": qid,
                    "cand_kind": kind,
                    "cand_file": "a.tf",
                    "label": label,
                    **feats,
                }
            )
        qrows.append(
            {"repo": name, "query_id": qid, "seed_is_new": False, "n_labels": max(n_pos, 1)}
        )
    return RepoData(name, "", pd.DataFrame(rows), pd.DataFrame(qrows), NAMES)


def _cfg(tmp_path, **ga):
    cfg = load_config()
    cfg.paths.outputs = tmp_path
    cfg.ga.population, cfg.ga.generations, cfg.ga.fitness_sample = 16, 8, 200
    cfg.evaluation.seeds = 2
    cfg.ga.tune = False
    for k, v in ga.items():
        setattr(cfg.ga, k, v)
    return cfg


def test_fitness_equals_repo_weighted_mrr_without_ties():
    repos = [_repo("a", 30, 1), _repo("b", 10, 2)]
    ts = G.TrainSet.build(repos, NAMES)
    batch = ts.sample(None, np.random.default_rng(0))
    chrom = np.r_[np.full(F, 1 / F), 0.1]
    fit = G.fitness(chrom[None], batch)[0]
    # reference: project metrics on the queries that have a true answer among candidates
    pq = []
    for d in repos:
        keep = d.features[d.features.label == 1].query_id.unique()
        f = d.features[d.features.query_id.isin(keep)]
        q = d.queries[d.queries.query_id.isin(keep)].assign(n_labels=1)
        s = G.scores(
            f[NAMES].to_numpy(np.float32), (f.cand_kind == "add_type").to_numpy(np.float32), chrom
        )[:, 0]
        pq.append(compute_query_metrics(f, q, s, ks=[1], drop_warmup=False))
    assert fit == pytest.approx(aggregate(pd.concat(pq))[1]["rr"], abs=1e-6)


def test_fitness_ties_use_expected_rank():
    b = G.Batch(
        X=np.zeros((3, F), np.float32),
        is_add=np.zeros(3, np.float32),
        y=np.array([False, True, False]),
        starts=np.array([0]),
        gid=np.zeros(3, np.int64),
        qw=np.array([1.0]),
    )
    # all three candidates tie -> expected rank of the true answer = 2
    assert G.fitness(np.r_[np.full(F, 1 / F), 0.0][None], b)[0] == pytest.approx(0.5)


def test_repair_keeps_constraints():
    r = G.repair(np.r_[np.linspace(-0.5, 2, F), 3.0], offset_range=1.0, negative=False)
    assert (r[:-1] >= 0).all() and r[:-1].sum() == pytest.approx(1) and r[-1] == 1.0
    n = G.repair(np.r_[np.linspace(-0.5, 2, F), -3.0], offset_range=1.0, negative=True)
    assert (n[:-1] < 0).any() and np.abs(n[:-1]).sum() == pytest.approx(1) and n[-1] == -1.0
    z = G.repair(np.r_[np.zeros(F), 0.0], 1.0, False)
    assert z[:-1] == pytest.approx(np.full(F, 1 / F))


def test_sample_is_equal_per_repo_and_permutation_keeps_positive_counts():
    repos = [_repo("a", 40, 1), _repo("b", 15, 2)]
    ts = G.TrainSet.build(repos, NAMES)
    batch = ts.sample(20, np.random.default_rng(0))  # 10 per repo, or all a repo has
    avail = np.bincount(ts.q_repo, minlength=2)
    assert len(batch.starts) == np.minimum(avail, 10).sum()
    assert batch.qw.sum() == pytest.approx(1)
    # each repo carries half of the fitness weight, however many queries it contributed
    n_a = min(avail[0], 10)
    assert batch.qw[:n_a].sum() == pytest.approx(0.5) and batch.qw[n_a:].sum() == pytest.approx(0.5)
    perm = G.TrainSet.build(repos, NAMES, permute_rng=np.random.default_rng(3))
    per_q = lambda t: np.add.reduceat(t.y.astype(int), t.starts)  # noqa: E731
    assert np.array_equal(per_q(ts), per_q(perm)) and not np.array_equal(ts.y, perm.y)


def test_evolve_learns_planted_signal_and_is_deterministic(tmp_path):
    cfg = _cfg(tmp_path, population=24, generations=15)
    batch = G.TrainSet.build([_repo("a", 60, 1)], NAMES).sample(None, np.random.default_rng(0))
    r1 = G.evolve(batch, cfg.ga, np.random.default_rng(5), F)
    r2 = G.evolve(batch, cfg.ga, np.random.default_rng(5), F)
    assert np.array_equal(r1.best, r2.best)
    assert r1.history[-1]["best"] >= r1.history[0]["best"]
    assert r1.best[NAMES.index("graph_propagation_conf")] == r1.best[:-1].max()
    assert len(r1.history) == cfg.ga.generations + 1


def test_random_search_has_the_same_budget(tmp_path):
    cfg = _cfg(tmp_path)
    batch = G.TrainSet.build([_repo("a", 20, 1)], NAMES).sample(None, np.random.default_rng(0))
    ga = G.evolve(batch, cfg.ga, np.random.default_rng(1), F)
    rs = G.random_search(batch, cfg.ga, np.random.default_rng(1), F)
    assert ga.evaluations == rs.evaluations == cfg.ga.population * (cfg.ga.generations + 1)


def test_runs_are_saved_and_resumed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    train = [_repo("a", 20, 1), _repo("b", 20, 2)]
    m1 = G.GAMethod(cfg, "ga").fit(train, fold=0)
    files = sorted((tmp_path / "ga_runs").rglob("*.json"))
    assert len(files) == cfg.evaluation.seeds

    def boom(*a, **k):
        raise AssertionError("should have been loaded from disk")

    monkeypatch.setattr("tfimpact.methods.ga.method.evolve", boom)
    monkeypatch.setattr("tfimpact.methods.ga.grid.evolve", boom)
    m2 = G.GAMethod(cfg, "ga").fit(train, fold=0)
    assert [r["weights"] for r in m1.runs] == [r["weights"] for r in m2.runs]
    cfg.ga.mutation_rate = 0.5  # changed setting -> different hash -> re-run
    with pytest.raises(AssertionError, match="loaded from disk"):
        G.GAMethod(cfg, "ga").fit(train, fold=0)


def test_rank_all_gives_one_ranking_per_seed(tmp_path):
    cfg = _cfg(tmp_path)
    train = [_repo("a", 20, 1)]
    m = G.GAMethod(cfg, "ga").fit(train, fold=0)
    rks = m.rank_all(_repo("t", 5, 9))
    assert len(rks) == cfg.evaluation.seeds and all(len(r.scores) == 60 for r in rks)
    assert len(G.GAMethod(cfg, "ga_permuted").fit(train, fold=0).runs) == cfg.ga.check_seeds
    with pytest.raises(ValueError):
        G.GAMethod(cfg, "nope")


def _small_grid(cfg):
    cfg.ga.tune = True
    cfg.ga.grid.pop_gen = [[8, 4], [4, 8]]
    cfg.ga.grid.mutation_rate = [0.3]
    cfg.ga.grid.crossover_rate = [0.6, 0.9]
    cfg.ga.grid.seeds = 2
    return cfg


def test_grid_search_picks_best_validation_config_and_resumes(tmp_path, monkeypatch):
    cfg = _small_grid(_cfg(tmp_path))
    train = [_repo(n, 15, i) for i, n in enumerate("abcd")]
    m = G.GAMethod(cfg, "ga").fit(train, fold=0)
    assert len(m.tuning) == 4 * cfg.ga.grid.seeds
    assert {r["fold"] for r in m.tuning} == {0}
    means = pd.DataFrame(m.tuning).groupby("config").valid.mean()
    assert m.chosen == means.idxmax()
    assert G.config_id(m.ga_cfg) == m.chosen and m.params()["chosen"] == m.chosen
    assert all(r["config"] == m.chosen for r in m.runs)  # final runs use the chosen config
    # grid runs only use the training repos: validation repos are a subset of them
    assert len(list((tmp_path / "ga_runs").glob("grid_*/fold0_*.json"))) == 8

    def boom(*a, **k):
        raise AssertionError("should have been loaded from disk")

    monkeypatch.setattr("tfimpact.methods.ga.method.evolve", boom)
    monkeypatch.setattr("tfimpact.methods.ga.grid.evolve", boom)
    m2 = G.GAMethod(cfg, "ga").fit(train, fold=0)
    assert m2.chosen == m.chosen and [r["weights"] for r in m2.runs] == [
        r["weights"] for r in m.runs
    ]


def test_grid_configs_cover_the_product():
    cfg = load_config()
    configs = G.grid_configs(cfg.ga)
    g = cfg.ga.grid
    assert len(configs) == len(g.pop_gen) * len(g.mutation_rate) * len(g.crossover_rate)
    assert len({G.config_id(c) for c in configs}) == len(configs)
    default = G.config_id(cfg.ga)
    assert default in {G.config_id(c) for c in configs}  # SPEC defaults are a grid point
