from __future__ import annotations

import numpy as np
import pandas as pd

from tests.test_baselines_eval import fake_cfg  # noqa: F401  (fixture)
from tfimpact.evaluation import studies as St
from tfimpact.evaluation.experiments import run_evaluation, tuning_table


def _small_ga(cfg):
    cfg.ga.population, cfg.ga.generations, cfg.ga.fitness_sample = 10, 4, 100
    cfg.ga.grid.pop_gen = [[6, 3], [3, 6]]
    cfg.ga.grid.mutation_rate = [0.3]
    cfg.ga.grid.crossover_rate = [0.8]
    cfg.ga.grid.seeds = 1
    cfg.evaluation.seeds = 3
    cfg.experiments.seeds = 2
    return cfg


def _chosen(cfg):
    main = run_evaluation(cfg, ["ga"])
    return main, St.chosen_configs(tuning_table(main.meta["method_params"]))


def test_family_features_and_specs(fake_cfg):  # noqa: F811
    names = list(fake_cfg.model.features)
    specs = St.ablation_specs(fake_cfg)
    assert list(specs) == ["ga", "ga_hist", "ga_graph", "ga_text"]
    fam = [f for k in ("ga_hist", "ga_graph", "ga_text") for f in specs[k]["features"]]
    assert sorted(fam) == sorted(names)  # the three families partition the model features
    fake_cfg.experiments.learning_curve = [0.5, 0.1]
    assert list(St.learning_curve_specs(fake_cfg)) == ["ga_lc10", "ga_lc50", "ga"]


def test_ablation_reuses_the_full_ga_and_compares_to_it(fake_cfg):  # noqa: F811
    cfg = _small_ga(fake_cfg)
    main, chosen = _chosen(cfg)
    assert set(chosen) == {0, 1}
    specs = St.ablation_specs(cfg)
    res = run_evaluation(cfg, list(specs), factory=St.factory(specs, chosen, 2), reference=St.FULL)
    # full GA = the main run's first seeds (loaded from its run files)
    key = ["method", "target", "repo", "query_id", "fold", "run"]
    a = main.per_query[main.per_query.run < 2].sort_values(key).reset_index(drop=True)
    b = res.per_query[res.per_query.method == "ga"].sort_values(key).reset_index(drop=True)
    pd.testing.assert_series_equal(a.rr, b.rr)
    assert set(zip(res.stats.method_a, res.stats.method_b, strict=True)) == {
        ("ga", "ga_hist"), ("ga", "ga_graph"), ("ga", "ga_text")
    }  # fmt: skip
    t = St.ablation_table(res, specs, list(cfg.model.features))
    assert len(t) == 4 * 2 and list(t.variant[:4]) == list(specs)
    row = t[(t.variant == "ga_hist") & (t.target == "all")].iloc[0]
    assert row.n_features == 6
    assert np.isclose(row.ri_full, (row.full_minus_variant + row.mrr) / row.mrr - 1)
    assert np.isnan(t[t.variant == "ga"].p_holm).all()
    md = St.study_markdown("Ablation", "note", t, {"variant": "v", "ri_full": "RI", "p_holm": "p"})
    assert "| ga_hist |" in md and "%" in md


def test_learning_curve_tables(fake_cfg):  # noqa: F811
    cfg = _small_ga(fake_cfg)
    cfg.experiments.learning_curve = [0.5]
    _, chosen = _chosen(cfg)
    specs = St.learning_curve_specs(cfg)
    res = run_evaluation(cfg, list(specs), factory=St.factory(specs, chosen, 2), reference=St.FULL)
    runs = St.learning_curve_runs(res, specs)
    # 4 core repos, 2 folds -> 2 training repos per fold; 50 % -> 1 repo
    assert set(runs[runs.method == "ga_lc50"].n_train_repos) == {1}
    assert set(runs[runs.method == "ga"].n_train_repos) == {2}
    assert len(runs) == 2 * 2 * 2 * 2  # methods x targets x folds x runs
    assert runs.train_fitness.notna().all()
    t = St.learning_curve_table(res, specs)
    assert list(t[t.target == "all"].train_frac) == [0.5, 1.0]
    assert t[["mrr", "ci_low", "ci_high", "train_fitness", "sd_folds"]].notna().all().all()
