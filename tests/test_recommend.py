from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from tfimpact.config import load_config
from tfimpact.features.vector import FEATURES, FeatureBuilder
from tfimpact.inference.recommend import (
    Model,
    RecommendError,
    mine_state,
    recommend,
    resolve_block,
    worktree_seeds,
)
from tfimpact.processing.pipeline import process_repo


def _bucket(name: str, extra: str = "") -> str:
    return f'resource "aws_s3_bucket" "{name}" {{\n  bucket = "{name}"\n  {extra}\n}}\n'


def _policy(name: str, bucket: str, n: int = 0) -> str:
    return (
        f'resource "aws_s3_bucket_policy" "{name}" {{\n'
        f'  bucket = aws_s3_bucket.{bucket}.id\n  policy = "p{n}"\n}}\n'
    )


def _history() -> list[dict[str, str | None]]:
    h: list[dict[str, str | None]] = [
        {
            "main.tf": _bucket("logs") + _policy("logs", "logs"),
            "data.tf": _bucket("data") + _policy("data", "data"),
            "vars.tf": 'variable "env" { default = "dev" }\n',
            "mod/main.tf": _bucket("mod", "tags = { a = var.x }") + 'variable "x" {}\n',
        }
    ]
    for n in range(1, 9):  # buckets and their policies change together
        h.append(
            {
                "main.tf": _bucket("logs", f'tags = {{ v = "{n}" }}') + _policy("logs", "logs", n),
                "data.tf": _bucket("data", f'tags = {{ v = "{n}" }}')
                + _policy("data", "data", n - 1),
            }
        )
    return h


def _cfg():
    cfg = load_config()
    cfg.mining.warmup_frac = 0.0
    return cfg


def _model(tmp_path, cfg) -> Model:
    path = tmp_path / "model.json"
    names = list(cfg.model.features)
    path.write_text(
        json.dumps(
            {
                "features": names,
                "weights": {n: 1 / len(names) for n in names},
                "offset": 0.1,
                "features_config": asdict(cfg.features),
            }
        )
    )
    return Model.load(path)


LAST = {  # an edit: the logs bucket and its policy change, a new bucket is added
    "main.tf": _bucket("logs", 'tags = { v = "99" }')
    + _policy("logs", "logs", 99)
    + _bucket("archive"),
}


def test_worktree_features_equal_training_features(make_git_repo):
    """Features `recommend` builds for uncommitted edits == the training pipeline's features
    for the same edits committed (same candidates, same values)."""
    cfg = _cfg()
    committed = make_git_repo(_history() + [LAST])
    res = process_repo(committed.path, "a", cfg, blobless=False)
    last_idx = len(_history())
    q = res.queries[res.queries.idx == last_idx]
    assert len(q) == 3

    edited = make_git_repo(_history())
    for path, text in LAST.items():
        (edited.path / path).write_text(text)
    state = mine_state(edited.path, cfg)
    items, errors = worktree_seeds(edited.path, state)
    assert not errors and len(items) == 3
    builder = FeatureBuilder(
        cfg.features,
        state.snap,
        state.graph,
        state.tokens,
        state.miner,
        state.propagation,
        state.content,
    )
    cf = builder.build_commit(state.last_idx + 1, "worktree", items)
    for qrow, X, ids in zip(cf.queries, cf.X, cf.cand_ids, strict=True):
        ref = q[q.seed_address == qrow["seed_address"]].query_id.iloc[0]
        f = res.features[res.features.query_id == ref]
        assert list(f.cand_id) == ids
        np.testing.assert_allclose(f[FEATURES].to_numpy(), X, rtol=0, atol=1e-6)


def test_diff_recommendation_lists(make_git_repo, tmp_path):
    cfg = _cfg()
    repo = make_git_repo(_history())
    (repo.path / "main.tf").write_text(LAST["main.tf"])
    rec = recommend(repo.path, cfg, _model(tmp_path, cfg), diff=True)
    seeds = {a for a, _ in rec.seeds}
    assert seeds == {
        ".::resource.aws_s3_bucket.logs",
        ".::resource.aws_s3_bucket_policy.logs",
        ".::resource.aws_s3_bucket.archive",
    }
    assert rec.modify and not seeds & {s.candidate for s in rec.modify}  # seeds are not suggested
    assert [s.score for s in rec.modify] == sorted((s.score for s in rec.modify), reverse=True)
    assert all(s.kind == "add_type" and not s.candidate.startswith("+") for s in rec.add_types)
    assert len(rec.modify) <= cfg.recommend.top
    assert rec.n_commits == len(_history())
    d = rec.to_dict()
    assert d["modify"][0]["contributions"] and json.dumps(d)


def test_block_seed_short_names_and_errors(make_git_repo, tmp_path):
    cfg = _cfg()
    repo = make_git_repo(_history())
    state = mine_state(repo.path, cfg)
    assert resolve_block(state, "aws_s3_bucket.logs") == ".::resource.aws_s3_bucket.logs"
    assert resolve_block(state, "var.x") == "mod::var.x"
    with pytest.raises(RecommendError, match="no block"):
        resolve_block(state, "aws_s3_bucket.nope")
    rec = recommend(repo.path, cfg, _model(tmp_path, cfg), block="aws_s3_bucket.logs", top=3)
    assert rec.seeds == [(".::resource.aws_s3_bucket.logs", "~")] and len(rec.modify) <= 3
    # the policy that always changed with the bucket and references it ranks first
    assert rec.modify[0].candidate == ".::resource.aws_s3_bucket_policy.logs"
    with pytest.raises(RecommendError, match="no block differs"):
        recommend(repo.path, cfg, _model(tmp_path, cfg), diff=True)
    with pytest.raises(RecommendError, match="exactly one"):
        recommend(repo.path, cfg, _model(tmp_path, cfg))


def test_ambiguous_block_and_incompatible_model(make_git_repo, tmp_path):
    cfg = _cfg()
    repo = make_git_repo([{"a/main.tf": _bucket("x"), "b/main.tf": _bucket("x")}])
    state = mine_state(repo.path, cfg)
    with pytest.raises(RecommendError, match="ambiguous"):
        resolve_block(state, "aws_s3_bucket.x")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"weights": {"ref_direct": 1.0}, "offset": 0.0}))
    with pytest.raises(RecommendError, match="incompatible"):
        Model.load(bad)
    with pytest.raises(RecommendError, match="not found"):
        Model.load(tmp_path / "missing.json")


def test_removed_block_in_worktree_is_a_removed_seed(make_git_repo, tmp_path):
    cfg = _cfg()
    repo = make_git_repo(_history())
    (repo.path / "data.tf").unlink()
    rec = recommend(repo.path, cfg, _model(tmp_path, cfg), diff=True)
    assert dict(rec.seeds) == {
        ".::resource.aws_s3_bucket.data": "-",
        ".::resource.aws_s3_bucket_policy.data": "-",
    }


def test_final_model_file_is_compatible():
    cfg = load_config()
    m = Model.load(cfg.paths.models / "ga_weights.json")
    assert m.features == list(cfg.model.features)
    assert m.weights.sum() == pytest.approx(1.0) and (m.weights >= 0).all()
