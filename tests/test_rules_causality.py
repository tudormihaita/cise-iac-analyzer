from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd
import pytest

from tfimpact.config import load_config
from tfimpact.features.association_rules import AssociationRuleMiner, changed_key
from tfimpact.features.graph import adamic_adar_weights
from tfimpact.features.similarity import ContentIndex, cosine
from tfimpact.features.vector import FEATURES, validate_feature_names
from tfimpact.processing.diff import ChangeItem
from tfimpact.processing.hcl_parser import parse_blocks
from tfimpact.processing.pipeline import process_repo


def _res(name: str, refs: str = "", n: int = 0) -> str:
    return f'resource "aws_s3_bucket" "{name}" {{\n  bucket = "{name}-{n}"\n  {refs}\n}}\n'


def _history() -> list[dict[str, str | None]]:
    """Recurring co-changes (a+b, a+var, b+c), plus adds and a removal, over ~14 commits."""
    hist: list[dict[str, str | None]] = []
    hist.append(
        {
            "main.tf": _res("a", "tags = { x = var.env }") + _res("b", "x = aws_s3_bucket.a.id"),
            "vars.tf": 'variable "env" { default = "dev" }\n',
            "mod/main.tf": 'variable "y" {}\nresource "aws_sqs_queue" "q" { name = var.y }\n',
        }
    )
    for n in range(1, 13):
        files: dict[str, str | None] = {}
        if n % 3 == 0:  # a + var co-change
            files["main.tf"] = _res("a", "tags = { x = var.env }", n) + _res(
                "b", "x = aws_s3_bucket.a.id", n - 1 if n > 1 else 0
            )
            files["vars.tf"] = f'variable "env" {{ default = "v{n}" }}\n'
        elif n % 3 == 1:  # a + b
            files["main.tf"] = _res("a", "tags = { x = var.env }", n) + _res(
                "b", "x = aws_s3_bucket.a.id", n
            )
        else:  # b + new block / module call
            files["main.tf"] = (
                _res("a", "tags = { x = var.env }", n - 1)
                + _res("b", "x = aws_s3_bucket.a.id", n)
                + f'resource "aws_iam_role" "r{n}" {{ name = aws_s3_bucket.b.id }}\n'
            )
            files["calls.tf"] = (
                f'module "m" {{\n  source = "./mod"\n  y = "{n}"\n}}\n'
                f'output "o" {{ value = module.m.q{n} }}\n'
            )
        hist.append(files)
    return hist


@pytest.fixture
def repo(make_git_repo):
    return make_git_repo(_history())


def _cfg():
    cfg = load_config()
    cfg.mining.warmup_frac = 0.0
    return cfg


def _rows_at(res, idx: int) -> pd.DataFrame:
    q = res.queries[res.queries.idx == idx]
    f = res.features[res.features.query_id.isin(q.query_id)].drop(columns=["repo"])
    f = f.astype({"cand_kind": str})  # category list depends on which kinds occur overall
    return f.sort_values(["query_id", "cand_id"]).reset_index(drop=True)


def test_prefix_invariance_no_leakage(repo):
    full = process_repo(repo.path, "t", _cfg(), blobless=False)
    assert len(full.queries) > 0
    checked = 0
    for idx in sorted(full.queries.idx.unique()):
        truncated = process_repo(repo.path, "t", _cfg(), blobless=False, rev=repo.shas[idx])
        a, b = _rows_at(full, idx), _rows_at(truncated, idx)
        assert len(a) > 0
        pd.testing.assert_frame_equal(a, b)
        qa = full.queries[full.queries.idx == idx].drop(columns=["in_warmup"])
        qb = truncated.queries[truncated.queries.idx == idx].drop(columns=["in_warmup"])
        pd.testing.assert_frame_equal(qa.reset_index(drop=True), qb.reset_index(drop=True))
        checked += 1
    assert checked >= 5


def test_features_do_not_see_current_commit(repo):
    """At the first commit after the start, all miners are empty: no mined feature may fire."""
    res = process_repo(repo.path, "t", _cfg(), blobless=False)
    first = res.queries.idx.min()
    assert first == 1
    f = _rows_at(res, first)
    mined = [c for c in FEATURES if c.startswith("hist_")] + ["graph_propagation_conf"]
    assert (f[mined] == 0).all().all()


def test_labels_match_changes(repo):
    res = process_repo(repo.path, "t", _cfg(), blobless=False)
    for _, q in res.queries.iterrows():
        ch = res.changes[res.changes.idx == q.idx]
        others = ch[(ch.address != q.seed_address) & ch.action.isin(["~", "-"])]
        assert q.n_labels_existing == len(others)
        f = res.features[(res.features.query_id == q.query_id) & (res.features.label == 1)]
        assert set(f[f.cand_kind == "existing"].cand_id) <= set(others.address)


# ----------------------------------------------------------------- miner unit tests


def _item(addr: str, action: str, btype: str = "aws_s3_bucket") -> ChangeItem:
    return ChangeItem(
        addr, "resource", btype, addr.rsplit(".", 1)[-1], action, "main.tf", None, None
    )


def test_instance_confidence_and_support():
    m = AssociationRuleMiner(recency_tau=100)
    m.update(1, [_item("A", "~"), _item("B", "~")])
    m.update(2, [_item("A", "~"), _item("C", "~")])
    m.update(3, [_item("A", "~")])  # single-item commits still count toward N(A)
    assert m.n_item["A"] == 3
    assert m.support("A", "B") == 1
    assert m.confidence("A", "B") == pytest.approx(1 / 3)
    assert m.confidence("B", "A") == pytest.approx(1.0)
    assert m.confidence("Z", "A") == 0.0


def test_recency_decay_and_churn():
    m = AssociationRuleMiner(recency_tau=10)
    m.update(0, [_item("A", "~"), _item("B", "~")])
    m.update(10, [_item("A", "~"), _item("B", "~")])
    expected = math.exp(-(20 - 0) / 10) + math.exp(-(20 - 10) / 10)
    assert m.recency("A", "B", 20) == pytest.approx(expected)
    assert m.churn("A", now=20, window=15) == 1  # only idx 10 is within (5, 20)
    assert m.churn("A", now=20, window=25) == 2


def test_type_rules_changed_key_excludes_own_occurrence():
    m = AssociationRuleMiner(recency_tau=100)
    # one modified iam policy + an added table: the policy alone must not support *iam_policy
    m.update(1, [_item("P", "~", "aws_iam_policy"), _item("T", "+", "aws_dynamodb_table")])
    assert m.type_confidence("~aws_iam_policy", changed_key("aws_iam_policy")) == 0.0
    assert m.type_confidence("+aws_dynamodb_table", changed_key("aws_iam_policy")) == 1.0
    assert m.type_confidence("+aws_dynamodb_table", "+aws_dynamodb_table") == 0.0
    # two modified policies: each one supports "another policy changed"
    m.update(2, [_item("P", "~", "aws_iam_policy"), _item("Q", "-", "aws_iam_policy")])
    assert m.type_confidence("~aws_iam_policy", changed_key("aws_iam_policy")) == 0.5


def test_cochange_conf_recomputed_from_strictly_earlier_commits(repo):
    """Independent recomputation of f1 from `changes` using only commits < t (1..max_items items).

    Catches leakage of commit t itself (which prefix invariance cannot see) and of later commits.
    """
    cfg = _cfg()
    res = process_repo(repo.path, "t", cfg, blobless=False)
    ch, commits = res.changes, res.commits
    eligible = set(commits[(commits.n_items >= 1) & (commits.n_items <= cfg.mining.max_items)].idx)
    tx = {i: set(g.address) for i, g in ch[ch.idx.isin(eligible)].groupby("idx")}
    feats = res.features.merge(res.queries[["query_id", "idx", "seed_address"]], on="query_id")
    feats = feats[feats.cand_kind == "existing"]
    n_checked_pos = 0
    for row in feats.itertuples():
        past = [s for i, s in tx.items() if i < row.idx]
        n_q = sum(row.seed_address in s for s in past)
        n_qc = sum(row.seed_address in s and row.cand_id in s for s in past)
        expected = n_qc / n_q if n_q else 0.0
        assert row.hist_cochange_conf == pytest.approx(expected, abs=1e-6), (row.idx, row.cand_id)
        n_checked_pos += expected > 0
    assert n_checked_pos > 10  # the test history must actually exercise non-zero confidences


def test_file_cochange_conf_recomputed_from_strictly_earlier_commits(repo):
    """Independent recomputation of the file-level rule from `changes` (commits < t only)."""
    cfg = _cfg()
    res = process_repo(repo.path, "t", cfg, blobless=False)
    ch, commits = res.changes, res.commits
    eligible = set(commits[(commits.n_items >= 1) & (commits.n_items <= cfg.mining.max_items)].idx)
    per_tx = {
        i: g.file.value_counts().to_dict() for i, g in ch[ch.idx.isin(eligible)].groupby("idx")
    }
    seed_file = dict(zip(zip(ch.idx, ch.address, strict=True), ch.file, strict=True))
    feats = res.features.merge(res.queries[["query_id", "idx", "seed_address"]], on="query_id")
    feats = feats[feats.cand_kind == "existing"]
    n_pos = 0
    for row in feats.itertuples():
        sf = seed_file[(row.idx, row.seed_address)]
        past = [files for i, files in per_tx.items() if i < row.idx and sf in files]
        if row.cand_file == sf:
            hits = sum(files[sf] >= 2 for files in past)
        else:
            hits = sum(row.cand_file in files for files in past)
        expected = hits / len(past) if past else 0.0
        assert row.hist_file_cochange_conf == pytest.approx(expected, abs=1e-6), (
            row.idx,
            row.cand_id,
        )
        n_pos += expected > 0
    assert n_pos > 10


# ------------------------------------------------------------- propagation rule (mined)


def _q(name: str, extra: str = "") -> str:
    return f'resource "aws_sqs_queue" "{name}" {{\n  name = "{name}"\n  {extra}\n}}\n'


def _t(name: str, target: str | None) -> str:
    val = f"aws_sqs_queue.{target}.arn" if target else '"none"'
    return f'resource "aws_sns_topic" "{name}" {{\n  x = {val}\n}}\n'


def test_propagation_learns_only_from_earlier_commits(make_git_repo):
    """Two removals with a dependent block each. The first removal has no earlier evidence
    (conf 0 although its dependent changes in that very commit); the second sees 1/1."""
    repo = make_git_repo(
        [
            {
                "a.tf": _q("a"),
                "b.tf": _t("b", "a"),
                "c.tf": _q("c"),
                "d.tf": _t("d", "c"),
                "e.tf": _q("e"),
                "f.tf": _q("f"),
            },
            {"e.tf": _q("e", 'tags = { v = "1" }'), "f.tf": _q("f", 'tags = { v = "1" }')},
            {"a.tf": None, "b.tf": _t("b", None)},  # remove a, fix its dependent b
            {"c.tf": None, "d.tf": _t("d", None)},  # remove c, fix its dependent d
        ]
    )
    res = process_repo(repo.path, "t", _cfg(), blobless=False)
    feats = res.features.merge(res.queries[["query_id", "idx", "seed_address"]], on="query_id")

    def val(idx: int, seed: str, cand: str) -> tuple[float, int]:
        r = feats[(feats.idx == idx) & (feats.seed_address == seed) & (feats.cand_id == cand)]
        assert len(r) == 1
        return float(r.graph_propagation_conf.iloc[0]), int(r.label.iloc[0])

    # commit 2 removes a: its dependent b changes in that very commit, but conf must still be 0
    assert val(2, ".::resource.aws_sqs_queue.a", ".::resource.aws_sns_topic.b") == (0.0, 1)
    # commit 3 removes c: one earlier (removed, incoming) observation, and it changed -> 1/1
    assert val(3, ".::resource.aws_sqs_queue.c", ".::resource.aws_sns_topic.d") == (1.0, 1)
    # non-neighbors never get the propagation value
    other = feats[(feats.idx == 3) & (feats.cand_id == ".::resource.aws_sqs_queue.e")]
    assert len(other) > 0 and (other.graph_propagation_conf == 0).all()


# ---------------------------------------------------------- content similarity (TF-IDF)


def _snapshot_texts(hist: list[dict[str, str | None]], upto: int) -> dict[str, str]:
    files: dict[str, str] = {}
    for commit in hist[: upto + 1]:
        for path, text in commit.items():
            if text is None:
                files.pop(path, None)
            else:
                files[path] = text
    return files


def _tfidf(tokens: Counter, df: Counter, n: int) -> dict[str, float]:
    return {
        t: (1 + math.log(c)) * (math.log((1 + n) / (1 + df.get(t, 0))) + 1)
        for t, c in tokens.items()
    }


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return sum(v * b.get(t, 0.0) for t, v in a.items()) / (na * nb) if na and nb else 0.0


def test_content_similarity_uses_parent_bodies_and_parent_idf(repo):
    """Independent recomputation: seed = its body after t, candidates = bodies at parent(t),
    IDF over all blocks at parent(t). Candidates edited in t must use their old text."""
    hist = _history()
    res = process_repo(repo.path, "t", _cfg(), blobless=False)
    feats = res.features.merge(res.queries[["query_id", "idx", "seed_address"]], on="query_id")
    feats = feats[feats.cand_kind == "existing"]
    n_edited_cands = 0
    for idx in sorted(feats.idx.unique()):
        parent = {}
        for path, text in _snapshot_texts(hist, idx - 1).items():
            parent.update(parse_blocks(text, path))
        after = {}
        for path, text in _snapshot_texts(hist, idx).items():
            after.update(parse_blocks(text, path))
        tok = {a: Counter(re.findall(r"[a-z0-9]+", b.body_norm.lower())) for a, b in parent.items()}
        df: Counter = Counter()
        for t in tok.values():
            df.update(t.keys())
        for row in feats[feats.idx == idx].itertuples():
            seed = after.get(row.seed_address) or parent[row.seed_address]
            sv = _tfidf(Counter(re.findall(r"[a-z0-9]+", seed.body_norm.lower())), df, len(tok))
            expected = _cos(sv, _tfidf(tok[row.cand_id], df, len(tok)))
            assert row.text_content_similarity == pytest.approx(expected, abs=1e-5), (
                idx,
                row.seed_address,
                row.cand_id,
            )
            edited = row.cand_id in after and after[row.cand_id].body_hash != (
                parent[row.cand_id].body_hash
            )
            n_edited_cands += edited
    assert n_edited_cands > 5  # candidates changed in t were checked against their old text


def test_content_index_incremental_equals_rebuild():
    v1 = parse_blocks(_res("a", "x = var.env") + _res("b"), "main.tf")
    v2 = parse_blocks(_res("a", "x = var.region", 3) + _res("c"), "main.tf")
    inc = ContentIndex()
    inc.rebuild(v1)
    inc.update(v1, v2)
    fresh = ContentIndex()
    fresh.rebuild(v2)
    assert inc.n_docs == fresh.n_docs == 2
    assert +inc.df == +fresh.df  # unary + drops zero counts


def test_cosine_identical_and_disjoint_bodies():
    blocks = parse_blocks(_res("a") + _res("b") + 'variable "z" {}\n', "main.tf")
    idx = ContentIndex()
    idx.rebuild(blocks)
    a, b = blocks[".::resource.aws_s3_bucket.a"], blocks[".::resource.aws_s3_bucket.b"]
    assert cosine(idx.vector(a), idx.vector(a)) == pytest.approx(1.0)
    assert 0 < cosine(idx.vector(a), idx.vector(b)) < 1
    assert cosine(idx.vector(a), idx.vector(blocks[".::var.z"])) == 0.0


# --------------------------------------------------------------- Adamic-Adar weights


def test_adamic_adar_weights_favour_rare_neighbors():
    w = adamic_adar_weights({"hub", "rare", "leaf"}, {"hub": 80, "rare": 3, "leaf": 1})
    assert w["rare"] == pytest.approx(1 / math.log(3))
    assert w["hub"] == pytest.approx(1 / math.log(80))
    assert w["rare"] > w["hub"]
    assert w["leaf"] == pytest.approx(1 / math.log(2))  # degree floored at 2, no ln(1) = 0


def test_file_confidence_counts():
    m = AssociationRuleMiner(recency_tau=100)
    m.update(1, [_item("A", "~"), _item("B", "~")])  # both in main.tf
    m.update(2, [_item("A", "~"), _item_in("C", "~", "iam.tf")])
    m.update(3, [_item("A", "~")])
    assert m.file_confidence("main.tf", "main.tf") == pytest.approx(1 / 3)
    assert m.file_confidence("main.tf", "iam.tf") == pytest.approx(1 / 3)
    assert m.file_confidence("iam.tf", "main.tf") == pytest.approx(1.0)
    assert m.file_confidence("new.tf", "main.tf") == 0.0


def _item_in(addr: str, action: str, file: str) -> ChangeItem:
    return ChangeItem(addr, "resource", "aws_s3_bucket", addr, action, file, None, None)


def test_default_model_features_are_stored_columns():
    names = validate_feature_names(load_config().model.features)
    assert len(names) == 12
    assert not any(n.startswith("loc_") for n in names)
    with pytest.raises(ValueError):
        validate_feature_names(["ref_direct"])
