from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pandas as pd
import pytest

from tfimpact import config as config_mod
from tfimpact.processing import gitio
from tfimpact.processing.pipeline import process_repo

from .conftest import GitRepo


def res(name: str, val: str = "x") -> str:
    return f'resource "aws_s3_bucket" "{name}" {{\n  bucket = "{val}"\n}}\n'


MakeRepo = Callable[..., GitRepo]


@pytest.fixture
def scenario(make_git_repo: MakeRepo) -> GitRepo:
    """Linear history + one --no-ff merge.

    idx: 0 start | 1 +c (2 mods? see below) ... documented inline.
    """
    r = make_git_repo()
    # 0: start commit with a, b, c
    r.commit({"main.tf": res("a") + res("b") + res("c")})
    # 1: modify a and b -> 2 items (usable)
    r.commit({"main.tf": res("a", "1") + res("b", "1") + res("c")})
    # 2: modify only a -> too_few_items
    r.commit({"main.tf": res("a", "2") + res("b", "1") + res("c")})
    # 3: non-tf only -> no_tf_change, 0 tf files
    r.commit({"README.md": "hi"})
    # 4: .tf.json and .tfvars ignored
    r.commit({"x.tf.json": "{}", "vals.tfvars": 'a = "b"\n'})
    # 5: add two resources in a new file -> usable (+2)
    r.commit({"more.tf": res("d") + res("e")})
    # 6: formatting-only edit -> tf file touched, 0 items
    r.commit(
        {"more.tf": '# comment\nresource "aws_s3_bucket" "d" {\n  bucket   =  "x"\n}\n' + res("e")}
    )
    # 7: move block e from more.tf to main.tf (same dir) + modify d -> only d modified: 1 item
    r.commit(
        {
            "more.tf": res("d", "7"),
            "main.tf": res("a", "2") + res("b", "1") + res("c") + res("e"),
        }
    )
    # 8: delete more.tf, add var in main -> -d, +var -> usable
    r.commit(
        {
            "more.tf": None,
            "main.tf": res("a", "2")
            + res("b", "1")
            + res("c")
            + res("e")
            + 'variable "v" {\n  default = 1\n}\n',
        }
    )
    # 9: invalid hcl in new file -> parse error, 0 items
    r.commit({"broken.tf": 'resource "aws_x" "y" {\n  a = \n'})
    # 10: modify b, c on main (usable)
    r.commit(
        {
            "main.tf": res("a", "2")
            + res("b", "10")
            + res("c", "10")
            + res("e")
            + 'variable "v" {\n  default = 1\n}\n'
        }
    )
    # 11: merge of side branch changing 2 blocks -> usable merge
    r.merge_side_branch(
        {
            "main.tf": res("a", "m")
            + res("b", "10")
            + res("c", "m")
            + res("e")
            + 'variable "v" {\n  default = 1\n}\n'
        }
    )
    return r


def test_first_parent_history_oldest_first_and_merge_detected(scenario: GitRepo):
    hist = gitio.first_parent_history(scenario.path, 2000)
    assert [c.sha for c in hist] == scenario.shas
    assert hist[0].parents == ()
    assert [c.is_merge for c in hist] == [False] * (len(hist) - 1) + [True]
    assert len(hist[-1].parents) == 2


def test_side_commit_not_in_first_parent_history(make_git_repo: MakeRepo):
    r = make_git_repo([{"a.tf": res("a")}])
    side, merge = r.merge_side_branch({"b.tf": res("b")})
    shas = [c.sha for c in gitio.first_parent_history(r.path, 100)]
    assert side not in shas and merge in shas and len(shas) == 2


def test_merge_diffed_against_first_parent_only(make_git_repo: MakeRepo):
    r = make_git_repo([{"a.tf": res("a")}])
    # main advances after the branch point; the side branch does not have it
    r._git("checkout", "-q", "-b", "side")
    r.shas, saved = [], r.shas
    r.commit({"side.tf": res("s")})
    r.shas = saved
    r._git("checkout", "-q", "main")
    r.commit({"main_only.tf": res("m")})
    r._git("merge", "-q", "--no-ff", "-m", "merge", "side")
    r.shas.append(r.head())
    hist = gitio.first_parent_history(r.path, 100)
    merge = hist[-1]
    assert merge.is_merge
    # vs first parent (main), only side.tf is new; main_only.tf must not show up
    assert [(c.status, c.path) for c in merge.changes] == [("A", "side.tf")]


def test_history_ignores_non_tf_and_tf_json(make_git_repo: MakeRepo):
    r = make_git_repo(
        [
            {"a.tf": res("a")},
            {"b.tf.json": "{}", "c.tfvars": "x=1", "README.md": "x", ".terraform.lock.hcl": "x"},
            {"sub/d.tf": res("d")},
        ]
    )
    hist = gitio.first_parent_history(r.path, 100)
    assert [[c.path for c in h.changes] for h in hist] == [["a.tf"], [], ["sub/d.tf"]]


def test_file_change_statuses(make_git_repo: MakeRepo):
    r = make_git_repo([{"a.tf": res("a"), "b.tf": res("b")}, {"a.tf": res("a", "2"), "b.tf": None}])
    hist = gitio.first_parent_history(r.path, 100)
    by_path = {c.path: c for c in hist[1].changes}
    assert by_path["a.tf"].status == "M" and by_path["a.tf"].old_oid and by_path["a.tf"].new_oid
    assert by_path["b.tf"].status == "D" and by_path["b.tf"].new_oid is None
    assert [c.status for c in hist[0].changes] == ["A", "A"]
    assert hist[0].changes[0].old_oid is None


def test_max_commits_window(make_git_repo: MakeRepo):
    r = make_git_repo([{"a.tf": res(str(i))} for i in range(5)])
    hist = gitio.first_parent_history(r.path, 3)
    assert [c.sha for c in hist] == r.shas[-3:]


def test_ls_tree_tf(make_git_repo: MakeRepo):
    r = make_git_repo(
        [{"a.tf": res("a"), "sub/b.tf": res("b"), "c.tf.json": "{}", "d.txt": "x", "e.tfvars": ""}]
    )
    tree = gitio.ls_tree_tf(r.path, "HEAD")
    assert set(tree) == {"a.tf", "sub/b.tf"}
    assert all(len(oid) == 40 for oid in tree.values())


def test_blob_reader_reads_content_and_missing(make_git_repo: MakeRepo):
    text = res("a") + "\n# unicode: é\n"
    r = make_git_repo([{"a.tf": text, "b.tf": "second"}])
    tree = gitio.ls_tree_tf(r.path, "HEAD")
    with gitio.BlobReader(r.path) as br:
        assert br.read(tree["a.tf"]).decode() == text
        assert br.read(tree["b.tf"]) == b"second"
        assert br.read(tree["a.tf"]).decode() == text  # reuse after another read
        with pytest.raises(gitio.GitError):
            br.read("0" * 40)


def test_count_tf_commits(scenario: GitRepo):
    hist = gitio.first_parent_history(scenario.path, 2000)
    with_merges = gitio.count_tf_commits(scenario.path, 2000, include_merges=True)
    without = gitio.count_tf_commits(scenario.path, 2000, include_merges=False)
    assert with_merges == without + 1
    assert with_merges == sum(1 for c in hist[1:] if c.changes)


# ---------------------------------------------------------------------------------------------
# process_repo
# ---------------------------------------------------------------------------------------------


def test_process_repo_commits_table(scenario: GitRepo):
    cfg = config_mod.load_config()
    res_ = process_repo(scenario.path, "test", cfg, blobless=False)
    c = res_.commits.set_index("idx")
    assert len(c) == len(scenario.shas)
    assert list(c.sha) == scenario.shas
    assert (c.repo == "test").all()

    assert not c.loc[0, "usable"]
    expected = {
        1: (True, None, 2),
        2: (False, "too_few_items", 1),
        3: (False, "no_tf_change", 0),
        4: (False, "no_tf_change", 0),
        5: (True, None, 2),
        6: (False, "no_tf_change", 0),
        7: (False, "too_few_items", 1),
        8: (True, None, 2),
        9: (False, "no_tf_change", 0),
        10: (True, None, 2),
        11: (True, None, 2),
    }
    for idx, (usable, reason, n_items) in expected.items():
        assert bool(c.loc[idx, "usable"]) is usable, idx
        got = c.loc[idx, "skip_reason"]
        assert (None if pd.isna(got) else got) == reason, idx
        assert c.loc[idx, "n_items"] == n_items, idx

    assert c.loc[3, "n_tf_files"] == 0 and c.loc[4, "n_tf_files"] == 0
    assert c.loc[6, "n_tf_files"] == 1  # formatting-only still touches a .tf file
    assert bool(c.loc[11, "is_merge"]) and not c.loc[:10, "is_merge"].any()

    # 5 usable -> warmup int(5 * 0.2) = 1 -> only the first usable commit (idx 1)
    assert list(c.index[c.in_warmup]) == [1]
    assert not c.loc[0, "in_warmup"]


def test_process_repo_changes_table(scenario: GitRepo):
    res_ = process_repo(scenario.path, "test", config_mod.load_config(), blobless=False)
    ch = res_.changes

    def rows(idx: int) -> set[tuple[str, str]]:
        sub = ch[ch.idx == idx]
        return set(zip(sub.action, sub.address, strict=True))

    assert rows(1) == {("~", ".::resource.aws_s3_bucket.a"), ("~", ".::resource.aws_s3_bucket.b")}
    assert rows(5) == {("+", ".::resource.aws_s3_bucket.d"), ("+", ".::resource.aws_s3_bucket.e")}
    assert rows(6) == set()
    # block e moved within the directory: no add/remove, only d modified
    assert rows(7) == {("~", ".::resource.aws_s3_bucket.d")}
    assert rows(8) == {("-", ".::resource.aws_s3_bucket.d"), ("+", ".::var.v")}
    assert rows(9) == set()
    assert rows(11) == {("~", ".::resource.aws_s3_bucket.a"), ("~", ".::resource.aws_s3_bucket.c")}
    assert set(ch.columns) == {"repo", "idx", "address", "kind", "type", "action", "file"}
    row = ch[(ch.idx == 8) & (ch.action == "+")].iloc[0]
    assert (row.kind, row.type, row.file) == ("variable", "variable", "main.tf")


def test_process_repo_parse_errors_and_stats(scenario: GitRepo):
    res_ = process_repo(scenario.path, "test", config_mod.load_config(), blobless=False)
    pe = res_.parse_errors
    assert len(pe) == 1
    assert pe.iloc[0].file == "broken.tf" and pe.iloc[0].idx == 9
    s = res_.stats
    assert s["n_commits"] == 12
    assert s["n_merges"] == 1
    assert s["n_usable"] == 5 and s["n_usable_merge"] == 1
    assert s["n_query_commits"] == 4
    assert 0 < s["parse_rate"] < 1
    assert s["n_dirs_final"] == 1


def test_process_repo_merges_disabled(scenario: GitRepo):
    cfg = config_mod.load_config()
    cfg = dataclasses.replace(cfg, mining=dataclasses.replace(cfg.mining, use_merge_commits=False))
    c = process_repo(scenario.path, "test", cfg, blobless=False).commits.set_index("idx")
    assert c.loc[11, "skip_reason"] == "merge" and not c.loc[11, "usable"]
    # 4 usable now -> warmup int(4 * 0.2) = 0
    assert int(c.usable.sum()) == 4
    assert not c.in_warmup.any()


def test_process_repo_warmup_frac_zero_and_half(scenario: GitRepo):
    base = config_mod.load_config()
    for frac, n_warm in [(0.0, 0), (0.5, 2)]:
        cfg = dataclasses.replace(base, mining=dataclasses.replace(base.mining, warmup_frac=frac))
        c = process_repo(scenario.path, "test", cfg, blobless=False).commits
        assert int(c.in_warmup.sum()) == n_warm
        # warmup flags are only ever set on the earliest usable commits
        usable_idx = list(c.idx[c.usable])
        assert list(c.idx[c.in_warmup]) == usable_idx[:n_warm]
        assert not (c.in_warmup & ~c.usable).any()


def test_process_repo_max_items_filter(make_git_repo: MakeRepo):
    r = make_git_repo([{"a.tf": res("a")}, {"a.tf": "".join(res(f"r{i}") for i in range(31))}])
    c = process_repo(r.path, "t", config_mod.load_config(), blobless=False).commits.set_index("idx")
    assert (
        c.loc[1, "skip_reason"] == "too_many_items" and c.loc[1, "n_items"] == 32
    )  # 31 added + `a` removed


def test_process_repo_empty_history_raises(make_git_repo: MakeRepo):
    r = make_git_repo()
    with pytest.raises(Exception):  # noqa: B017 - git log on an unborn branch
        process_repo(r.path, "t", config_mod.load_config(), blobless=False)


def test_process_repo_start_commit_has_skip_reason(scenario: GitRepo):
    c = process_repo(scenario.path, "test", config_mod.load_config(), blobless=False).commits
    assert c.skip_reason.iloc[0] == "start_commit"
