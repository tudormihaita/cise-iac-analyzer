from __future__ import annotations

from tfimpact.processing.diff import diff_blocks
from tfimpact.processing.gitio import FileChange
from tfimpact.processing.hcl_parser import Block, parse_blocks
from tfimpact.processing.snapshot import ParseFailure, Snapshot


def make_loader(contents: dict[str, str], fail: set[str] | None = None):
    """oid -> text map; path comes from the caller."""
    fail = fail or set()

    def loader(path: str, oid: str) -> dict[str, Block]:
        if oid in fail:
            raise ParseFailure(path, oid, "boom")
        return parse_blocks(contents[oid], path)

    return loader


R1 = 'resource "aws_s3_bucket" "a" {\n  bucket = "a"\n}\n'
R2 = 'resource "aws_s3_bucket" "b" {\n  bucket = "b"\n}\n'
R1m = 'resource "aws_s3_bucket" "a" {\n  bucket = "changed"\n}\n'


def add(path: str, oid: str) -> FileChange:
    return FileChange("A", path, None, oid)


def mod(path: str, old: str, new: str) -> FileChange:
    return FileChange("M", path, old, new)


def dele(path: str, old: str) -> FileChange:
    return FileChange("D", path, old, None)


def test_from_tree_builds_state():
    contents = {"o1": R1, "o2": R2}
    snap, errs = Snapshot.from_tree({"a.tf": "o1", "m/b.tf": "o2"}, make_loader(contents))
    assert errs == []
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.a", "m::resource.aws_s3_bucket.b"}
    assert snap.directories == [".", "m"]
    assert set(snap.dir_blocks("m")) == {"m::resource.aws_s3_bucket.b"}
    assert snap.dir_blocks("nope") == {}


def test_stage_leaves_state_untouched_until_apply():
    contents = {"o1": R1, "o2": R2}
    loader = make_loader(contents)
    snap, _ = Snapshot.from_tree({"a.tf": "o1"}, loader)
    staged = snap.stage([add("b.tf", "o2")], loader)
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.a"}
    assert "b.tf" not in snap.files
    assert set(staged.before) == {".::resource.aws_s3_bucket.a"}
    assert set(staged.after) == {".::resource.aws_s3_bucket.a", ".::resource.aws_s3_bucket.b"}
    assert [i.action for i in diff_blocks(staged.before, staged.after)] == ["+"]
    snap.apply(staged)
    assert set(snap.blocks) == set(staged.after)
    assert "b.tf" in snap.files


def test_modify_gives_modified_item():
    contents = {"o1": R1, "o2": R1m}
    loader = make_loader(contents)
    snap, _ = Snapshot.from_tree({"a.tf": "o1"}, loader)
    staged = snap.stage([mod("a.tf", "o1", "o2")], loader)
    (it,) = diff_blocks(staged.before, staged.after)
    assert it.action == "~"
    snap.apply(staged)
    assert "changed" in snap.blocks[".::resource.aws_s3_bucket.a"].body_norm


def test_block_moved_between_files_same_dir_is_no_change():
    contents = {"o1": R1 + R2, "o2": R2, "o3": R1}
    loader = make_loader(contents)
    snap, _ = Snapshot.from_tree({"a.tf": "o1"}, loader)
    # move block `a` from a.tf into new b.tf
    staged = snap.stage(
        [mod("a.tf", "o1", "o2"), add("b.tf", "o3")],
        loader,
    )
    assert diff_blocks(staged.before, staged.after) == []
    snap.apply(staged)
    assert snap.blocks[".::resource.aws_s3_bucket.a"].file == "b.tf"
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.a", ".::resource.aws_s3_bucket.b"}


def test_block_moved_between_directories_is_remove_plus_add():
    contents = {"o1": R1, "o2": R1}
    loader = make_loader(contents)
    snap, _ = Snapshot.from_tree({"a.tf": "o1"}, loader)
    staged = snap.stage([dele("a.tf", "o1"), add("sub/a.tf", "o2")], loader)
    assert sorted(i.action for i in diff_blocks(staged.before, staged.after)) == ["+", "-"]


def test_file_delete_removes_its_blocks():
    contents = {"o1": R1, "o2": R2}
    loader = make_loader(contents)
    snap, _ = Snapshot.from_tree({"a.tf": "o1", "b.tf": "o2"}, loader)
    staged = snap.stage([dele("a.tf", "o1")], loader)
    items = diff_blocks(staged.before, staged.after)
    assert [(i.action, i.address) for i in items] == [("-", ".::resource.aws_s3_bucket.a")]
    snap.apply(staged)
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.b"}
    assert "a.tf" not in snap.files


def test_deleting_last_file_in_dir_removes_directory():
    loader = make_loader({"o1": R1})
    snap, _ = Snapshot.from_tree({"m/a.tf": "o1"}, loader)
    snap.apply(snap.stage([dele("m/a.tf", "o1")], loader))
    assert snap.blocks == {}
    assert snap.directories == []


def test_parse_failure_keeps_previous_blocks():
    contents = {"o1": R1 + R2, "o2": "garbage"}
    loader = make_loader(contents, fail={"o2"})
    snap, errs = Snapshot.from_tree({"a.tf": "o1"}, loader)
    assert errs == []
    staged = snap.stage([mod("a.tf", "o1", "o2")], loader)
    assert len(staged.errors) == 1 and staged.errors[0].path == "a.tf"
    assert diff_blocks(staged.before, staged.after) == []
    snap.apply(staged)
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.a", ".::resource.aws_s3_bucket.b"}
    assert snap.files["a.tf"].parsed_ok is False
    assert snap.files["a.tf"].oid == "o2"


def test_parse_failure_on_new_file_yields_no_blocks():
    loader = make_loader({"o1": R1, "bad": "x"}, fail={"bad"})
    snap, _ = Snapshot.from_tree({"a.tf": "o1"}, loader)
    staged = snap.stage([add("b.tf", "bad")], loader)
    assert len(staged.errors) == 1
    snap.apply(staged)
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.a"}


def test_from_tree_reports_parse_failures():
    loader = make_loader({"o1": R1, "bad": "x"}, fail={"bad"})
    snap, errs = Snapshot.from_tree({"a.tf": "o1", "b.tf": "bad"}, loader)
    assert [e.path for e in errs] == ["b.tf"]
    assert set(snap.blocks) == {".::resource.aws_s3_bucket.a"}


def test_parse_failure_message():
    e = ParseFailure("a.tf", "abcdef123456", "bad")
    assert "a.tf@abcdef12" in str(e)


def test_empty_stage_noop():
    loader = make_loader({})
    snap, _ = Snapshot.from_tree({}, loader)
    staged = snap.stage([], loader)
    assert staged.before == {} and staged.after == {}
    snap.apply(staged)
    assert snap.blocks == {}
