from __future__ import annotations

from collections.abc import Callable

import pytest

from tfimpact.processing.hcl_parser import module_dir, parse_blocks


def test_module_dir():
    assert module_dir("main.tf") == "."
    assert module_dir("modules/net/main.tf") == "modules/net"


def test_addresses_and_kinds(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("main.tf"), "main.tf")
    assert ".::resource.aws_s3_bucket.this" in blocks
    b = blocks[".::resource.aws_s3_bucket.this"]
    assert (b.kind, b.type, b.name) == ("resource", "aws_s3_bucket", "this")
    d = blocks[".::data.aws_iam_policy_document.doc"]
    assert (d.kind, d.type, d.name) == ("data", "data.aws_iam_policy_document", "doc")
    m = blocks[".::module.net"]
    assert (m.kind, m.type, m.name) == ("module", "module", "net")


def test_variable_output_in_subdir(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("vars.tf"), "modules/net/vars.tf")
    v = blocks["modules/net::var.prefix"]
    assert (v.kind, v.type, v.name) == ("variable", "variable", "prefix")
    assert v.file == "modules/net/vars.tf"
    assert v.directory == "modules/net"
    o = blocks["modules/net::output.bucket_arn"]
    assert (o.kind, o.type, o.name) == ("output", "output", "bucket_arn")


def test_root_directory_is_dot(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("vars.tf"), "vars.tf")
    assert all(b.directory == "." and b.address.startswith(".::") for b in blocks.values())
    assert all(b.file == "vars.tf" for b in blocks.values())


def test_one_block_per_local_across_multiple_locals_blocks(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("vars.tf"), "vars.tf")
    locs = {a for a, b in blocks.items() if b.kind == "local"}
    assert locs == {".::local.tags", ".::local.other", ".::local.third"}
    assert blocks[".::local.third"].type == "local"
    assert blocks[".::local.third"].name == "third"


def test_block_counts(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("main.tf"), "main.tf")
    assert len(blocks) == 6  # 4 resources + data + module


def test_heredoc_and_nested_blocks_parse(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("main.tf"), "main.tf")
    web = blocks[".::resource.aws_instance.web"]
    assert "echo hello" in web.body_norm
    sg = blocks[".::resource.aws_security_group.sg"]
    assert "ingress" in sg.body_norm and "tcp" in sg.body_norm


def test_terraform_and_provider_blocks_ignored():
    text = 'terraform {\n  required_version = ">= 1.0"\n}\nprovider "aws" {\n  region = "x"\n}\n'
    assert parse_blocks(text, "main.tf") == {}


BASE = 'resource "aws_s3_bucket" "this" {\n  bucket = "b"\n  acl    = "private"\n}\n'


def test_formatting_only_edit_same_hash():
    reformatted = (
        'resource   "aws_s3_bucket"   "this"   {\n'
        '    bucket="b"\n\n\n    acl         =     "private"\n}\n'
    )
    a = parse_blocks(BASE, "a.tf")[".::resource.aws_s3_bucket.this"]
    b = parse_blocks(reformatted, "a.tf")[".::resource.aws_s3_bucket.this"]
    assert a.body_hash == b.body_hash
    assert a.body_norm == b.body_norm


def test_attribute_order_same_hash():
    swapped = 'resource "aws_s3_bucket" "this" {\n  acl = "private"\n  bucket = "b"\n}\n'
    a = parse_blocks(BASE, "a.tf")[".::resource.aws_s3_bucket.this"]
    b = parse_blocks(swapped, "a.tf")[".::resource.aws_s3_bucket.this"]
    assert a.body_hash == b.body_hash


def test_comment_only_edit_same_hash():
    commented = (
        "# header\n"
        'resource "aws_s3_bucket" "this" {\n'
        '  bucket = "b" # trailing\n'
        "  // another\n"
        '  /* block */ acl = "private"\n'
        "}\n"
    )
    a = parse_blocks(BASE, "a.tf")[".::resource.aws_s3_bucket.this"]
    b = parse_blocks(commented, "a.tf")[".::resource.aws_s3_bucket.this"]
    assert a.body_hash == b.body_hash


def test_expression_whitespace_normalized():
    t1 = "locals {\n  a = var.create?1:0\n}\n"
    t2 = "locals {\n  a = var.create ? 1 : 0\n}\n"
    assert (
        parse_blocks(t1, "a.tf")[".::local.a"].body_hash
        == parse_blocks(t2, "a.tf")[".::local.a"].body_hash
    )


def test_attribute_change_changes_hash():
    changed = BASE.replace('"private"', '"public-read"')
    a = parse_blocks(BASE, "a.tf")[".::resource.aws_s3_bucket.this"]
    b = parse_blocks(changed, "a.tf")[".::resource.aws_s3_bucket.this"]
    assert a.body_hash != b.body_hash


def test_same_body_different_address_same_hash_different_file_dir():
    a = parse_blocks(BASE, "a.tf")[".::resource.aws_s3_bucket.this"]
    b = parse_blocks(BASE, "x/a.tf")["x::resource.aws_s3_bucket.this"]
    assert a.body_hash == b.body_hash and a.address != b.address


def test_unrelated_block_edit_does_not_change_other_hash():
    t1 = BASE + 'variable "v" {\n  default = 1\n}\n'
    t2 = BASE + 'variable "v" {\n  default = 2\n}\n'
    a, b = parse_blocks(t1, "a.tf"), parse_blocks(t2, "a.tf")
    assert (
        a[".::resource.aws_s3_bucket.this"].body_hash
        == b[".::resource.aws_s3_bucket.this"].body_hash
    )
    assert a[".::var.v"].body_hash != b[".::var.v"].body_hash


@pytest.mark.parametrize(
    "bad", ['resource "aws_x" "y" {\n  a = \n', "this is { not hcl", 'variable "x" {']
)
def test_invalid_hcl_raises(bad: str):
    with pytest.raises(Exception):  # noqa: B017 - lark error types vary
        parse_blocks(bad, "bad.tf")
