from __future__ import annotations

from tfimpact.processing.diff import diff_blocks
from tfimpact.processing.hcl_parser import parse_blocks


def P(text: str, path: str = "main.tf"):
    return parse_blocks(text, path)


A = 'resource "aws_s3_bucket" "a" {\n  bucket = "a"\n}\n'
B = 'resource "aws_dynamodb_table" "t" {\n  name = "t"\n}\n'


def test_add():
    items = diff_blocks(P(A), P(A + B))
    assert len(items) == 1
    it = items[0]
    assert it.action == "+" and it.address == ".::resource.aws_dynamodb_table.t"
    assert it.old is None and it.new is not None
    assert it.type_item == "+aws_dynamodb_table"
    assert (it.kind, it.name, it.file) == ("resource", "t", "main.tf")


def test_remove():
    items = diff_blocks(P(A + B), P(A))
    assert [(i.action, i.address) for i in items] == [("-", ".::resource.aws_dynamodb_table.t")]
    assert items[0].new is None and items[0].old is not None
    assert items[0].type_item == "-aws_dynamodb_table"


def test_modify():
    items = diff_blocks(P(A), P(A.replace('"a"\n', '"b"\n')))
    assert len(items) == 1
    it = items[0]
    assert it.action == "~" and it.type_item == "~aws_s3_bucket"
    assert it.old is not None and it.new is not None
    assert it.old.body_hash != it.new.body_hash


def test_formatting_only_change_yields_nothing():
    reformatted = '# c\nresource "aws_s3_bucket" "a" {\n      bucket   =   "a"   # x\n}\n'
    assert diff_blocks(P(A), P(reformatted)) == []


def test_no_change_empty():
    assert diff_blocks(P(A + B), P(A + B)) == []


def test_mixed_sorted_by_address():
    before = P(A + B)
    after = P(A.replace('"a"\n', '"z"\n') + 'variable "v" {\n  default = 1\n}\n')
    items = diff_blocks(before, after)
    assert [(i.action, i.address) for i in items] == [
        ("-", ".::resource.aws_dynamodb_table.t"),
        ("~", ".::resource.aws_s3_bucket.a"),
        ("+", ".::var.v"),
    ]
    assert items[2].type_item == "+variable"


def test_data_type_item_has_data_prefix():
    d = 'data "aws_caller_identity" "me" {}\n'
    (it,) = diff_blocks({}, P(d))
    assert it.type_item == "+data.aws_caller_identity"


def test_rename_is_remove_plus_add():
    renamed = A.replace('"a" {', '"a2" {')
    items = diff_blocks(P(A), P(renamed))
    assert sorted(i.action for i in items) == ["+", "-"]
