from __future__ import annotations

from collections.abc import Callable

from tfimpact.processing.hcl_parser import parse_blocks
from tfimpact.processing.references import extract_refs, make_address

D = "."


def refs(text: str, d: str = D) -> set[str]:
    return set(extract_refs(text, d))


def a(local: str, d: str = D) -> str:
    return make_address(d, local)


def test_make_address():
    assert make_address("modules/x", "var.a") == "modules/x::var.a"


def test_resource_ref():
    assert refs("${aws_s3_bucket.this.id}") == {a("resource.aws_s3_bucket.this")}


def test_data_ref_not_double_counted_as_resource():
    r = refs("${data.aws_iam_policy_document.doc.json}")
    assert r == {a("data.aws_iam_policy_document.doc")}


def test_module_var_local_refs():
    r = refs("${module.net.vpc_id} ${var.cidr} ${local.tags}")
    assert r == {a("module.net"), a("var.cidr"), a("local.tags")}


def test_hyphenated_names():
    assert refs("${var.my-var}") == {a("var.my-var")}


def test_string_template_ref():
    assert refs('"${aws_s3_bucket.this.arn}/*"') == {a("resource.aws_s3_bucket.this")}


def test_multiple_refs_in_template():
    r = refs('"${var.a}-${local.b}-${aws_x_y.z.id}"')
    assert r == {a("var.a"), a("local.b"), a("resource.aws_x_y.z")}


def test_depends_on_list():
    r = refs('{"depends_on":["${aws_s3_bucket.this}","${module.net}"]}')
    assert r == {a("resource.aws_s3_bucket.this"), a("module.net")}


def test_no_match_mid_traversal():
    # `b_c.d` follows a dot, so it must not be read as resource b_c.d
    assert refs("${module.a.b_c.d}") == {a("module.a")}


def test_no_match_mid_traversal_resource():
    r = refs("${aws_s3_bucket.this.tags_all.x}")
    assert r == {a("resource.aws_s3_bucket.this")}


def test_non_reference_tokens_produce_nothing():
    for tok in ("${each.value}", "${each.key}", "${count.index}", "${path.module}", "${self.arn}"):
        assert refs(tok) == set(), tok


def test_type_without_underscore_is_not_resource():
    assert refs("${foo.bar}") == set()


def test_refs_are_scoped_to_given_directory():
    r = refs("${var.x} ${aws_s3_bucket.b.id}", "modules/net")
    assert r == {"modules/net::var.x", "modules/net::resource.aws_s3_bucket.b"}


def test_refs_scoped_in_parse_blocks():
    text = 'resource "aws_s3_bucket" "b" {\n  bucket = var.x\n}\n'
    in_root = parse_blocks(text, "main.tf")[".::resource.aws_s3_bucket.b"]
    in_sub = parse_blocks(text, "modules/n/main.tf")["modules/n::resource.aws_s3_bucket.b"]
    assert in_root.refs == {".::var.x"}
    assert in_sub.refs == {"modules/n::var.x"}


def test_self_reference_excluded_via_parse_blocks():
    text = 'resource "aws_s3_bucket" "b" {\n  bucket = "${aws_s3_bucket.b.id}-${var.x}"\n}\n'
    b = parse_blocks(text, "main.tf")[".::resource.aws_s3_bucket.b"]
    assert b.refs == {".::var.x"}


def test_fixture_refs(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("main.tf"), "main.tf")
    pol = blocks[".::resource.aws_s3_bucket_policy.p"]
    assert pol.refs == {
        ".::resource.aws_s3_bucket.this",
        ".::data.aws_iam_policy_document.doc",
        ".::module.net",
    }
    doc = blocks[".::data.aws_iam_policy_document.doc"]
    assert doc.refs == {".::resource.aws_s3_bucket.this"}
    assert blocks[".::resource.aws_instance.web"].refs == {".::var.enabled"}
    # each.* / ingress.value produce nothing
    assert blocks[".::resource.aws_security_group.sg"].refs == {".::var.rules"}


def test_fixture_non_refs(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("refs_misc.tf"), "refs_misc.tf")
    fn = blocks[".::resource.aws_lambda_function.f"]
    assert fn.refs == {".::module.net"}


def test_fixture_locals_and_vars(fixture_text: Callable[[str], str]):
    blocks = parse_blocks(fixture_text("vars.tf"), "vars.tf")
    assert blocks[".::local.other"].refs == {".::local.tags"}
    assert blocks[".::local.tags"].refs == {".::var.prefix"}
    assert blocks[".::output.bucket_arn"].refs == {".::resource.aws_s3_bucket.this"}
