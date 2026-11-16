from __future__ import annotations

import posixpath
import re

from tfimpact.constants import ADDR_SEP, RESOURCE_TYPE_RE, ROOT_DIR

_IDENT = r"[A-Za-z_][\w-]*"
# (?<![\w.]) prevents matching mid-traversal, e.g. the `this.arn` in `aws_x.this.arn`
# or the `aws_x.y` inside `data.aws_x.y`.
_LEAD = r"(?<![\w.$-])"
_DATA_RE = re.compile(rf"{_LEAD}data\.({_IDENT})\.({_IDENT})")
_MODULE_RE = re.compile(rf"{_LEAD}module\.({_IDENT})")
_VAR_RE = re.compile(rf"{_LEAD}var\.({_IDENT})")
_LOCAL_RE = re.compile(rf"{_LEAD}local\.({_IDENT})")
_RES_RE = re.compile(rf"{_LEAD}([a-z0-9]+_[a-z0-9_]+)\.({_IDENT})")


def make_address(directory: str, local_addr: str) -> str:
    return f"{directory}{ADDR_SEP}{local_addr}"


def extract_refs(body_text: str, directory: str) -> frozenset[str]:
    """Extract the different Terraform types of blocks: all addresses (same directory) syntactically referenced in `body_text`."""
    refs: set[str] = set()
    for t, n in _DATA_RE.findall(body_text):
        refs.add(make_address(directory, f"data.{t}.{n}"))
    for n in _MODULE_RE.findall(body_text):
        refs.add(make_address(directory, f"module.{n}"))
    for n in _VAR_RE.findall(body_text):
        refs.add(make_address(directory, f"var.{n}"))
    for n in _LOCAL_RE.findall(body_text):
        refs.add(make_address(directory, f"local.{n}"))
    for t, n in _RES_RE.findall(body_text):
        if RESOURCE_TYPE_RE.match(t):
            refs.add(make_address(directory, f"resource.{t}.{n}"))
    return frozenset(refs)


# module.<name>.<output>, optionally with an index: module.x[0].out / module.x["k"].out
_MODULE_OUT_RE = re.compile(rf"{_LEAD}module\.({_IDENT})(?:\[[^\]]*\])?\.({_IDENT})")


def extract_module_output_refs(body_text: str, directory: str) -> frozenset[tuple[str, str]]:
    """(module address, output name) pairs read via `module.<name>.<output>`."""
    return frozenset(
        (make_address(directory, f"module.{m}"), o) for m, o in _MODULE_OUT_RE.findall(body_text)
    )


def resolve_local_source(directory: str, source: object) -> str | None:
    """Directory of a local module source (`./x`, `../x`) relative to the repo root, else None.

    Remote sources (registry, git, http, s3 ...) and paths escaping the repo return None.
    """
    if not isinstance(source, str) or not source.startswith(("./", "../")):
        return None
    base = "" if directory == ROOT_DIR else directory
    resolved = posixpath.normpath(posixpath.join(base, source))
    if resolved.startswith(".."):
        return None
    return ROOT_DIR if resolved in ("", ".") else resolved
