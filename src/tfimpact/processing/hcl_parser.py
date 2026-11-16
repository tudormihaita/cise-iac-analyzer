from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import dataclass
from typing import Any

import hcl2
from hcl2.utils import SerializationOptions

from tfimpact.constants import (
    DATA,
    LOCAL,
    MODULE,
    OUTPUT,
    RESOURCE,
    ROOT_DIR,
    VARIABLE,
)
from tfimpact.processing.references import (
    extract_module_output_refs,
    extract_refs,
    make_address,
    resolve_local_source,
)

# module-block arguments that are meta-arguments, not input variables of the module
MODULE_META_ARGS = frozenset({"source", "version", "count", "for_each", "providers", "depends_on"})

# serialization: comments dropped, no block markers, unquoted strings,
# heredocs flattened -> formatting-only edits yield identical output
HCL_OPTIONS = SerializationOptions(
    with_comments=False,
    explicit_blocks=False,
    preserve_heredocs=False,
    strip_string_quotes=True,
)


@dataclass(frozen=True)
class Block:
    address: str
    kind: str
    type: str
    name: str
    file: str
    directory: str
    body_norm: str
    body_hash: str
    refs: frozenset[str]
    # module plumbing; empty for non-module blocks / remote sources
    mod_out_refs: frozenset[tuple[str, str]] = frozenset()  # (module address, output) read
    module_source_dir: str | None = None  # resolved local `source` of a module block
    module_args: frozenset[str] = frozenset()  # input variables set by a module block


def module_dir(path: str) -> str:
    d = posixpath.dirname(path)
    return d if d else ROOT_DIR


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def parse_hcl(text: str) -> dict[str, Any]:
    """Raw python-hcl2 dict (raises on syntax errors)."""
    return hcl2.loads(text, serialization_options=HCL_OPTIONS)


def _make_block(kind: str, btype: str, name: str, local_addr: str, body: Any, path: str) -> Block:
    directory = module_dir(path)
    body_norm = canonical_json(body)
    address = make_address(directory, local_addr)
    refs = extract_refs(body_norm, directory) - {address}
    src_dir, args = None, frozenset()
    if kind == MODULE and isinstance(body, dict):
        src_dir = resolve_local_source(directory, body.get("source"))
        if src_dir is not None:
            args = frozenset(k for k in body if k not in MODULE_META_ARGS)
    return Block(
        address=address,
        kind=kind,
        type=btype,
        name=name,
        file=path,
        directory=directory,
        body_norm=body_norm,
        body_hash=hashlib.blake2b(body_norm.encode(), digest_size=12).hexdigest(),
        refs=refs,
        mod_out_refs=extract_module_output_refs(body_norm, directory),
        module_source_dir=src_dir,
        module_args=args,
    )


def _iter_labeled(entries: Any) -> Any:
    """Yield (label, body) from `[{label: body}, ...]`."""
    for entry in entries or []:
        if isinstance(entry, dict):
            yield from entry.items()


def blocks_from_parsed(parsed: dict[str, Any], path: str) -> dict[str, Block]:
    """Convert a parsed file into blocks. Unknown top-level keywords are ignored."""
    out: dict[str, Block] = {}

    def add(b: Block) -> None:
        out[b.address] = b  # duplicate address in one file: last wins (invalid TF anyway)

    for rtype, named in _iter_labeled(parsed.get("resource")):
        for name, body in (named or {}).items():
            add(_make_block(RESOURCE, rtype, name, f"resource.{rtype}.{name}", body, path))
    for dtype, named in _iter_labeled(parsed.get("data")):
        for name, body in (named or {}).items():
            add(_make_block(DATA, f"data.{dtype}", name, f"data.{dtype}.{name}", body, path))
    for name, body in _iter_labeled(parsed.get("module")):
        add(_make_block(MODULE, MODULE, name, f"module.{name}", body, path))
    for name, body in _iter_labeled(parsed.get("variable")):
        add(_make_block(VARIABLE, VARIABLE, name, f"var.{name}", body, path))
    for name, body in _iter_labeled(parsed.get("output")):
        add(_make_block(OUTPUT, OUTPUT, name, f"output.{name}", body, path))
    for key, expr in _iter_labeled(parsed.get("locals")):
        add(_make_block(LOCAL, LOCAL, key, f"local.{key}", expr, path))
    return out


def parse_blocks(text: str, path: str) -> dict[str, Block]:
    """Parse `.tf` text at repo-relative `path` into {address: Block}. Raises on parse errors."""
    return blocks_from_parsed(parse_hcl(text), path)
