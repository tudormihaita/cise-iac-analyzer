"""Text signals: name similarity and TF-IDF content similarity.

Content similarity follows the leakage rule: the seed's *new* body is compared
with the candidates' *parent(t)* bodies, and IDF comes from the parent(t) snapshot of the same
repo. `ContentIndex` keeps the document frequencies of the current snapshot and is advanced
only after the queries of a commit were built.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import lru_cache

from tfimpact.processing.hcl_parser import Block

_SPLIT_RE = re.compile(r"[_\-.\s/]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_BODY_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=200_000)
def name_tokens(name: str) -> frozenset[str]:
    """Split on `_`, `-`, `.`, `/`, whitespace and camelCase; lower-cased."""
    parts: list[str] = []
    for chunk in _SPLIT_RE.split(name):
        parts.extend(_CAMEL_RE.split(chunk))
    return frozenset(p.lower() for p in parts if p)


def type_tokens(btype: str) -> frozenset[str]:
    """Tokens of a block type without its provider prefix (`aws_iam_role` -> {iam, role})."""
    t = btype.removeprefix("data.")
    _, _, rest = t.partition("_")
    return name_tokens(rest or t)


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


class TokenIndex:
    """token -> addresses whose block name contains it; kept in sync with the snapshot."""

    def __init__(self) -> None:
        self._idx: dict[str, set[str]] = defaultdict(set)

    def rebuild(self, blocks: Mapping[str, Block]) -> None:
        self._idx = defaultdict(set)
        for b in blocks.values():
            for t in name_tokens(b.name):
                self._idx[t].add(b.address)

    def update(self, before: Mapping[str, Block], after: Mapping[str, Block]) -> None:
        for addr, b in before.items():
            if addr not in after:
                for t in name_tokens(b.name):
                    self._idx[t].discard(addr)
        for addr, b in after.items():
            if addr not in before:
                for t in name_tokens(b.name):
                    self._idx[t].add(addr)

    def sharing_any(self, tokens: frozenset[str]) -> set[str]:
        out: set[str] = set()
        for t in tokens:
            out |= self._idx.get(t, set())
        return out


# --- content similarity ---
def body_tokens(body_norm: str) -> Counter[str]:
    """Lower-case alphanumeric runs of the canonical body: attribute keys, values, references."""
    return Counter(_BODY_TOKEN_RE.findall(body_norm.lower()))


TfIdfVector = tuple[dict[str, float], float]  # (token -> weight, L2 norm)


def cosine(a: TfIdfVector, b: TfIdfVector) -> float:
    (wa, na), (wb, nb) = a, b
    if not na or not nb:
        return 0.0
    if len(wa) > len(wb):
        wa, wb = wb, wa
    return sum(w * wb[t] for t, w in wa.items() if t in wb) / (na * nb)


class ContentIndex:
    """Document frequencies of body tokens over the blocks of the current snapshot.

    TF-IDF (standard scikit-learn weighting): tf = 1 + ln(count), idf = ln((1 + N) / (1 + df)) + 1,
    N = number of blocks in the snapshot. Tokens are cached per body hash.
    """

    def __init__(self) -> None:
        self.df: Counter[str] = Counter()
        self.n_docs = 0
        self._tokens: dict[str, Counter[str]] = {}

    def tokens(self, b: Block) -> Counter[str]:
        tok = self._tokens.get(b.body_hash)
        if tok is None:
            tok = self._tokens[b.body_hash] = body_tokens(b.body_norm)
        return tok

    def rebuild(self, blocks: Mapping[str, Block]) -> None:
        self.df = Counter()
        self.n_docs = 0
        for b in blocks.values():
            self._add(b)

    def update(self, before: Mapping[str, Block], after: Mapping[str, Block]) -> None:
        """Apply a staged commit (`before`/`after` of the affected dirs), like `RefGraph.update`."""
        for addr, old in before.items():
            new = after.get(addr)
            if new is None or new.body_hash != old.body_hash:
                self._remove(old)
        for addr, new in after.items():
            old = before.get(addr)
            if old is None or new.body_hash != old.body_hash:
                self._add(new)

    def _add(self, b: Block) -> None:
        self.df.update(self.tokens(b).keys())
        self.n_docs += 1

    def _remove(self, b: Block) -> None:
        self.df.subtract(self.tokens(b).keys())
        self.n_docs -= 1

    def idf(self, token: str) -> float:
        return math.log((1 + self.n_docs) / (1 + max(self.df.get(token, 0), 0))) + 1.0

    def vector(self, b: Block) -> TfIdfVector:
        w = {t: (1.0 + math.log(c)) * self.idf(t) for t, c in self.tokens(b).items()}
        return w, math.sqrt(sum(x * x for x in w.values()))
