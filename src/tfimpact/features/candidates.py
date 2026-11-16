"""Candidate generation + pre-filter.

Existing-block candidates: every block of the parent snapshot except the seed, pre-filtered to
`max_candidates` by keeping the union of
    rule support > 0 | graph distance <= prefilter_max_dist | same directory | name similarity > 0
If the union is larger than the budget it is trimmed by a fixed priority order (strongest
evidence first); if smaller, it is filled with the highest-degree blocks.

Add-type candidates: `+type:T` for every T with type-rule confidence(seed type item -> +T) > 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tfimpact.constants import ADDED
from tfimpact.features.association_rules import AssociationRuleMiner
from tfimpact.features.similarity import TokenIndex, jaccard, name_tokens
from tfimpact.processing.hcl_parser import Block

NO_DIST = 10**6


@dataclass
class CandidateSet:
    existing: list[str]
    add_types: list[str]
    dist: dict[str, int]  # graph distances from the seed (<= graph_max_dist)
    n_union: int  # size of the pre-filter union before trimming / filling


def generate(
    seed: Block,
    seed_type_item: str,
    blocks: Mapping[str, Block],
    dist: dict[str, int],
    same_dir: set[str],
    miner: AssociationRuleMiner,
    tokens: TokenIndex,
    degree: Mapping[str, int],
    by_degree: Sequence[str],
    max_candidates: int,
    prefilter_max_dist: int,
) -> CandidateSet:
    q = seed.address
    partners = miner.partners(q)
    seed_tok = name_tokens(seed.name)

    union: set[str] = {c for c in partners if c in blocks}
    union |= {c for c, d in dist.items() if d <= prefilter_max_dist}
    union |= same_dir
    union |= tokens.sharing_any(seed_tok)
    union.discard(q)
    n_union = len(union)

    if n_union > max_candidates:

        def priority(c: str) -> tuple:
            b = blocks[c]
            st = partners.get(c)
            return (
                -(st[0] if st else 0),  # 1. co-change support
                dist.get(c, NO_DIST),  # 2. graph distance
                b.file != seed.file,  # 3. same file
                b.directory != seed.directory,  # 4. same directory
                -jaccard(seed_tok, name_tokens(b.name)),  # 5. name similarity
                -degree.get(c, 0),  # 6. degree
                c,  # deterministic tie-break
            )

        existing = sorted(union, key=priority)[:max_candidates]
    else:
        existing = sorted(union)
        if n_union < max_candidates:
            for c in by_degree:
                if len(existing) >= max_candidates:
                    break
                if c != q and c not in union:
                    existing.append(c)

    add_types = sorted(k[1:] for k in miner.type_consequents(seed_type_item) if k.startswith(ADDED))
    return CandidateSet(existing, add_types, dist, n_union)
