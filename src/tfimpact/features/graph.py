"""Reference graph over the blocks of a snapshot.

Undirected adjacency of a block a:
  * direct:   blocks a references (that exist) + blocks referencing a
  * plumbing: module block m with a local source dir D
                m  <-> D::var.<arg>     for every argument m sets
                b  <-> D::output.<o>    for every block b reading module.m.<o>

The graph is maintained incrementally (`update` after `Snapshot.apply`), because rebuilding
it per commit is too slow for repos with ~15k blocks. Plumbing edges are resolved at query
time from the current module index, so a module whose `source` changes is always current.

Queries can override the seed's body (`seed=` argument): its *outgoing* edges then come from
the seed's new body (what the developer just wrote).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Mapping

from tfimpact.constants import ADDR_SEP
from tfimpact.processing.hcl_parser import Block
from tfimpact.processing.references import make_address


def _split(addr: str) -> tuple[str, str]:
    d, _, local = addr.partition(ADDR_SEP)
    return d, local


class RefGraph:
    def __init__(self) -> None:
        self.blocks: Mapping[str, Block] = {}
        self._inv: dict[str, set[str]] = defaultdict(set)  # target -> sources (raw refs)
        self._inv_modout: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._dir_modules: dict[str, set[str]] = defaultdict(set)  # source dir -> modules

    # --- maintenance ---
    def rebuild(self, blocks: Mapping[str, Block]) -> None:
        self.__init__()
        self.blocks = blocks
        for b in blocks.values():
            self._index(b)

    def update(
        self, before: Mapping[str, Block], after: Mapping[str, Block], blocks: Mapping[str, Block]
    ) -> None:
        """Apply a staged commit (`before`/`after` of the affected dirs); `blocks` = new state."""
        for addr, old in before.items():
            new = after.get(addr)
            if new is None or new.body_hash != old.body_hash or new.file != old.file:
                self._unindex(old)
        for addr, new in after.items():
            old = before.get(addr)
            if old is None or new.body_hash != old.body_hash or new.file != old.file:
                self._index(new)
        self.blocks = blocks

    def _index(self, b: Block) -> None:
        for t in b.refs:
            self._inv[t].add(b.address)
        for mo in b.mod_out_refs:
            self._inv_modout[mo].add(b.address)
        if b.module_source_dir is not None:
            self._dir_modules[b.module_source_dir].add(b.address)

    def _unindex(self, b: Block) -> None:
        for t in b.refs:
            self._inv[t].discard(b.address)
        for mo in b.mod_out_refs:
            self._inv_modout[mo].discard(b.address)
        if b.module_source_dir is not None:
            self._dir_modules[b.module_source_dir].discard(b.address)

    # --- queries ---
    def out_neighbors(self, b: Block) -> set[str]:
        """Edges defined by b's own body (direct refs + plumbing), existing targets only."""
        bl = self.blocks
        out = {t for t in b.refs if t in bl}
        if b.module_source_dir is not None:
            for arg in b.module_args:
                t = make_address(b.module_source_dir, f"var.{arg}")
                if t in bl:
                    out.add(t)
        for m, o in b.mod_out_refs:
            mb = bl.get(m)
            if mb is not None and mb.module_source_dir is not None:
                t = make_address(mb.module_source_dir, f"output.{o}")
                if t in bl:
                    out.add(t)
        out.discard(b.address)
        return out

    def in_neighbors(self, addr: str) -> set[str]:
        """Blocks whose bodies point at `addr` (direct refs + plumbing)."""
        bl = self.blocks
        res = {s for s in self._inv.get(addr, ()) if s in bl}
        d, local = _split(addr)
        mods = self._dir_modules.get(d)
        if mods:
            if local.startswith("var."):
                arg = local[4:]
                res |= {m for m in mods if m in bl and arg in bl[m].module_args}
            elif local.startswith("output."):
                o = local[7:]
                for m in mods:
                    res |= {s for s in self._inv_modout.get((m, o), ()) if s in bl}
        res.discard(addr)
        return res

    def neighbors(self, addr: str, seed: Block | None = None) -> set[str]:
        """Undirected neighbors. If `seed` has this address, its body overrides the stored one."""
        if seed is not None and seed.address == addr:
            own = self.out_neighbors(seed)
        else:
            b = self.blocks.get(addr)
            own = self.out_neighbors(b) if b is not None else set()
        return own | self.in_neighbors(addr)

    def distances(self, seed: Block, max_dist: int) -> dict[str, int]:
        """Undirected shortest-path distance from the seed (seed body overrides), <= max_dist."""
        dist = {seed.address: 0}
        frontier = deque([seed.address])
        while frontier:
            a = frontier.popleft()
            d = dist[a]
            if d == max_dist:
                continue
            for n in self.neighbors(a, seed):
                if n not in dist:
                    dist[n] = d + 1
                    frontier.append(n)
        del dist[seed.address]
        return dist

    def degrees(self) -> dict[str, int]:
        return {a: len(self.neighbors(a)) for a in self.blocks}


def adamic_adar_weights(nodes: set[str], degree: Mapping[str, int]) -> dict[str, float]:
    """Adamic–Adar weight 1 / ln(deg z) per node: a rare shared neighbor counts more than a hub.

    The degree is floored at 2 (a shared neighbor touches both blocks, so its true degree is
    >= 2; the floor only matters for the normalizer and avoids ln(1) = 0).
    """
    return {z: 1.0 / math.log(max(degree.get(z, 0), 2)) for z in nodes}
