"""Mined propagation rules over reference edges (`graph_propagation_conf`).

A relational association rule, counted per repo over earlier commits:

    changed(S, a)  ∧  edge(S, C, e)  =>  changed(C)

    a ∈ {+, ~, -}   the action of the changed block S
    e ∈ {in, out}   in:  C references S (C depends on S; a removed S breaks C)
                    out: S references C (S depends on C)

For every changed block S of a commit and every direct neighbor C of S at parent(t), the
cell (a, e) gets one observation, and one "hit" if C was itself changed (~ or -) in that commit.
The feature for a candidate is the confidence of its cell if it is a direct neighbor of the
seed, else 0. Edges are those of `RefGraph` (incl. module plumbing); S's outgoing edges come
from its new body (removed S: its last body), as for the query seed. A neighbor linked in both
directions counts as `in` (dependence on S dominates).

Like `AssociationRuleMiner`, `update` must run after the queries of commit t were built and
before the graph moves to t (leakage boundary).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from tfimpact.constants import MODIFIED, REMOVED
from tfimpact.features.graph import RefGraph
from tfimpact.processing.diff import ChangeItem

IN = "in"
OUT = "out"


def edge_direction(c: str, incoming: set[str], outgoing: set[str]) -> str | None:
    if c in incoming:
        return IN
    if c in outgoing:
        return OUT
    return None


class PropagationMiner:
    def __init__(self) -> None:
        self.n: Counter[tuple[str, str]] = Counter()
        self.n_changed: Counter[tuple[str, str]] = Counter()

    def update(self, items: Iterable[ChangeItem], graph: RefGraph) -> None:
        """Count one commit; `graph` must still describe parent(t)."""
        items = list(items)
        changed = {it.address for it in items if it.action in (MODIFIED, REMOVED)}
        for it in items:
            s = it.new if it.new is not None else it.old
            assert s is not None
            incoming = graph.in_neighbors(it.address)
            outgoing = graph.out_neighbors(s)
            for c in incoming | outgoing:
                if c == it.address:
                    continue
                key = (it.action, edge_direction(c, incoming, outgoing) or OUT)
                self.n[key] += 1
                if c in changed:
                    self.n_changed[key] += 1

    def confidence(self, action: str, direction: str) -> float:
        n = self.n.get((action, direction), 0)
        return self.n_changed.get((action, direction), 0) / n if n else 0.0
