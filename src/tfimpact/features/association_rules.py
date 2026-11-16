"""Incremental pairwise association rules over change transactions.

ROSE (Zimmermann et al., ICSE 2004) mines Apriori rules with a *constrained antecedent*
(the current situation) and *single consequents*, incrementally as transactions arrive. With a
single-entity antecedent (the seed) this reduces to exact pair counting, done here:

    support(q => c)    = N(q, c)          transactions changing both q and c
    confidence(q => c) = N(q, c) / N(q)

Three levels are kept:
  * instance level - block addresses (`.::resource.aws_s3_bucket.logs`)
  * type level     - type items `{action}{type}` (`+aws_dynamodb_table`). As consequent, the
    extra key `*{type}` means "a block of that type was changed (~ or -)", matching the label
    of existing candidates.
  * file level     - `.tf` file paths (`iam.tf => lambda.tf`), GA-IaCRec's granularity. For the
    seed's own file the rule is "another block of the same file changed too".

The miner is purely cumulative: `update(t, ...)` must be called only after all queries of
commit t were built (leakage boundary).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable

from tfimpact.constants import MODIFIED, REMOVED
from tfimpact.processing.diff import ChangeItem

CHANGED = "*"  # consequent-only type key prefix: ~ or -


def changed_key(btype: str) -> str:
    return f"{CHANGED}{btype}"


class AssociationRuleMiner:
    def __init__(self, recency_tau: float) -> None:
        self.tau = recency_tau
        self.n_transactions = 0
        # instance level
        self.n_item: Counter[str] = Counter()
        # pair stats: a -> b -> [count, decayed_sum, last_idx]; symmetric
        self._pair: dict[str, dict[str, list[float]]] = defaultdict(dict)
        self._change_idx: dict[str, list[int]] = defaultdict(list)  # sorted commit idxs
        # type level
        self.n_titem: Counter[str] = Counter()
        self._tpair: dict[str, Counter[str]] = defaultdict(Counter)
        # file level
        self.n_file: Counter[str] = Counter()  # transactions touching the file
        self.n_file_multi: Counter[str] = Counter()  # ... with >= 2 changed blocks in it
        self._fpair: dict[str, Counter[str]] = defaultdict(Counter)

    # --- update ---
    def update(self, idx: int, items: Iterable[ChangeItem]) -> None:
        items = list(items)
        if not items:
            return
        self.n_transactions += 1
        addrs = sorted({it.address for it in items})
        for a in addrs:
            self.n_item[a] += 1
            self._change_idx[a].append(idx)
        for i, a in enumerate(addrs):
            for b in addrs[i + 1 :]:
                self._bump_pair(a, b, idx)
                self._bump_pair(b, a, idx)
        self._update_types(items)
        self._update_files(items)

    def _update_files(self, items: list[ChangeItem]) -> None:
        per_file: Counter[str] = Counter(it.file for it in items)
        files = sorted(per_file)
        for f in files:
            self.n_file[f] += 1
            if per_file[f] >= 2:
                self.n_file_multi[f] += 1
        for i, f in enumerate(files):
            for g in files[i + 1 :]:
                self._fpair[f][g] += 1
                self._fpair[g][f] += 1

    def _bump_pair(self, a: str, b: str, idx: int) -> None:
        st = self._pair[a].get(b)
        if st is None:
            self._pair[a][b] = [1, 1.0, idx]
        else:
            st[0] += 1
            st[1] = st[1] * math.exp(-(idx - st[2]) / self.tau) + 1.0
            st[2] = idx

    def _update_types(self, items: list[ChangeItem]) -> None:
        cnt: Counter[str] = Counter(it.type_item for it in items)
        changed: Counter[str] = Counter()
        for it in items:
            if it.action in (MODIFIED, REMOVED):
                changed[changed_key(it.type)] += 1
        consequents = {**cnt, **changed}
        for a in cnt:
            self.n_titem[a] += 1
            a_type = a[1:]
            for b, n_b in consequents.items():
                # the antecedent occurrence itself does not count as its own consequent
                own = int(a == b or (b == changed_key(a_type) and a[0] in (MODIFIED, REMOVED)))
                if n_b - own > 0:
                    self._tpair[a][b] += 1

    # --- instance queries ---
    def partners(self, q: str) -> dict[str, list[float]]:
        """All c with N(q, c) > 0 -> [count, decayed_sum, last_idx]."""
        return self._pair.get(q, {})

    def support(self, q: str, c: str) -> int:
        st = self._pair.get(q, {}).get(c)
        return int(st[0]) if st else 0

    def confidence(self, q: str, c: str) -> float:
        n_q = self.n_item.get(q, 0)
        return self.support(q, c) / n_q if n_q else 0.0

    def recency(self, q: str, c: str, now: int) -> float:
        """sum over co-change commits i of exp(-(now - i) / tau)."""
        st = self._pair.get(q, {}).get(c)
        if not st:
            return 0.0
        return st[1] * math.exp(-(now - st[2]) / self.tau)

    def churn(self, c: str, now: int, window: int) -> int:
        """Number of commits in (now - window, now) that changed c."""
        idxs = self._change_idx.get(c)
        if not idxs:
            return 0
        return len(idxs) - bisect_left(idxs, now - window + 1)

    # --- type queries ---
    def type_confidence(self, antecedent: str, consequent: str) -> float:
        n_a = self.n_titem.get(antecedent, 0)
        return self._tpair.get(antecedent, {}).get(consequent, 0) / n_a if n_a else 0.0

    def type_consequents(self, antecedent: str) -> dict[str, int]:
        return dict(self._tpair.get(antecedent, {}))

    # --- file queries ---
    def file_confidence(self, seed_file: str, cand_file: str) -> float:
        """conf(seed_file => cand_file); same file: share of its transactions with >= 2 blocks."""
        n = self.n_file.get(seed_file, 0)
        if not n:
            return 0.0
        if seed_file == cand_file:
            return self.n_file_multi.get(seed_file, 0) / n
        return self._fpair.get(seed_file, {}).get(cand_file, 0) / n
