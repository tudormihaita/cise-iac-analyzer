from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tfimpact.constants import ADDED, MODIFIED, REMOVED
from tfimpact.processing.hcl_parser import Block


@dataclass(frozen=True)
class ChangeItem:
    address: str
    kind: str
    type: str
    name: str
    action: str  # + / - / ~
    file: str
    old: Block | None
    new: Block | None

    @property
    def type_item(self) -> str:
        return f"{self.action}{self.type}"


def diff_blocks(before: Mapping[str, Block], after: Mapping[str, Block]) -> list[ChangeItem]:
    """Compare by address; `~` only if the body differs (formatting is invisible)."""
    items: list[ChangeItem] = []
    for addr in sorted(before.keys() | after.keys()):
        b, a = before.get(addr), after.get(addr)
        if b is None and a is not None:
            action, ref = ADDED, a
        elif a is None and b is not None:
            action, ref = REMOVED, b
        elif a is not None and b is not None and a.body_hash != b.body_hash:
            action, ref = MODIFIED, a
        else:
            continue
        items.append(ChangeItem(addr, ref.kind, ref.type, ref.name, action, ref.file, b, a))
    return items
