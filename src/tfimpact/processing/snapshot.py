"""Incremental repository state snapshot.

A two-phase update keeps the leakage boundary explicit:
    staged = snap.stage(file_changes, loader) # computes post-commit blocks, state untouched
    ... build queries from `snap` (== parent state) + diff(staged.before, staged.after) ...
    snap.apply(staged) # advance to the commit
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from tfimpact.processing.gitio import FileChange
from tfimpact.processing.hcl_parser import Block, module_dir

# loader(path, oid) -> blocks; raises ParseFailure if the file cannot be parsed
Loader = Callable[[str, str], dict[str, Block]]


class ParseFailure(Exception):
    def __init__(self, path: str, oid: str, error: str):
        super().__init__(f"{path}@{oid[:8]}: {error}")
        self.path, self.oid, self.error = path, oid, error


@dataclass
class FileState:
    oid: str
    blocks: dict[str, Block]
    parsed_ok: bool = True


@dataclass
class StagedCommit:
    files: dict[str, FileState | None]  # path -> new state (None = deleted)
    before: dict[str, Block]  # blocks of affected directories at parent
    after: dict[str, Block]  # blocks of affected directories after the commit
    dirs: set[str]
    errors: list[ParseFailure] = field(default_factory=list)


class Snapshot:
    def __init__(self) -> None:
        self.files: dict[str, FileState] = {}
        self._dir_files: dict[str, set[str]] = {}
        self.blocks: dict[str, Block] = {}
        self._dir_addrs: dict[str, set[str]] = {}

    # --- queries ---
    def dir_blocks(self, directory: str) -> dict[str, Block]:
        return {a: self.blocks[a] for a in self._dir_addrs.get(directory, ())}

    @property
    def directories(self) -> list[str]:
        return sorted(self._dir_addrs)

    # --- construction ---
    @classmethod
    def from_tree(
        cls, tree: Mapping[str, str], loader: Loader
    ) -> tuple[Snapshot, list[ParseFailure]]:
        snap = cls()
        changes = [FileChange("A", p, None, oid) for p, oid in sorted(tree.items())]
        staged = snap.stage(changes, loader)
        snap.apply(staged)
        return snap, staged.errors

    # --- two-phase update ---
    def stage(self, changes: Iterable[FileChange], loader: Loader) -> StagedCommit:
        new_files: dict[str, FileState | None] = {}
        errors: list[ParseFailure] = []
        for fc in changes:
            if fc.new_oid is None:
                new_files[fc.path] = None
                continue
            try:
                new_files[fc.path] = FileState(fc.new_oid, loader(fc.path, fc.new_oid))
            except ParseFailure as e:
                errors.append(e)
                prev = self.files.get(fc.path)
                # keep the previous blocks of this file
                new_files[fc.path] = FileState(
                    fc.new_oid, dict(prev.blocks) if prev else {}, parsed_ok=False
                )
        dirs = {module_dir(p) for p in new_files}
        before: dict[str, Block] = {}
        after: dict[str, Block] = {}
        for d in dirs:
            before.update(self.dir_blocks(d))
            after.update(self._merge_dir(d, new_files))
        return StagedCommit(new_files, before, after, dirs, errors)

    def apply(self, staged: StagedCommit) -> None:
        for path, st in staged.files.items():
            d = module_dir(path)
            if st is None:
                self.files.pop(path, None)
                self._dir_files.get(d, set()).discard(path)
            else:
                self.files[path] = st
                self._dir_files.setdefault(d, set()).add(path)
        for d in staged.dirs:
            for a in self._dir_addrs.pop(d, set()):
                self.blocks.pop(a, None)
            new = {a: b for a, b in staged.after.items() if b.directory == d}
            if new:
                self.blocks.update(new)
                self._dir_addrs[d] = set(new)
            if not self._dir_files.get(d):
                self._dir_files.pop(d, None)

    def _merge_dir(
        self, directory: str, overrides: Mapping[str, FileState | None]
    ) -> dict[str, Block]:
        paths = set(self._dir_files.get(directory, set()))
        paths |= {p for p in overrides if module_dir(p) == directory}
        merged: dict[str, Block] = {}
        for p in sorted(paths):  # deterministic: later path wins on duplicate address
            st = overrides[p] if p in overrides else self.files.get(p)
            if st is not None:
                merged.update(st.blocks)
        return merged
