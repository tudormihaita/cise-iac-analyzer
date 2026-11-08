from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tfimpact.processing.gitio import run_git

log = logging.getLogger(__name__)

BLOBLESS = "blobless_prefetch"
FULL = "full"


@contextmanager
def bare_clone(url: str, tmp_dir: Path, slug: str, mode: str, timeout: float) -> Iterator[Path]:
    """Clone `url` into `tmp_dir/<slug>.git`; the directory is removed on exit."""
    if mode not in (BLOBLESS, FULL):
        raise ValueError(f"unknown clone mode {mode!r}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{slug}.git"
    shutil.rmtree(path, ignore_errors=True)
    try:
        args = ["clone", "--bare", "-q"]
        if mode == BLOBLESS:
            args.append("--filter=blob:none")
        run_git(None, *args, url, str(path), timeout=timeout)
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def clean_tmp(tmp_dir: Path) -> int:
    """Remove every leftover clone in `tmp_dir`. Returns the number removed."""
    if not tmp_dir.exists():
        return 0
    n = 0
    for child in tmp_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        n += 1
    return n
