"""Run bookkeeping: reproducible random streams and resumable run files.

Every finished run is saved as JSON together with a hash of the settings it was produced with.
Refitting reuses a saved run only if the hash matches, so long runs can be interrupted and
resumed, and a changed setting always re-runs. Each run draws from its own random stream, so
resumed and uninterrupted executions give identical results.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np


def run_rng(base_seed: int, tag: str, fold: int, seed: int) -> np.random.Generator:
    """Independent random stream per (tag, fold, seed)."""
    return np.random.default_rng([base_seed, zlib.crc32(tag.encode()), fold + 1, seed])


def settings_hash(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def load_run(path: Path, h: str) -> dict | None:
    """The saved run at `path`, or None if missing or produced with other settings."""
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    return rec if rec.get("settings_hash") == h else None


def save_run(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)  # atomic: an interrupted write never leaves a half file
