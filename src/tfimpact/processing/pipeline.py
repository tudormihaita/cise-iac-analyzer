"""Data collection pipeline for selected TerraDS IaC Terraform repositories. Flow:

Per commit t:
    staged = snapshot.stage(changes(t)) # host post-commit blocks look after commit t; snapshot still == parent(t)
    items  = diff(staged.before, staged.after) # compare the two versions block by block
    if usable(t): emit queries from (snapshot, miner, items)  <- leakage boundary
        miner.update(items)
    snapshot.apply(staged) # move the snapshot forwards to the state after t, ready to compare with t+1

A percentage of "warmup" commits is established and skipped: they usually represent initial commits where most blocks are added
and the miner has no prior context yet for a fair co-change assessment.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tfimpact.collection.clone import BLOBLESS, bare_clone
from tfimpact.config import Config, config_to_dict
from tfimpact.constants import ADDED, SKIP_START
from tfimpact.processing.diff import diff_blocks
from tfimpact.processing.filters import classify_commit, warmup_count
from tfimpact.processing.gitio import (
    BlobReader,
    first_parent_history,
    ls_remote_head,
    ls_tree_tf,
    prefetch_blobs,
    rev_parse,
)
from tfimpact.processing.hcl_parser import Block, blocks_from_parsed, parse_hcl
from tfimpact.processing.io import write_table
from tfimpact.processing.snapshot import ParseFailure, Snapshot

log = logging.getLogger(__name__)

SUCCESS = "_SUCCESS"


@dataclass
class RepoResult:
    commits: pd.DataFrame
    changes: pd.DataFrame
    parse_errors: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


class ParseCache:
    """Parses each blob once (keyed by oid); blocks are rebuilt per path."""
    def __init__(self, reader: BlobReader):
        self._reader = reader
        self._parsed: dict[str, dict[str, Any] | str] = {}

    def load(self, path: str, oid: str) -> dict[str, Block]:
        if oid not in self._parsed:
            try:
                text = self._reader.read(oid).decode("utf-8", errors="replace")
                self._parsed[oid] = parse_hcl(text)
            except Exception as e:  # lark errors are varied; record and move on
                self._parsed[oid] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        parsed = self._parsed[oid]
        if isinstance(parsed, str):
            raise ParseFailure(path, oid, parsed)
        return blocks_from_parsed(parsed, path)

    @property
    def n_blobs(self) -> int:
        return len(self._parsed)

    @property
    def n_failed(self) -> int:
        return sum(isinstance(v, str) for v in self._parsed.values())


def process_repo(
    repo: Path, slug: str, cfg: Config, blobless: bool, rev: str = "HEAD"
) -> RepoResult:
    m = cfg.mining
    t0 = time.time()
    history = first_parent_history(repo, m.max_commits, rev)
    if not history:
        raise RuntimeError("empty history")
    start = history[0]
    tree = ls_tree_tf(repo, start.sha)

    if blobless:
        needed = set(tree.values())
        needed |= {fc.new_oid for c in history[1:] for fc in c.changes if fc.new_oid}
        prefetch_blobs(repo, sorted(needed), cfg.clone.prefetch_chunk, cfg.clone.timeout_s)
    t_fetch = time.time() - t0

    commit_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    def record_errors(idx: int, sha: str, errors: list[ParseFailure]) -> None:
        for e in errors:
            error_rows.append(
                {
                    "repo": slug,
                    "idx": idx,
                    "sha": sha,
                    "file": e.path,
                    "oid": e.oid,
                    "error": e.error,
                }
            )

    with BlobReader(repo) as reader:
        cache = ParseCache(reader)
        snap, errs = Snapshot.from_tree(tree, cache.load)
        record_errors(0, start.sha, errs)
        commit_rows.append(_commit_row(slug, 0, start, 0, 0, True, SKIP_START))

        for idx, c in enumerate(history[1:], start=1):
            staged = snap.stage(c.changes, cache.load)
            record_errors(idx, c.sha, staged.errors)
            items = diff_blocks(staged.before, staged.after)
            usable, reason = classify_commit(
                is_start=False,
                is_merge=c.is_merge,
                n_items=len(items),
                min_items=m.min_items,
                max_items=m.max_items,
                use_merge_commits=m.use_merge_commits,
            )
            # query emission + miner.update go here (snapshot == parent)
            for it in items:
                change_rows.append(
                    {
                        "repo": slug,
                        "idx": idx,
                        "address": it.address,
                        "kind": it.kind,
                        "type": it.type,
                        "action": it.action,
                        "file": it.file,
                    }
                )
            commit_rows.append(
                _commit_row(slug, idx, c, len(c.changes), len(items), False, reason, usable)
            )
            snap.apply(staged)

    commits = pd.DataFrame(commit_rows)
    n_usable = int(commits.usable.sum())
    n_warm = warmup_count(n_usable, m.warmup_frac)
    usable_idx = commits.index[commits.usable]
    commits["in_warmup"] = False
    commits.loc[usable_idx[:n_warm], "in_warmup"] = True

    changes = pd.DataFrame(
        change_rows, columns=["repo", "idx", "address", "kind", "type", "action", "file"]
    )
    errors = pd.DataFrame(error_rows, columns=["repo", "idx", "sha", "file", "oid", "error"])
    stats = _stats(slug, commits, changes, cache, snap, time.time() - t0, t_fetch)
    return RepoResult(commits, changes, errors, stats)


def _commit_row(
    slug: str,
    idx: int,
    c: Any,
    n_files: int,
    n_items: int,
    is_start: bool,
    reason: str | None,
    usable: bool = False,
) -> dict[str, Any]:
    return {
        "repo": slug,
        "idx": idx,
        "sha": c.sha,
        "date": c.date,
        "author": c.author,
        "is_merge": c.is_merge,
        "n_tf_files": n_files,
        "n_items": n_items,
        "usable": usable and not is_start,
        "skip_reason": reason,
    }


def _stats(
    slug: str,
    commits: pd.DataFrame,
    changes: pd.DataFrame,
    cache: ParseCache,
    snap: Snapshot,
    runtime: float,
    t_fetch: float,
) -> dict[str, Any]:
    usable = commits[commits.usable]
    ch_u = changes[changes.idx.isin(usable.idx)]
    if len(ch_u):
        add_only = ch_u.groupby("idx").action.agg(lambda a: bool((a == ADDED).all()))
        add_only_frac = float(add_only.mean())
        add_item_frac = float((ch_u.action == ADDED).mean())
    else:
        add_only_frac = add_item_frac = float("nan")
    return {
        "repo": slug,
        "n_commits": len(commits),
        "n_merges": int(commits.is_merge.sum()),
        "n_tf_commits": int((commits.n_tf_files > 0).sum()),
        "n_with_items": int((commits.n_items > 0).sum()),
        "n_usable": len(usable),
        "n_usable_merge": int(usable.is_merge.sum()),
        "n_query_commits": int((commits.usable & ~commits.in_warmup).sum()),
        "skip_reasons": commits.skip_reason.value_counts().to_dict(),
        "n_items_usable_median": float(usable.n_items.median()) if len(usable) else float("nan"),
        "add_only_frac": add_only_frac,
        "add_item_frac": add_item_frac,
        "n_blobs": cache.n_blobs,
        "parse_rate": 1 - cache.n_failed / cache.n_blobs if cache.n_blobs else float("nan"),
        "n_blocks_final": len(snap.blocks),
        "n_dirs_final": len(snap.directories),
        "first_date": commits.date.iloc[0],
        "last_date": commits.date.iloc[-1],
        "fetch_s": round(t_fetch, 1),
        "runtime_s": round(runtime, 1),
    }


# --- Orchestration ---

def processed_head(out_dir: Path) -> str | None:
    """HEAD SHA a finished repo was mined at (None if not finished / unknown)."""
    if not (out_dir / SUCCESS).exists() or not (out_dir / "meta.json").exists():
        return None
    return json.loads((out_dir / "meta.json").read_text()).get("head_sha")


def _pin(row: pd.Series) -> str | None:
    v = row.get("head_sha")
    return v if isinstance(v, str) and v else None


def mine_repo(
    row: pd.Series,
    cfg: Config,
    force: bool = False,
    csv: bool | None = None,
    latest: bool = False,
) -> dict:
    """Clone -> process -> write parquet -> _SUCCESS. Returns a summary row (incl. `head_sha`).

    Revision: the pinned `row.head_sha` by default (HEAD if no pin); with `latest`, the remote's
    current HEAD. A finished repo is skipped when it was already mined at that exact SHA.
    """
    out_dir = cfg.paths.processed / row.slug
    done_at = processed_head(out_dir)
    t0 = time.time()
    try:
        target = ls_remote_head(row.clone_url, cfg.clone.timeout_s) if latest else _pin(row)
        if not force and done_at and (target is None or target == done_at):
            return {"repo": row.slug, "status": "skipped_done", "head_sha": done_at}
        with bare_clone(
            row.clone_url, cfg.paths.tmp, row.slug, cfg.clone.mode, cfg.clone.timeout_s
        ) as path:
            sha = rev_parse(path, target or "HEAD")  # fails clearly if a pin was force-pushed away
            res = process_repo(path, row.slug, cfg, blobless=cfg.clone.mode == BLOBLESS, rev=sha)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log.exception("mining %s failed", row.slug)
        return {
            "repo": row.slug,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "runtime_s": round(time.time() - t0, 1),
        }

    write_csv = cfg.output.csv if csv is None else csv
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUCCESS).unlink(missing_ok=True)
    write_table(res.commits, out_dir / "commits", csv=write_csv)
    write_table(res.changes, out_dir / "changes", csv=write_csv)
    write_table(res.parse_errors, out_dir / "parse_errors", csv=write_csv)
    meta = {
        "head_sha": sha,
        "stats": res.stats,
        "config": config_to_dict(cfg),
        "repo": row.to_dict(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / SUCCESS).write_text("")
    return {"status": "ok", "head_sha": sha, **res.stats, "repo_kind": row.get("repo_kind")}


def append_summary(summary_csv: Path, row: dict) -> None:
    flat = {k: (str(v) if isinstance(v, dict) else v) for k, v in row.items()}
    df = pd.DataFrame([flat])
    if summary_csv.exists():
        old = pd.read_csv(summary_csv)
        old = old[old.repo != flat["repo"]]
        df = pd.concat([old, df], ignore_index=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
