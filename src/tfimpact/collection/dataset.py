from __future__ import annotations

import json

import pandas as pd

from tfimpact.config import Config
from tfimpact.processing.pipeline import processed_head

STAT_COLS = [
    "n_commits",
    "n_usable",
    "n_usable_merge",
    "n_query_commits",
    "add_only_frac",
    "add_item_frac",
    "parse_rate",
    "n_blocks_final",
]


def build_dataset(cfg: Config) -> pd.DataFrame:
    """One row per repo in repos.csv with stats, `included` and `exclude_reason`."""
    s = cfg.selection
    repos = pd.read_csv(cfg.paths.repos_csv)
    rows = []
    for _, r in repos.iterrows():
        out_dir = cfg.paths.processed / r.slug
        head = processed_head(out_dir)
        row = {
            "slug": r.slug,
            "full_name": r.full_name,
            "repo_kind": r.get("repo_kind"),
            "head_sha": head,
        }
        if head is None:
            rows.append({**row, "included": False, "exclude_reason": "not_mined"})
            continue
        stats = json.loads((out_dir / "meta.json").read_text())["stats"]
        row.update({k: stats.get(k) for k in STAT_COLS})
        reasons = []
        if stats["n_usable"] < s.min_usable_commits:
            reasons.append(f"n_usable<{s.min_usable_commits}")
        if (stats.get("add_only_frac") or 0) > s.max_add_only_frac:
            reasons.append(f"add_only_frac>{s.max_add_only_frac}")
        rows.append({**row, "included": not reasons, "exclude_reason": ";".join(reasons) or None})
    df = pd.DataFrame(rows)
    cfg.paths.dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.paths.dataset_csv, index=False)
    return df
