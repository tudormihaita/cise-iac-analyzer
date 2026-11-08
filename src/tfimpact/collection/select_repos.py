from __future__ import annotations

import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tfimpact.collection.clone import BLOBLESS, bare_clone
from tfimpact.collection.utils import slugify
from tfimpact.config import Config
from tfimpact.processing.gitio import count_tf_commits, rev_parse, root_commit

log = logging.getLogger(__name__)

SHORTLIST_SQL = """
WITH agg AS (
    SELECT m.RepositoryId AS rid,
           SUM(r.ResourceType = 'Managed' AND r.Provider = :provider) AS provider_managed,
           SUM(r.ResourceType = 'Managed') AS managed,
           COUNT(DISTINCT m.Id) AS n_modules
    FROM Resources r JOIN Modules m ON r.ModuleId = m.Id
    GROUP BY m.RepositoryId
)
SELECT p.FullName AS full_name, p.Name AS name, p.Description AS description,
       p.Topics AS topics, p.CloneUrl AS clone_url, p.StarCount AS stars,
       p.License AS license, p.LatestCommitAt AS latest_commit_at,
       p.Archived AS archived,
       agg.provider_managed, agg.managed, agg.n_modules
FROM agg JOIN Repositories p ON p.Id = agg.rid
WHERE agg.provider_managed >= :min_resources
  AND p.StarCount >= :min_stars
ORDER BY p.StarCount DESC, p.FullName
LIMIT :limit  -- -1 = no limit (SQLite)
"""

_EDU_RE = re.compile(
    r"tutorial|example|course|learn|workshop|training|demo|goat|vulnerab|boilerplate|book|lab\b",
    re.IGNORECASE,
)
_MODULE_NAME_RE = re.compile(r"^terraform-[a-z0-9]+-[a-z0-9_-]+$")


def repo_kind(name: str, description: str | None, topics: str | None) -> str:
    """Labeling logic for distinguishing different types of repositories (used for breakdowns, not filtering)."""
    try:
        topic_list = json.loads(topics) if isinstance(topics, str) and topics else []
    except json.JSONDecodeError:
        topic_list = []
    desc = description if isinstance(description, str) else ""
    text = " ".join([name, desc, " ".join(topic_list)])
    if _EDU_RE.search(text):
        return "educational"
    if _MODULE_NAME_RE.match(name.lower()) or "terraform-module" in topic_list:
        return "module_library"
    return "deployment"


def shortlist(cfg: Config, limit: int | None = None) -> pd.DataFrame:
    """TerraDS candidates ordered by stars; `limit` -1 / shortlist None = all."""
    s = cfg.selection
    with sqlite3.connect(f"file:{cfg.paths.terrads_db}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(
            SHORTLIST_SQL,
            con,
            params={
                "provider": s.provider,
                "min_resources": s.min_resources,
                "min_stars": s.min_stars,
                "limit": limit if limit is not None else (s.shortlist or -1),
            },
        )
    excl_kw = [k.lower() for k in s.exclude_keywords]
    excl = set(s.exclude_repos)

    def excluded(row: pd.Series) -> bool:
        text = f"{row.full_name} {row.description} {row.topics}".lower()
        return row.full_name in excl or any(k in text for k in excl_kw)

    df = df[~df.apply(excluded, axis=1)].copy()
    df["slug"] = df.full_name.map(slugify)
    df["repo_kind"] = [
        repo_kind(n, d, t) for n, d, t in zip(df.name, df.description, df.topics, strict=True)
    ]
    return df.reset_index(drop=True)


def _depth_one(
    row: pd.Series, cfg: Config
) -> tuple[int | None, str | None, str | None, str | None]:
    """(n_tf_commits, head_sha, root_sha, error). Counts at the pinned SHA when `row.pin` is set."""
    try:
        with bare_clone(
            row.clone_url, cfg.paths.tmp, f"depth_{row.slug}", BLOBLESS, cfg.clone.timeout_s
        ) as path:
            pin = row.get("pin")
            sha = rev_parse(path, pin if isinstance(pin, str) and pin else "HEAD")
            n = count_tf_commits(
                path, cfg.mining.max_commits, cfg.mining.use_merge_commits, rev=sha
            )
            return n, sha, root_commit(path, sha), None
    except Exception as e:  # GitError, timeouts, OSError ...
        return None, None, None, f"{type(e).__name__}: {str(e)[:200]}"


def depth_check(df: pd.DataFrame, cfg: Config, workers: int) -> pd.DataFrame:
    """Annotate `n_tf_commits`, `head_sha`, `depth_error` per shortlisted repo (blobless clones)."""
    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(
            tqdm(ex.map(lambda r: _depth_one(r, cfg), rows), total=len(rows), desc="depth check")
        )
    out = df.copy()
    out["n_tf_commits"] = [r[0] for r in results]
    out["head_sha"] = [r[1] for r in results]
    out["root_sha"] = [r[2] for r in results]
    out["depth_error"] = [r[3] for r in results]
    return out


def mark_fork_duplicates(df: pd.DataFrame) -> pd.Series:
    """True for repos whose root commit already belongs to a higher-starred repo.

    Forks/copies share history; splitting them across CV folds would leak training history
    into test repos.
    """
    has_root = df.root_sha.notna()
    return has_root & df.root_sha.duplicated(keep="first")


def load_pins(repos_csv: Path) -> dict[str, str]:
    """{slug: head_sha} from an existing repos.csv (empty if none)."""
    if not repos_csv.exists():
        return {}
    df = pd.read_csv(repos_csv)
    if "head_sha" not in df.columns:
        return {}
    return {s: h for s, h in zip(df.slug, df.head_sha, strict=True) if isinstance(h, str) and h}


SELECTED_COLS = [
    "slug",
    "full_name",
    "clone_url",
    "head_sha",
    "stars",
    "license",
    "archived",
    "repo_kind",
    "n_tf_commits",
    "provider_managed",
    "root_sha",
]


def select(
    cfg: Config, limit: int | None, workers: int, latest: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (shortlist with depth info, selected repos).

    Repos already pinned in repos.csv are evaluated at their pinned SHA unless `latest`.
    """
    sl = shortlist(cfg, limit)
    pins = {} if latest else load_pins(cfg.paths.repos_csv)
    sl["pin"] = sl.slug.map(pins)
    sl = depth_check(sl, cfg, workers)
    sl["fork_duplicate"] = mark_fork_duplicates(sl) if cfg.selection.dedup_forks else False
    ok = sl[(sl.n_tf_commits.fillna(-1) >= cfg.selection.min_tf_commits) & ~sl.fork_duplicate]
    return sl, ok[SELECTED_COLS].reset_index(drop=True)


def write_selection(sl: pd.DataFrame, selected: pd.DataFrame, cfg: Config) -> Path:
    cfg.paths.repos_csv.parent.mkdir(parents=True, exist_ok=True)
    sl.drop(columns=["description"]).to_csv(cfg.paths.shortlist_csv, index=False)
    selected.to_csv(cfg.paths.repos_csv, index=False)
    return cfg.paths.repos_csv
