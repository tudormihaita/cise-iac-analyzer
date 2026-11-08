from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler

from tfimpact.config import Config, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
_state: dict[str, Config] = {}
console = Console()
_log_console: Console | None = None

# commands whose output is kept as evidence in data/logs/<command>_<timestamp>.log
LOGGED_COMMANDS = {"select-repos", "mine", "build-dataset"}


def cfg() -> Config:
    return _state["cfg"]


def out(*objs: Any) -> None:
    """Print to the terminal and, for logged commands, to the run's log file."""
    console.print(*objs)
    if _log_console is not None:
        _log_console.print(*objs)


def _code_version() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _start_run_log(command: str, logs_dir: Path) -> Path:
    global _log_console
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{command}_{datetime.now():%Y%m%d-%H%M%S}.log"
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    _log_console = Console(file=path.open("a", buffering=1), no_color=True, width=200)
    _log_console.print(f"# command: tfimpact {' '.join(sys.argv[1:])}")
    _log_console.print(f"# started: {datetime.now().isoformat(timespec='seconds')}")
    _log_console.print(f"# code version: {_code_version()}")
    return path


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(None, "--config", "-c", help="YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    _state["cfg"] = load_config(config)
    if ctx.invoked_subcommand in LOGGED_COMMANDS:
        path = _start_run_log(ctx.invoked_subcommand, _state["cfg"].paths.logs)
        console.print(f"[dim]log: {path}[/]")


def _todo(name: str) -> None:
    raise typer.Exit(code=_not_implemented(name))


def _not_implemented(name: str) -> int:
    log = logging.getLogger(__name__)
    log.error("`%s` is not implemented", name)
    return 2


@app.command("select-repos")
def select_repos(
    terrads: Path = typer.Option(None, help="TerraDS sqlite (default from config)"),
    limit: int = typer.Option(None, help="Shortlist size (default selection.shortlist)"),
    all_repos: bool = typer.Option(
        False, "--all", help="Ignore the shortlist: depth-check every TerraDS candidate"
    ),
    workers: int = typer.Option(None, help="Parallel depth-check clones"),
    latest: bool = typer.Option(
        False, "--latest", help="Ignore pins in repos.csv; evaluate current HEADs"
    ),
) -> None:
    """Select candidate repositories from TerraDS (+ blobless depth check)."""
    from tfimpact.collection.select_repos import select, write_selection

    c = cfg()
    if terrads:
        c.paths.terrads_db = terrads
    sl, selected = select(
        c, -1 if all_repos else limit, workers or c.selection.workers, latest=latest
    )
    out = write_selection(sl, selected, c)
    out(
        f"shortlist={len(sl)}  depth_errors={sl.depth_error.notna().sum()}  "
        f"fork_duplicates={int(sl.fork_duplicate.sum())}  "
        f"passing(>= {c.selection.min_tf_commits} tf commits)="
        f"{(sl.n_tf_commits.fillna(-1) >= c.selection.min_tf_commits).sum()}  "
        f"selected={len(selected)} (archived: {int(selected.archived.sum())}) -> {out}"
    )
    out("upper bound: usable-commit + add-only filters are applied after `mine`")
    out(selected.repo_kind.value_counts().to_string())


@app.command()
def mine(
    repos: Path = typer.Option(None, help="repos.csv (default from config)"),
    only: str = typer.Option(None, help="Mine a single slug from repos.csv"),
    url: str = typer.Option(None, help="Ad-hoc: mine `owner/name` (not in repos.csv)"),
    force: bool = typer.Option(False, help="Re-mine even if _SUCCESS exists"),
    csv: bool = typer.Option(None, "--csv/--no-csv", help="Override output.csv"),
    latest: bool = typer.Option(
        False,
        "--latest",
        help="Mine each repo at its current remote HEAD (re-mines only repos with new commits) "
        "and update the pins in repos.csv. Default: use the pinned head_sha when present.",
    ),
) -> None:
    """Mine repositories into data/processed/<slug>/ (resumable, pinned to head_sha)."""
    import pandas as pd

    from tfimpact.collection.utils import slugify
    from tfimpact.processing.pipeline import append_summary, mine_repo

    c = cfg()
    if url:
        df = pd.DataFrame(
            [{"slug": slugify(url), "full_name": url, "clone_url": f"https://github.com/{url}.git"}]
        )
    else:
        df = pd.read_csv(repos or c.paths.repos_csv)
        if only:
            df = df[df.slug == only]
            if df.empty:
                raise typer.BadParameter(f"slug {only!r} not in repos csv")
    summary_csv = c.paths.processed / "summary.csv"
    new_pins: dict[str, str] = {}
    for _, row in df.iterrows():
        out(f"[bold]mining[/] {row.slug}")
        res = mine_repo(row, c, force=force, csv=csv, latest=latest)
        if res.get("head_sha"):
            new_pins[row.slug] = res["head_sha"]
        if res.get("status") != "skipped_done":
            append_summary(summary_csv, res)
        keys = [
            "status",
            "n_commits",
            "n_usable",
            "n_usable_merge",
            "parse_rate",
            "add_only_frac",
            "n_blocks_final",
            "runtime_s",
            "error",
        ]
        out({k: res[k] for k in keys if k in res})
    if not url and new_pins:  # record what was actually mined -> exact rerun later
        full = pd.read_csv(repos or c.paths.repos_csv)
        if "head_sha" not in full.columns:
            full["head_sha"] = None
        full["head_sha"] = full.slug.map(new_pins).fillna(full.head_sha)
        full.to_csv(repos or c.paths.repos_csv, index=False)
    if not url and not only and repos is None:
        _build_dataset()


@app.command("build-dataset")
def build_dataset_cmd() -> None:
    """Pick the mined repos that form the dataset (min usable commits, max add-only fraction).
    Runs automatically at the end of `mine`; re-run alone after changing a threshold.
    """
    _build_dataset()


def _build_dataset() -> None:
    from tfimpact.collection.dataset import build_dataset

    df = build_dataset(cfg())
    inc = df[df.included]
    out(df.drop(columns=["full_name", "head_sha"]).round(3).to_string(index=False))
    out(
        f"[bold]dataset: {len(inc)}/{len(df)} repos[/]  usable={int(inc.n_usable.sum())}  "
        f"query_commits={int(inc.n_query_commits.sum())} -> {cfg().paths.dataset_csv}"
    )


@app.command()
def clean() -> None:
    """Remove leftover clones in data/tmp."""
    from tfimpact.collection.clone import clean_tmp

    n = clean_tmp(cfg().paths.tmp)
    out(f"removed {n} item(s) from {cfg().paths.tmp}")


@app.command("export-csv")
def export_csv(
    only: str = typer.Option(None, help="Single slug"),
    include_features: bool = typer.Option(False, help="Also export features (large)"),
) -> None:
    """Regenerate CSV copies from existing parquet tables."""
    from tfimpact.processing.io import read_table, write_table

    root = cfg().paths.processed
    dirs = [root / only] if only else sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        for pq in sorted(d.glob("*.parquet")):
            if pq.stem == "features" and not include_features:
                continue
            write_table(read_table(pq.with_suffix("")), pq.with_suffix(""), csv=True)
        out(f"exported {d.name}")


@app.command()
def train() -> None:
    """Train a scorer (GA by default)."""
    _todo("train")


@app.command()
def evaluate() -> None:
    """Run evaluation methods."""
    _todo("evaluate")


@app.command()
def experiments() -> None:
    """Run ablation / learning-curve experiments."""
    _todo("experiments")


@app.command()
def report() -> None:
    """Build results tables, plots and reports."""
    _todo("report")


@app.command()
def recommend() -> None:
    """Recommend co-changes for a block in a local repository."""
    _todo("recommend")
