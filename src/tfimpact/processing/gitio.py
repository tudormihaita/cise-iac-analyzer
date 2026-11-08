from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tfimpact.constants import GIT_REGULAR_MODES, NULL_OID, TF_SUFFIX

log = logging.getLogger(__name__)

# never block on credential prompts (deleted/private repos) or pagers.
GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "GCM_INTERACTIVE": "never"}


class GitError(RuntimeError):
    pass


def run_git(
    repo: Path | None,
    *args: str,
    input_text: str | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> str:
    cmd = ["git"]
    if repo is not None:
        cmd += ["-C", str(repo)]
    cmd += list(args)
    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=GIT_ENV,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise GitError(f"{' '.join(cmd[:6])}... failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


@dataclass(frozen=True)
class FileChange:
    """One `.tf` path touched by a commit, relative to its first parent."""
    status: str  # A / M / D / T
    path: str
    old_oid: str | None
    new_oid: str | None


@dataclass
class CommitInfo:
    sha: str
    parents: tuple[str, ...]
    date: str
    author: str
    changes: list[FileChange] = field(default_factory=list)

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1


def is_tf_path(path: str) -> bool:
    return path.endswith(TF_SUFFIX)


def _parse_raw_line(line: str) -> FileChange | None:
    # git raw diff record: :<old mode> <new mode> <old blob> <new blob> <status>\t<path>
    # example: ":100644 100644 <old> <new> M\tpath"
    meta, _, path = line[1:].partition("\t")
    parts = meta.split()
    if len(parts) < 5 or not is_tf_path(path):
        return None
    old_mode, new_mode, old_oid, new_oid, status = parts[:5]
    status = status[0]
    old_ok = old_oid != NULL_OID and old_mode in GIT_REGULAR_MODES
    new_ok = new_oid != NULL_OID and new_mode in GIT_REGULAR_MODES
    if not old_ok and not new_ok:
        return None  # symlinks / submodules
    return FileChange(
        status=status,
        path=path,
        old_oid=old_oid if old_ok else None,
        new_oid=new_oid if new_ok else None,
    )


_REC = "\x01"
_FS = "\x00"


def first_parent_history(repo: Path, max_commits: int, rev: str = "HEAD") -> list[CommitInfo]:
    """Last `max_commits` first-parent commits, oldest first, with `.tf` changes vs. first parent.

    One `git log` call; merges are diffed against their first parent only
    (`--diff-merges=first-parent`), renames are disabled (rename is considered as remove + add).
    """
    out = run_git(
        repo,
        "log",
        "--first-parent",
        "--diff-merges=first-parent",
        "--raw",
        "--no-abbrev",
        "--no-renames",
        f"--max-count={max_commits}",
        "--format=%x01%H%x00%P%x00%aI%x00%an",
        rev,
    )
    commits: list[CommitInfo] = []
    for chunk in out.split(_REC):
        if not chunk.strip():
            continue
        header, _, body = chunk.partition("\n")
        sha, parents, date, author = header.split(_FS)
        ci = CommitInfo(sha=sha, parents=tuple(parents.split()), date=date, author=author)
        for line in body.splitlines():
            if line.startswith(":"):
                fc = _parse_raw_line(line)
                if fc is not None:
                    ci.changes.append(fc)
        commits.append(ci)
    commits.reverse()
    return commits


def ls_tree_tf(repo: Path, rev: str) -> dict[str, str]:
    """{path: blob oid} for every regular `*.tf` file in the tree of `rev`."""
    out = run_git(repo, "ls-tree", "-r", "--full-tree", rev)
    files: dict[str, str] = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        mode, otype, oid = meta.split()
        if otype == "blob" and mode in GIT_REGULAR_MODES and is_tf_path(path):
            files[path] = oid
    return files


def count_tf_commits(repo: Path, max_commits: int, include_merges: bool, rev: str = "HEAD") -> int:
    """Depth-check proxy: first-parent commits (in the window) that touch `*.tf`."""
    commits = first_parent_history(repo, max_commits, rev)
    return sum(1 for c in commits[1:] if c.changes and (include_merges or not c.is_merge))


def rev_parse(repo: Path, rev: str = "HEAD") -> str:
    """Full SHA of `rev`; raises GitError if the commit is not in the clone."""
    return run_git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()


def root_commit(repo: Path, rev: str = "HEAD") -> str:
    """Oldest parentless commit reachable from `rev` (identifies forks / copies)."""
    roots = run_git(repo, "rev-list", "--max-parents=0", rev).split()
    return sorted(roots)[0]


def ls_remote_head(url: str, timeout: float) -> str:
    """Current HEAD SHA of a remote without cloning."""
    out = run_git(None, "ls-remote", url, "HEAD", timeout=timeout).split()
    if not out:
        raise GitError(f"no HEAD for {url}")
    return out[0]


def prefetch_blobs(repo: Path, oids: Sequence[str], chunk: int, timeout: float) -> None:
    """Fetch the given blobs into a blobless clone in a few batched `git fetch` calls."""
    oids = list(dict.fromkeys(oids))
    for i in range(0, len(oids), chunk):
        batch = oids[i : i + chunk]
        run_git(
            repo,
            "-c",
            "fetch.negotiationAlgorithm=noop",
            "fetch",
            "-q",
            "--no-tags",
            "--no-write-fetch-head",
            "--recurse-submodules=no",
            "--filter=blob:none",
            "--stdin",
            "origin",
            input_text="\n".join(batch) + "\n",
            timeout=timeout,
        )


class BlobReader:
    """Persistent `git cat-file --batch` process for fast blob reads."""
    def __init__(self, repo: Path):
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=GIT_ENV,
        )

    def read(self, oid: str) -> bytes:
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(f"{oid}\n".encode())
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().decode().split()
        if len(header) < 3 or header[1] == "missing":
            raise GitError(f"blob {oid} missing")
        size = int(header[2])
        data = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # trailing newline
        return data

    def close(self) -> None:
        if self._proc.poll() is None:
            assert self._proc.stdin
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self) -> BlobReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
