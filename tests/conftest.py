from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

Commit = Mapping[str, str | None]  # relative path -> content (None = delete)


class GitRepo:
    """Minimal helper around a non-bare repo on branch `main`."""

    def __init__(self, path: Path):
        self.path = path
        self.shas: list[str] = []
        self._git("-c", "init.defaultBranch=main", "init", "-q", str(path), cwd=None)
        self._git("config", "user.name", "Test")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "commit.gpgsign", "false")

    def _git(self, *args: str, cwd: Path | None | str = "repo") -> str:
        cmd = ["git"]
        if cwd == "repo":
            cmd += ["-C", str(self.path)]
        proc = subprocess.run(cmd + list(args), capture_output=True, text=True, check=True)
        return proc.stdout.strip()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def commit(self, files: Commit, message: str = "c") -> str:
        for rel, content in files.items():
            p = self.path / rel
            if content is None:
                self._git("rm", "-q", "-f", rel)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            self._git("add", rel)
        self._git("commit", "-q", "--allow-empty", "-m", message)
        sha = self.head()
        self.shas.append(sha)
        return sha

    def merge_side_branch(self, files: Commit, branch: str = "side") -> tuple[str, str]:
        """Branch off HEAD, commit `files` there, `--no-ff` merge into main.

        Returns (side_commit_sha, merge_sha). Appends only the merge to `shas`
        (the side commit is not on the first-parent line).
        """
        base = self._git("rev-parse", "--abbrev-ref", "HEAD")
        self._git("checkout", "-q", "-b", branch)
        side_shas = self.shas
        self.shas = []
        side = self.commit(files, f"side {branch}")
        self.shas = side_shas
        self._git("checkout", "-q", base)
        self._git("merge", "-q", "--no-ff", "-m", f"merge {branch}", branch)
        merge = self.head()
        self.shas.append(merge)
        return side, merge


@pytest.fixture
def make_git_repo(tmp_path: Path) -> Callable[..., GitRepo]:
    """Factory: `make_git_repo([{path: content|None}, ...])` -> GitRepo (.path, .shas, ...)."""
    counter = {"n": 0}

    def factory(commits: list[Commit] | None = None) -> GitRepo:
        counter["n"] += 1
        repo = GitRepo(tmp_path / f"repo{counter['n']}")
        for i, files in enumerate(commits or []):
            repo.commit(files, f"commit {i}")
        return repo

    return factory


@pytest.fixture
def fixture_text() -> Callable[[str], str]:
    return lambda name: (FIXTURES / name).read_text()
