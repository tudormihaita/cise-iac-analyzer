"""Export of the curated result files: from the run outputs (`paths.outputs`,
gitignored) to `paths.results/<experiment>/` (committed). Large files (per-query tables, GA run
files) stay in the outputs folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tfimpact.evaluation.figures import FIGURES

EXPORTED_FILES = [
    "summary.md",
    "metrics.csv",
    "stats.csv",
    "breakdowns.csv",
    "per_repo.csv",
    "file_level.csv",
    "learned_weights.csv",
    "grid.csv",
    "folds.csv",
    "run.json",
    "convergence_summary.csv",
    "convergence.png",
    "checks/summary.md",
    "checks/metrics.csv",
    "checks/learned_weights.csv",
]


def export_results(outputs_dir: Path, dest: Path) -> list[Path]:
    """Copy the small, report-relevant result files and all figures to `dest`."""
    copied = []
    for rel in EXPORTED_FILES:
        src = outputs_dir / rel
        if src.exists():
            target = dest / rel.replace("/", "_") if "/" in rel else dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied.append(target)
    for sub in (outputs_dir / FIGURES, outputs_dir / "checks" / FIGURES):
        if sub.is_dir():
            prefix = "checks_" if sub.parent.name == "checks" else ""
            for png in sorted(sub.glob("*.png")):
                target = dest / FIGURES / f"{prefix}{png.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(png, target)
                copied.append(target)
    return copied
