from __future__ import annotations

import re


def slugify(full_name: str) -> str:
    """`owner/repo` -> `owner__repo`"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", full_name.replace("/", "__")).strip("_")
