from __future__ import annotations

from pathlib import Path

import pandas as pd


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if (
            out[col].dtype == object
            and out[col].map(lambda v: isinstance(v, list | set | frozenset | tuple)).any()
        ):
            out[col] = out[col].map(
                lambda v: (
                    ";".join(sorted(map(str, v)))
                    if isinstance(v, list | set | frozenset | tuple)
                    else v
                )
            )
    return out


def write_table(df: pd.DataFrame, path_stem: Path, csv: bool = False) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_stem.with_suffix(".parquet"), index=False)
    if csv:
        _csv_safe(df).to_csv(path_stem.with_suffix(".csv"), index=False)


def read_table(path_stem: Path) -> pd.DataFrame:
    return pd.read_parquet(path_stem.with_suffix(".parquet"))
