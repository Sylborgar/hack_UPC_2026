from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_cases(path: str | Path) -> pd.DataFrame:
    path = resolve_data_path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {path}")


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            return pd.read_csv(path.with_suffix(".csv"))
    return pd.read_csv(path)


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except ImportError:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=index)
            return csv_path
    df.to_csv(path, index=index)
    return path


def load_feature_sets(path: str | Path) -> dict[str, list[str]]:
    with resolve_data_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    fallback = Path("dataset/final") / path.name
    if fallback.exists():
        return fallback
    return path
