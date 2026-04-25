"""I/O helpers for loading data and writing pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_CREATIVE_COLUMNS = {
    "creative_id",
    "asset_file",
    "cta_text",
    "headline",
    "subhead",
    "advertiser_name",
    "vertical",
    "format",
}


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    ensure_exists(path, label)
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} missing required columns: {joined}")


def load_creative_tables(creative_summary_path: Path, creatives_path: Path | None = None) -> pd.DataFrame:
    """
    Load creative-level metadata.

    `creative_summary.csv` is preferred because it already includes KPI columns.
    If some expected metadata fields are missing, they are backfilled from `creatives.csv`.
    """
    summary_df = load_csv(creative_summary_path, "creative_summary")

    missing = REQUIRED_CREATIVE_COLUMNS.difference(summary_df.columns)
    if not missing:
        return summary_df

    if creatives_path is None:
        joined = ", ".join(sorted(missing))
        raise ValueError(
            "creative_summary.csv is missing required metadata columns and --creatives "
            f"was not provided. Missing: {joined}"
        )

    creatives_df = load_csv(creatives_path, "creatives")
    validate_columns(creatives_df, REQUIRED_CREATIVE_COLUMNS, "creatives")

    backfill_cols = sorted(col for col in REQUIRED_CREATIVE_COLUMNS if col != "creative_id")
    backfill = creatives_df[["creative_id", *backfill_cols]].copy()

    merged = summary_df.merge(backfill, on="creative_id", how="left", suffixes=("", "_from_creatives"))
    for col in backfill_cols:
        alt = f"{col}_from_creatives"
        if col not in merged.columns:
            merged[col] = merged[alt]
        else:
            merged[col] = merged[col].fillna(merged[alt])
        merged.drop(columns=[alt], inplace=True, errors="ignore")

    validate_columns(merged, REQUIRED_CREATIVE_COLUMNS, "merged_creative_table")
    return merged


def parse_creative_id_filter(raw_ids: str | None) -> set[int] | None:
    if not raw_ids:
        return None

    out: set[int] = set()
    for token in raw_ids.split(","):
        token = token.strip()
        if not token:
            continue
        out.add(int(token))

    return out or None


def resolve_asset_path(asset_file: str, dataset_root: Path, assets_root: Path) -> Path:
    raw_path = Path(asset_file)

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(assets_root / raw_path.name)
        candidates.append(dataset_root / raw_path)
        candidates.append(dataset_root / "assets" / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Asset file could not be resolved. "
        f"asset_file={asset_file}, candidates={[str(p) for p in candidates]}"
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_existing_features(output_dir: Path, stem: str) -> pd.DataFrame | None:
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def write_dataframe(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    """Write DataFrame as parquet when available, else fall back to CSV."""
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")
