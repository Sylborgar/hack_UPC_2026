from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CBRConfig


def check_required_columns(df: pd.DataFrame, config: CBRConfig) -> list[str]:
    missing = [config.id_column] if config.id_column not in df.columns else []
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return missing


def check_duplicate_ids(df: pd.DataFrame, config: CBRConfig) -> pd.DataFrame:
    dupes = df[config.id_column].duplicated()
    if not dupes.any():
        return df
    policy = config.duplicate_id_policy
    if policy == "error":
        raise ValueError(f"Duplicate {config.id_column} values detected: {int(dupes.sum())}")
    if policy == "keep_first":
        return df.drop_duplicates(config.id_column, keep="first").reset_index(drop=True)
    if policy == "aggregate":
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        other = [c for c in df.columns if c not in numeric and c != config.id_column]
        agg = {c: "mean" for c in numeric if c != config.id_column}
        agg.update({c: "first" for c in other})
        return df.groupby(config.id_column, as_index=False).agg(agg)
    raise ValueError(f"Unsupported duplicate_id_policy={policy}")


def check_target_availability_for_evaluation(df: pd.DataFrame, config: CBRConfig) -> None:
    if config.target_column not in df.columns:
        raise ValueError(f"Target column {config.target_column} is required for evaluation")


def forbidden_columns(config: CBRConfig, columns: list[str]) -> list[str]:
    prefixes = config.forbidden_feature_prefixes_by_mode.get(config.mode, [])
    out = []
    for col in columns:
        if col in config.outcome_columns or any(col.startswith(prefix) for prefix in prefixes):
            out.append(col)
    return out


def check_leakage_columns(columns: list[str], config: CBRConfig) -> list[str]:
    leaked = forbidden_columns(config, columns)
    if leaked:
        raise ValueError(f"Leakage-prone columns found in feature set: {leaked[:20]}")
    return leaked


def check_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def check_missingness(df: pd.DataFrame, config: CBRConfig) -> dict[str, object]:
    return {
        "row_missing_ratio_max": float(df.isna().mean(axis=1).max()) if len(df) else 0.0,
        "columns_above_threshold": df.columns[df.isna().mean() > config.missing_values.max_missing_ratio_per_column].tolist(),
    }


def validate_input_dataframe(df: pd.DataFrame, config: CBRConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    check_required_columns(df, config)
    df = check_duplicate_ids(df, config)
    df = check_infinite_values(df)
    report = check_missingness(df, config)
    return df, report

