from __future__ import annotations

import pandas as pd

from .config import CBRConfig
from .feature_registry import FeatureRegistry
from .schemas import BuiltFeatures, FeatureBuildReport


class FeatureBuilder:
    def __init__(self, config: CBRConfig, feature_sets_path: str):
        self.config = config
        self.registry = FeatureRegistry(feature_sets_path)

    def build(self, df: pd.DataFrame) -> BuiltFeatures:
        cfg = self.config
        report = FeatureBuildReport()
        base_cols = self.registry.get(cfg.feature_set_name)
        present = [c for c in base_cols if c in df.columns]
        report.dropped_absent_columns = sorted(set(base_cols) - set(present))

        forbidden_prefixes = cfg.forbidden_feature_prefixes_by_mode.get(cfg.mode, [])
        safe_base: list[str] = []
        for col in present:
            if col in cfg.outcome_columns:
                report.dropped_outcome_columns.append(col)
            elif any(col.startswith(prefix) for prefix in forbidden_prefixes):
                report.dropped_forbidden_columns.append(col)
            elif col in cfg.metadata_columns and col not in self._allowed_categorical_columns():
                report.dropped_metadata_columns.append(col)
            else:
                safe_base.append(col)

        missing_ratio = df[safe_base].isna().mean() if safe_base else pd.Series(dtype=float)
        high_missing = missing_ratio[missing_ratio > cfg.missing_values.max_missing_ratio_per_column].index.tolist()
        report.dropped_missing_columns = high_missing
        safe_base = [c for c in safe_base if c not in set(high_missing)]

        columns_by_block: dict[str, list[str]] = {}
        for name, block in cfg.embedding_blocks.items():
            if not block.enabled or block.weight <= 0:
                continue
            cols = self._by_prefix(safe_base, block.prefixes, block.exclude_prefixes)
            cols = self._keep_numeric(df, cols, report)
            columns_by_block[name] = cols

        for name, block in cfg.numeric_blocks.items():
            if not block.enabled or block.weight <= 0:
                continue
            cols = self._by_prefix(safe_base, block.prefixes, block.exclude_prefixes)
            cols = self._keep_numeric(df, cols, report)
            columns_by_block[name] = cols

        for name, block in cfg.categorical_blocks.items():
            if not block.enabled or block.weight <= 0:
                continue
            cols = [c for c in block.columns if c in df.columns and c not in cfg.outcome_columns]
            cols = [c for c in cols if not any(c.startswith(prefix) for prefix in forbidden_prefixes)]
            cols = [c for c in cols if c not in set(high_missing)]
            columns_by_block[name] = cols

        for name, cols in columns_by_block.items():
            if not cols:
                report.warnings.append(f"Block {name} is enabled but has no usable columns")
        if not any(columns_by_block.values()):
            raise ValueError("No usable CBR feature block after leakage and quality filtering")

        report.selected_columns_by_block = {k: list(v) for k, v in columns_by_block.items()}
        return BuiltFeatures(columns_by_block=columns_by_block, report=report)

    def _allowed_categorical_columns(self) -> set[str]:
        allowed: set[str] = set()
        for block in self.config.categorical_blocks.values():
            if block.enabled:
                allowed.update(block.columns)
        return allowed

    @staticmethod
    def _by_prefix(cols: list[str], prefixes: list[str], exclude_prefixes: list[str]) -> list[str]:
        return [
            c
            for c in cols
            if (not prefixes or any(c.startswith(p) for p in prefixes))
            and not any(c.startswith(p) for p in exclude_prefixes)
        ]

    @staticmethod
    def _keep_numeric(df: pd.DataFrame, cols: list[str], report: FeatureBuildReport) -> list[str]:
        good = []
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                good.append(col)
            else:
                report.dropped_non_numeric_columns.append(col)
        return good

