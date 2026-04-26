from __future__ import annotations

from cbr_engine.config import CBRConfig
from cbr_engine.feature_builder import FeatureBuilder


def test_excludes_outcomes_and_forbidden_prefixes(sample_df, config, feature_sets_path):
    built = FeatureBuilder(config, str(feature_sets_path)).build(sample_df)
    flat = sum(built.columns_by_block.values(), [])
    assert "perf_score" not in flat
    assert "first_3d_ctr" not in flat
    assert "missing_col" in built.report.dropped_absent_columns


def test_clip_only(sample_df, feature_sets_path):
    cfg = CBRConfig.from_yaml("configs/cbr_clip_only.yaml")
    built = FeatureBuilder(cfg, str(feature_sets_path)).build(sample_df)
    assert set(built.columns_by_block) == {"clip"}


def test_cnn_only(sample_df, feature_sets_path):
    cfg = CBRConfig.from_yaml("configs/cbr_cnn_only.yaml")
    built = FeatureBuilder(cfg, str(feature_sets_path)).build(sample_df)
    assert set(built.columns_by_block) == {"cnn"}

