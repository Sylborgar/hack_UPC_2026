from __future__ import annotations

import numpy as np

from cbr_engine.feature_builder import FeatureBuilder
from cbr_engine.preprocessing import BlockPreprocessor


def test_preprocessing_no_nan_float32_l2(sample_df, config, feature_sets_path):
    built = FeatureBuilder(config, str(feature_sets_path)).build(sample_df)
    artifacts = BlockPreprocessor(config).fit_transform(sample_df, built.columns_by_block)
    assert artifacts.matrix.dtype == np.float32
    assert np.isfinite(artifacts.matrix).all()
    norms = np.linalg.norm(artifacts.matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_onehot_handle_unknown(sample_df, config, feature_sets_path):
    built = FeatureBuilder(config, str(feature_sets_path)).build(sample_df)
    pp = BlockPreprocessor(config)
    pp.fit_transform(sample_df, built.columns_by_block)
    new = sample_df.iloc[[0]].copy()
    new["theme"] = "new_theme"
    out = pp.transform(new)
    assert np.isfinite(out.matrix).all()

