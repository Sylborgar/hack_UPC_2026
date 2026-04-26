from __future__ import annotations

import json

from cbr_engine.weight_calibration import WeightCalibrator


def test_random_search_saves_weights(tmp_path, sample_df, config, feature_sets_path):
    data = tmp_path / "cases.parquet"
    sample_df.to_parquet(data, index=False)
    best = WeightCalibrator(config).random_search(str(data), str(feature_sets_path), str(tmp_path / "cal"), n_trials=1)
    assert abs(sum(best["weights"].values()) - 1.0) < 1e-6
    assert (tmp_path / "cal" / "best_weights.json").exists()
