from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbr_engine.config import CBRConfig


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "creative_id": [1, 2, 3, 4, 5, 6],
            "campaign_id": [10, 10, 11, 12, 13, 14],
            "advertiser_id": [1, 1, 2, 2, 3, 3],
            "vertical": ["game", "game", "fin", "game", "fin", "game"],
            "format": ["video", "video", "banner", "video", "banner", "video"],
            "language": ["en", "en", "es", "en", "es", "en"],
            "theme": ["a", "a", "b", "a", "b", "c"],
            "clip_pca_0": [1, 0.9, 0, 0.8, 0, 0.7],
            "clip_pca_1": [0, 0.1, 1, 0.2, 0.9, 0.3],
            "cnn_pca_0": [1, 0.8, 0, 0.6, 0, 0.5],
            "cnn_pca_1": [0, 0.2, 1, 0.4, 0.8, 0.5],
            "vis_brightness_mean": [1, 1.1, 5, np.inf, 5.1, 1.2],
            "txtsel_cta_text_length": [3, 4, 5, 4, 5, 3],
            "first_3d_ctr": [0.1, 0.2, 0.1, 0.2, 0.1, 0.3],
            "perf_score": [1.0, 0.9, -0.5, 0.8, -0.4, 0.7],
            "overall_roas": [2, 2.1, 0.5, 1.9, 0.6, 1.7],
            "overall_ipm": [4, 4.1, 1, 3.8, 1.1, 3.7],
            "overall_ctr": [0.04, 0.04, 0.01, 0.03, 0.01, 0.03],
            "overall_cvr": [0.2, 0.2, 0.05, 0.18, 0.05, 0.17],
            "creative_status": ["top_performer", "top_performer", "fatigued", "stable", "fatigued", "stable"],
            "has_fatigue": [False, False, True, False, True, False],
            "creative_launch_date": pd.date_range("2025-01-01", periods=6),
        }
    )


@pytest.fixture()
def feature_sets_path(tmp_path: Path) -> Path:
    payload = {
        "prelaunch_feature_cols": ["clip_pca_0", "clip_pca_1", "cnn_pca_0", "cnn_pca_1", "vis_brightness_mean", "txtsel_cta_text_length", "first_3d_ctr", "perf_score", "missing_col"],
        "early_3d_feature_cols": ["clip_pca_0", "clip_pca_1", "first_3d_ctr"],
    }
    path = tmp_path / "features.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def config() -> CBRConfig:
    return CBRConfig.from_yaml("configs/cbr_prelaunch.yaml")

