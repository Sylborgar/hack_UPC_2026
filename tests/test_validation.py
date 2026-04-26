from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbr_engine.validation import check_infinite_values, check_leakage_columns, validate_input_dataframe


def test_detects_missing_creative_id(config):
    with pytest.raises(ValueError):
        validate_input_dataframe(pd.DataFrame({"x": [1]}), config)


def test_detects_duplicates(sample_df, config):
    df = pd.concat([sample_df, sample_df.iloc[[0]]])
    with pytest.raises(ValueError):
        validate_input_dataframe(df, config)


def test_replaces_inf_with_nan(sample_df):
    out = check_infinite_values(sample_df)
    assert np.isnan(out.loc[3, "vis_brightness_mean"])


def test_detects_leakage_columns(config):
    with pytest.raises(ValueError):
        check_leakage_columns(["clip_pca_0", "perf_score", "first_3d_ctr"], config)

