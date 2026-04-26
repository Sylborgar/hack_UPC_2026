from __future__ import annotations

from cbr_engine.evaluation import evaluate_index
from cbr_engine.pipeline import build_cbr_index, load_retriever


def test_evaluation_keys_and_no_self(tmp_path, sample_df, config, feature_sets_path):
    data = tmp_path / "cases.parquet"
    sample_df.to_parquet(data, index=False)
    out = tmp_path / "idx"
    build_cbr_index(str(data), str(feature_sets_path), config, str(out), force=True)
    metrics, per = evaluate_index(load_retriever(str(out), config), [2])
    assert "neighbor_outcome_correlation_pearson_at_2" in metrics
    assert metrics["self_neighbor_violations"] == 0

