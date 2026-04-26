from __future__ import annotations

from cbr_engine.pipeline import build_cbr_index, load_retriever


def test_retrieve_by_id_excludes_self_and_filters(tmp_path, sample_df, config, feature_sets_path):
    data = tmp_path / "cases.parquet"
    sample_df.to_parquet(data, index=False)
    out = tmp_path / "idx"
    build_cbr_index(str(data), str(feature_sets_path), config, str(out), force=True)
    retriever = load_retriever(str(out), config)
    neigh, _ = retriever.retrieve_by_id(1, k=2, filters={"same_vertical": True, "same_format": True})
    assert "1" not in set(neigh["neighbor_creative_id"].astype(str))
    assert len(neigh) == 2
    assert neigh["vertical"].eq("game").all()
    assert neigh["format"].eq("video").all()

