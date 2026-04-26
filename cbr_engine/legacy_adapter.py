from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CBRConfig
from .data_loader import load_cases
from .pipeline import build_cbr_index, load_retriever


DISPLAY_COLS = [
    "creative_id",
    "campaign_id",
    "advertiser_name",
    "app_name",
    "vertical",
    "format",
    "theme",
    "hook_type",
    "cta_text",
    "headline",
    "subhead",
    "creative_status",
    "perf_score",
    "overall_roas",
    "overall_ipm",
    "overall_ctr",
    "overall_cvr",
    "has_fatigue",
    "fatigue_day",
]


class CreativeMemoryEngineAdapter:
    """Compatibility wrapper for the old creative_intelligence.cbr.CreativeMemory API.

    The dashboard and question pipeline were built around CreativeMemory. This
    adapter keeps that contract while using cbr_engine for feature selection,
    preprocessing, indexing, scoring and traces.
    """

    backend_name = "cbr_engine"

    def __init__(
        self,
        cases_path: str | Path,
        feature_sets_path: str | Path,
        feature_set_name: str = "prelaunch_feature_cols",
        config_path: str | Path = "configs/cbr_prelaunch.yaml",
        index_dir: str | Path = "outputs/cbr_index_app",
        force_rebuild: bool = False,
    ):
        self.cases_path = Path(cases_path)
        self.feature_sets_path = Path(feature_sets_path)
        self.feature_set_name = feature_set_name
        self.config_path = Path(config_path)
        self.index_dir = Path(index_dir)
        self.force_rebuild = force_rebuild

        self.config: CBRConfig | None = None
        self.retriever = None
        self.cases: pd.DataFrame | None = None
        self.matrix: np.ndarray | None = None
        self.id_to_pos: dict[int, int] = {}
        self.context_cols: list[str] = []
        self.numeric_cols: list[str] = []
        self.embedding_cols: list[str] = []
        self.feature_cols: list[str] = []
        self.columns_by_block: dict[str, list[str]] = {}
        self.manifest: dict[str, object] = {}
        self._similar_cache: dict[tuple[int, int, bool, bool, bool], pd.DataFrame] = {}

    def fit(self) -> "CreativeMemoryEngineAdapter":
        config = CBRConfig.from_yaml(self.config_path)
        config.feature_set_name = self.feature_set_name
        config.mode = _mode_from_feature_set(self.feature_set_name, config.mode)
        config.normalize_weights()
        self.config = config
        self.cases = load_cases(self.cases_path)

        manifest_path = self.index_dir / "build_manifest.json"
        should_rebuild = self.force_rebuild or not manifest_path.exists()
        if not should_rebuild:
            import json

            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            should_rebuild = (
                self.manifest.get("feature_set_name") != config.feature_set_name
                or self.manifest.get("mode") != config.mode
                or self.manifest.get("weights") != config.block_weights()
            )

        if should_rebuild:
            self.manifest = build_cbr_index(
                data_path=str(self.cases_path),
                feature_sets_path=str(self.feature_sets_path),
                config=config,
                output_dir=str(self.index_dir),
                force=True,
            )

        self.retriever = load_retriever(str(self.index_dir), config)
        self.matrix = self.retriever.matrix
        self.id_to_pos = {int(cid): pos for pos, cid in enumerate(self.retriever.ids)}
        self.columns_by_block = dict(self.manifest.get("columns_by_block") or {})
        self.embedding_cols = self.columns_by_block.get("clip", []) + self.columns_by_block.get("cnn", [])
        self.context_cols = self.columns_by_block.get("categorical_context", [])
        self.numeric_cols = [
            col
            for block in ("visual_numeric", "text_numeric", "business_early")
            for col in self.columns_by_block.get(block, [])
        ]
        self.feature_cols = list(dict.fromkeys(self.context_cols + self.numeric_cols + self.embedding_cols))
        return self

    def _require_fit(self) -> None:
        if self.retriever is None or self.matrix is None or self.cases is None:
            raise RuntimeError("CreativeMemoryEngineAdapter.fit() must be called before querying.")

    def similar_cases(
        self,
        creative_id: int,
        k: int = 10,
        same_vertical: bool = False,
        same_format: bool = False,
        exclude_same_campaign: bool = False,
        extra_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self._require_fit()
        cache_key = (int(creative_id), int(k), bool(same_vertical), bool(same_format), bool(exclude_same_campaign))
        if cache_key not in self._similar_cache:
            neighbors, _ = self.retriever.retrieve_by_id(
                creative_id,
                k=k,
                filters={
                    "same_vertical": same_vertical,
                    "same_format": same_format,
                    "exclude_same_campaign": exclude_same_campaign,
                },
            )
            self._similar_cache[cache_key] = self._materialize_neighbors(neighbors)

        cached = self._similar_cache[cache_key]
        if cached.empty:
            return _empty_neighbor_frame(extra_columns)
        out = cached.copy()
        keep_cols = list(dict.fromkeys(DISPLAY_COLS + list(extra_columns or [])))
        keep_cols = [col for col in keep_cols if col in out.columns]
        engine_cols = [
            "final_score",
            "similarity_clip",
            "similarity_cnn",
            "similarity_visual_numeric",
            "similarity_text_numeric",
            "similarity_business_early",
            "similarity_categorical_context",
            "block_agreement_score",
            "context_match_score",
            "data_quality_score",
            "reason_codes",
        ]
        engine_cols = [col for col in engine_cols if col in out.columns and col not in keep_cols]
        return out[["neighbor_rank", "distance", "similarity"] + keep_cols + engine_cols].reset_index(drop=True)

    def _materialize_neighbors(self, neighbors: pd.DataFrame) -> pd.DataFrame:
        if neighbors.empty:
            return pd.DataFrame()
        source = self.cases.set_index("creative_id", drop=False)
        rows = []
        for _, neighbor in neighbors.iterrows():
            neighbor_id = int(neighbor["neighbor_creative_id"])
            if neighbor_id not in source.index:
                continue
            row = source.loc[neighbor_id].copy()
            similarity = _float_or_none(neighbor.get("global_similarity")) or 0.0
            row["neighbor_rank"] = int(neighbor["rank"])
            row["distance"] = 1.0 - similarity
            row["similarity"] = similarity
            row["final_score"] = _float_or_none(neighbor.get("final_score"))
            row["block_agreement_score"] = _float_or_none(neighbor.get("block_agreement_score"))
            row["context_match_score"] = _float_or_none(neighbor.get("context_match_score"))
            row["data_quality_score"] = _float_or_none(neighbor.get("data_quality_score"))
            row["reason_codes"] = neighbor.get("reason_codes")
            for col in neighbors.columns:
                if col.startswith("similarity_"):
                    row[col] = _float_or_none(neighbor.get(col))
            rows.append(row)
        return pd.DataFrame(rows)

    def summarize_neighbors(self, creative_id: int, k: int = 10) -> dict[str, object]:
        neighbors = self.similar_cases(creative_id, k=k, same_vertical=True)
        if neighbors.empty:
            return {
                "creative_id": int(creative_id),
                "similar_case_count": 0,
                "avg_similarity": None,
                "top_performer_count": 0,
                "fatigued_count": 0,
                "median_perf_score": None,
                "median_roas": None,
            }
        status = neighbors.get("creative_status", pd.Series(dtype=str)).astype(str)
        return {
            "creative_id": int(creative_id),
            "similar_case_count": int(len(neighbors)),
            "avg_similarity": _round_or_none(neighbors["similarity"].mean()),
            "avg_final_score": _round_or_none(neighbors.get("final_score", pd.Series(dtype=float)).mean()),
            "max_similarity": _round_or_none(neighbors["similarity"].max()),
            "top_performer_count": int(status.eq("top_performer").sum()),
            "stable_count": int(status.eq("stable").sum()),
            "fatigued_count": int(status.eq("fatigued").sum()),
            "top_performer_ratio": _round_or_none(status.eq("top_performer").mean()),
            "fatigued_ratio": _round_or_none(status.eq("fatigued").mean()),
            "median_perf_score": _round_or_none(neighbors.get("perf_score", pd.Series(dtype=float)).median()),
            "median_roas": _round_or_none(neighbors.get("overall_roas", pd.Series(dtype=float)).median()),
            "median_ipm": _round_or_none(neighbors.get("overall_ipm", pd.Series(dtype=float)).median()),
        }


def _mode_from_feature_set(feature_set_name: str, default: str) -> str:
    if feature_set_name.startswith("early_3d"):
        return "early_3d"
    if feature_set_name.startswith("early_7d"):
        return "early_7d"
    if feature_set_name.startswith("early_14d"):
        return "early_14d"
    if feature_set_name.startswith("prelaunch"):
        return "prelaunch"
    return default


def _empty_neighbor_frame(extra_columns: Iterable[str] | None = None) -> pd.DataFrame:
    keep_cols = list(dict.fromkeys(DISPLAY_COLS + list(extra_columns or [])))
    return pd.DataFrame(columns=["neighbor_rank", "distance", "similarity"] + keep_cols)


def _float_or_none(value: object) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        return None
    return None


def _round_or_none(value: object, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)
