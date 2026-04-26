from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .paths import CBR_CASES_PATH, CBR_FEATURE_SETS_PATH


CONTEXT_CATEGORICAL = [
    "vertical",
    "format",
    "language",
    "theme",
    "hook_type",
    "cta_text",
    "dominant_color",
    "emotional_tone",
    "objective",
    "primary_theme",
    "target_age_segment",
    "target_os",
    "kpi_goal",
    "hq_region",
    "prompt_dominant_color_name",
    "prompt_background_color_name",
    "prompt_layout_template",
    "prompt_layout_pattern_type",
]

OUTCOME_COLS = [
    "creative_status",
    "fatigue_day",
    "has_fatigue",
    "perf_score",
    "overall_ctr",
    "overall_cvr",
    "overall_ipm",
    "overall_roas",
    "ctr_decay_pct",
    "cvr_decay_pct",
    "peak_rolling_ctr_5",
    "lifecycle_spend_usd",
    "lifecycle_impressions",
    "lifecycle_clicks",
    "lifecycle_conversions",
    "lifecycle_revenue_usd",
    "lifecycle_ctr",
    "lifecycle_cvr",
    "lifecycle_ipm",
    "lifecycle_roas",
]

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


def load_cases(path: Path | str = CBR_CASES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_feature_sets(path: Path | str = CBR_FEATURE_SETS_PATH) -> dict[str, list[str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _as_list(values: Iterable[str] | None) -> list[str]:
    return list(values) if values is not None else []


def _safe_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class CreativeMemory:
    """Reusable CBR/similarity layer.

    The matrix deliberately avoids targets and full-period outcomes. It combines:
    contextual categoricals, numeric creative features, and PCA visual embeddings.
    """

    cases_path: Path | str = CBR_CASES_PATH
    feature_sets_path: Path | str = CBR_FEATURE_SETS_PATH
    feature_set_name: str = "prelaunch_feature_cols"
    context_weight: float = 0.35
    numeric_weight: float = 0.35
    embedding_weight: float = 0.30

    def __post_init__(self) -> None:
        self.cases = load_cases(self.cases_path)
        self.feature_sets = load_feature_sets(self.feature_sets_path)
        self.id_to_pos: dict[int, int] = {}
        self.matrix: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None
        self.context_cols: list[str] = []
        self.numeric_cols: list[str] = []
        self.embedding_cols: list[str] = []
        self.feature_cols: list[str] = []

    def fit(self) -> "CreativeMemory":
        base_features = [
            col
            for col in self.feature_sets.get(self.feature_set_name, [])
            if col in self.cases.columns and col not in OUTCOME_COLS
        ]

        self.context_cols = [col for col in CONTEXT_CATEGORICAL if col in self.cases.columns]
        self.embedding_cols = [
            col
            for col in base_features
            if col.startswith("clip_pca_") or col.startswith("cnn_pca_")
        ]
        self.numeric_cols = [
            col
            for col in base_features
            if col not in self.embedding_cols and pd.api.types.is_numeric_dtype(self.cases[col])
        ]
        self.feature_cols = self.context_cols + self.numeric_cols + self.embedding_cols

        blocks: list[np.ndarray] = []
        if self.context_cols:
            context = self.cases[self.context_cols].fillna("__missing__").astype(str)
            context_matrix = pd.get_dummies(context, dummy_na=False).to_numpy(dtype=np.float32)
            blocks.append(_l2_normalize(context_matrix) * self.context_weight)

        if self.numeric_cols:
            numeric = _safe_numeric_frame(self.cases, self.numeric_cols)
            numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
            means = numeric.mean(axis=0)
            stds = numeric.std(axis=0).replace(0, 1.0)
            numeric_matrix = ((numeric - means) / stds).to_numpy(dtype=np.float32)
            blocks.append(_l2_normalize(numeric_matrix) * self.numeric_weight)

        if self.embedding_cols:
            embedding = _safe_numeric_frame(self.cases, self.embedding_cols)
            embedding = embedding.fillna(embedding.median(numeric_only=True)).fillna(0.0)
            means = embedding.mean(axis=0)
            stds = embedding.std(axis=0).replace(0, 1.0)
            embedding_matrix = ((embedding - means) / stds).to_numpy(dtype=np.float32)
            blocks.append(_l2_normalize(embedding_matrix) * self.embedding_weight)

        if not blocks:
            raise ValueError("No usable CBR features were found.")

        self.matrix = np.concatenate(blocks, axis=1)
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.matrix)
        self.id_to_pos = {int(cid): i for i, cid in enumerate(self.cases["creative_id"])}
        return self

    def _require_fit(self) -> None:
        if self.matrix is None or self.nn is None:
            raise RuntimeError("CreativeMemory.fit() must be called before querying.")

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
        if int(creative_id) not in self.id_to_pos:
            raise KeyError(f"creative_id {creative_id} not found")

        pos = self.id_to_pos[int(creative_id)]
        query = self.cases.iloc[pos]
        n_neighbors = min(len(self.cases), max(k * 12 + 1, 80))
        distances, indices = self.nn.kneighbors(self.matrix[pos : pos + 1], n_neighbors=n_neighbors)

        neighbors = self.cases.iloc[indices[0]].copy()
        neighbors["distance"] = distances[0]
        neighbors["similarity"] = 1 - neighbors["distance"]
        neighbors = neighbors[neighbors["creative_id"].astype(int) != int(creative_id)]

        if same_vertical and "vertical" in neighbors.columns:
            filtered = neighbors[neighbors["vertical"].eq(query["vertical"])]
            if len(filtered) >= min(k, 3):
                neighbors = filtered

        if same_format and "format" in neighbors.columns:
            filtered = neighbors[neighbors["format"].eq(query["format"])]
            if len(filtered) >= min(k, 3):
                neighbors = filtered

        if exclude_same_campaign and "campaign_id" in neighbors.columns:
            filtered = neighbors[~neighbors["campaign_id"].eq(query["campaign_id"])]
            if len(filtered) >= min(k, 3):
                neighbors = filtered

        keep_cols = list(dict.fromkeys(DISPLAY_COLS + _as_list(extra_columns)))
        keep_cols = [col for col in keep_cols if col in neighbors.columns]
        neighbors = neighbors.head(k).reset_index(drop=True)
        neighbors.insert(0, "neighbor_rank", np.arange(1, len(neighbors) + 1))
        return neighbors[["neighbor_rank", "distance", "similarity"] + keep_cols]

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
            "avg_similarity": round(float(neighbors["similarity"].mean()), 4),
            "max_similarity": round(float(neighbors["similarity"].max()), 4),
            "top_performer_count": int(status.eq("top_performer").sum()),
            "stable_count": int(status.eq("stable").sum()),
            "fatigued_count": int(status.eq("fatigued").sum()),
            "top_performer_ratio": round(float(status.eq("top_performer").mean()), 4),
            "fatigued_ratio": round(float(status.eq("fatigued").mean()), 4),
            "median_perf_score": _round_or_none(neighbors.get("perf_score", pd.Series(dtype=float)).median()),
            "median_roas": _round_or_none(neighbors.get("overall_roas", pd.Series(dtype=float)).median()),
            "median_ipm": _round_or_none(neighbors.get("overall_ipm", pd.Series(dtype=float)).median()),
        }


def _round_or_none(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)
