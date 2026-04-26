from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import CBRConfig
from .data_loader import read_dataframe
from .explainer import CBRExplainer
from .index_store import make_index_store
from .recommender import CBRRecommender
from .reuse import ReuseAggregator
from .scoring import FinalNeighborScorer
from .similarity_blocks import compute_pairwise_block_similarities


class CBRRetriever:
    def __init__(
        self,
        config: CBRConfig,
        ids: np.ndarray,
        matrix: np.ndarray,
        metadata: pd.DataFrame,
        block_matrices: dict[str, np.ndarray],
        index_store: Any,
        preprocessor: Any | None = None,
    ):
        self.config = config
        self.ids = ids
        self.matrix = matrix.astype(np.float32)
        self.metadata = metadata.reset_index(drop=True)
        self.block_matrices = block_matrices
        self.index_store = index_store
        self.preprocessor = preprocessor
        self.id_to_pos = {str(v): i for i, v in enumerate(ids)}
        self.scorer = FinalNeighborScorer(config)

    @classmethod
    def load(cls, index_dir: str | Path, config: CBRConfig):
        path = Path(index_dir)
        ids = np.load(path / "ids.npy", allow_pickle=True)
        matrix = np.load(path / "matrix.npy")
        metadata_path = path / "metadata.parquet"
        metadata = read_dataframe(metadata_path if metadata_path.exists() else path / "metadata.csv")
        blocks = {p.stem: np.load(p) for p in (path / "block_matrices").glob("*.npy")}
        pp = joblib.load(path / "preprocessors.joblib")
        backend = "faiss" if (path / "index.faiss").exists() else "sklearn"
        store = make_index_store(backend).load(path, matrix)
        return cls(config, ids, matrix, metadata, blocks, store, pp["preprocessor"])

    def retrieve_by_id(self, creative_id: Any, k: int | None = None, filters: dict[str, Any] | None = None, return_trace: bool = True) -> tuple[pd.DataFrame, dict[str, object]]:
        key = str(creative_id)
        if key not in self.id_to_pos:
            raise KeyError(f"{self.config.id_column}={creative_id} not found in index")
        pos = self.id_to_pos[key]
        return self.retrieve_by_vector(self.matrix[pos : pos + 1], k=k, filters=filters, query_pos=pos, query_id=creative_id, return_trace=return_trace)

    def retrieve_by_row(self, row: pd.Series | pd.DataFrame, k: int | None = None, filters: dict[str, Any] | None = None):
        if self.preprocessor is None:
            raise RuntimeError("Cannot transform rows without persisted preprocessors")
        df = row.to_frame().T if isinstance(row, pd.Series) else row
        artifacts = self.preprocessor.transform(df)
        return self.retrieve_by_vector(artifacts.matrix, k=k, filters=filters, query_pos=None, query_id=df.iloc[0].get(self.config.id_column))

    def retrieve_by_vector(
        self,
        vector: np.ndarray,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        query_pos: int | None = None,
        query_id: Any | None = None,
        return_trace: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        k = int(k or self.config.retrieval.k)
        filters = self._resolve_filters(filters or {})
        overfetch = min(len(self.ids), max(k * int(self.config.index.overfetch_multiplier), k + 1, 20))
        scores, indices = self.index_store.search(vector.astype(np.float32), overfetch)
        cand_idx = [int(i) for i in indices[0].tolist() if i >= 0]
        score_map = {int(i): float(s) for i, s in zip(indices[0].tolist(), scores[0].tolist()) if int(i) >= 0}
        discarded: list[dict[str, object]] = []
        if query_pos is not None and self.config.retrieval.exclude_self:
            cand_idx = [i for i in cand_idx if i != query_pos]
        query_meta = self.metadata.iloc[query_pos] if query_pos is not None else pd.Series(dtype=object)
        filtered = []
        for i in cand_idx:
            ok, reason = self._passes_filters(query_meta, self.metadata.iloc[i], filters)
            if ok:
                filtered.append(i)
            else:
                discarded.append({"candidate_id": self.ids[i].item() if hasattr(self.ids[i], "item") else self.ids[i], "reason": reason})
        warnings = []
        if len(filtered) < min(k, 3) and filters.get("fallback_relax_filters", True):
            warnings.append("Filters left too few neighbors; relaxed optional filters.")
            filtered = [i for i in cand_idx if not (query_pos is not None and i == query_pos)]
        selected = filtered[:k]
        result = self._build_result(query_id, query_pos, selected, score_map)
        trace = {
            "query_creative_id": query_id,
            "config": {"mode": self.config.mode, "feature_set_name": self.config.feature_set_name, "weights": self.config.block_weights()},
            "filters_applied": filters,
            "k_requested": k,
            "overfetch": overfetch,
            "discarded_by_filter": discarded,
            "warnings": warnings,
        }
        trace["neighbors_final"] = result.to_dict(orient="records")
        return result, trace

    def enrich(self, neighbors: pd.DataFrame) -> dict[str, object]:
        reuse = ReuseAggregator(self.config.target_column).summarize(neighbors)
        recommendation = CBRRecommender().recommend(reuse)
        explanation = CBRExplainer().explain(neighbors, reuse, recommendation)
        return {"reuse_summary": reuse, "recommendation": recommendation, "explanation": explanation}

    def _build_result(self, query_id: Any, query_pos: int | None, selected: list[int], score_map: dict[int, float] | None = None) -> pd.DataFrame:
        if not selected:
            return pd.DataFrame()
        rows = self.metadata.iloc[selected].copy().reset_index(drop=True)
        rows.insert(0, "neighbor_creative_id", [self.ids[i] for i in selected])
        rows.insert(0, "query_creative_id", query_id)
        rows.insert(2, "rank", np.arange(1, len(rows) + 1))
        if query_pos is not None:
            rows["global_similarity"] = (self.matrix[query_pos : query_pos + 1] @ self.matrix[selected].T).ravel()
        else:
            rows["global_similarity"] = [float((score_map or {}).get(i, np.nan)) for i in selected]
        if query_pos is not None:
            block_sims = compute_pairwise_block_similarities(query_pos, selected, self.block_matrices, self.config.block_weights())
            for block, sims in block_sims.items():
                rows[f"similarity_{block}"] = sims
        rows["matched_filters"] = [[] for _ in range(len(rows))]
        query = self.metadata.iloc[query_pos] if query_pos is not None else pd.Series(dtype=object)
        block_cols = [c for c in rows.columns if c.startswith("similarity_")]
        if query_pos is not None:
            rows = self.scorer.score(query, rows, block_cols)
        else:
            rows["final_score"] = rows["global_similarity"]
            rows["reason_codes"] = [[] for _ in range(len(rows))]
        rows["rank"] = np.arange(1, len(rows) + 1)
        return rows

    def _resolve_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        out = dict(self.config.retrieval.filters)
        out["exclude_same_campaign"] = self.config.retrieval.exclude_same_campaign_default
        out.update(filters)
        return out

    @staticmethod
    def _passes_filters(query: pd.Series, row: pd.Series, filters: dict[str, Any]) -> tuple[bool, str | None]:
        pairs = [("same_vertical", "vertical"), ("same_format", "format"), ("same_language", "language")]
        for flag, col in pairs:
            if filters.get(flag) and col in query.index and col in row.index and query[col] != row[col]:
                return False, flag
        if filters.get("exclude_same_campaign") and "campaign_id" in query.index and "campaign_id" in row.index and query["campaign_id"] == row["campaign_id"]:
            return False, "exclude_same_campaign"
        for col in ("advertiser_id", "campaign_id"):
            if col in filters and filters[col] is not None and col in row.index and row[col] != filters[col]:
                return False, col
        if filters.get("min_spend") is not None:
            spend_cols = [c for c in ("lifecycle_spend_usd", "first_7d_spend_usd", "first_3d_spend_usd") if c in row.index]
            values = [float(row[c]) for c in spend_cols if pd.notna(row[c])]
            if values and max(values) < float(filters["min_spend"]):
                return False, "min_spend"
        if filters.get("min_impressions") is not None:
            imp_cols = [c for c in ("lifecycle_impressions", "first_7d_impressions", "first_3d_impressions") if c in row.index]
            values = [float(row[c]) for c in imp_cols if pd.notna(row[c])]
            if values and max(values) < float(filters["min_impressions"]):
                return False, "min_impressions"
        return True, None
