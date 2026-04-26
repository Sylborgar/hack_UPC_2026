from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CBRConfig
from .similarity_blocks import block_agreement_score


class FinalNeighborScorer:
    def __init__(self, config: CBRConfig):
        self.config = config

    def score(self, query: pd.Series, neighbors: pd.DataFrame, block_columns: list[str]) -> pd.DataFrame:
        out = neighbors.copy()
        agreement = []
        context = []
        quality = []
        reasons = []
        for _, row in out.iterrows():
            block_sims = {c.replace("similarity_", ""): float(row[c]) for c in block_columns if c in row and pd.notna(row[c])}
            agree = block_agreement_score(block_sims)
            ctx = self._context_score(query, row)
            qual = self._data_quality_score(row)
            reason = self._reason_codes(query, row, block_sims, agree, qual)
            agreement.append(agree)
            context.append(ctx)
            quality.append(qual)
            reasons.append(reason)
        s = self.config.scoring
        out["block_agreement_score"] = agreement
        out["context_match_score"] = context
        out["data_quality_score"] = quality
        out["final_score"] = (
            s.alpha * out["global_similarity"].astype(float)
            + s.beta * out["block_agreement_score"]
            + s.gamma * out["context_match_score"]
            + s.delta * out["data_quality_score"]
        ).clip(-1.0, 1.0)
        out["reason_codes"] = reasons
        return out.sort_values("final_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _eq(query: pd.Series, row: pd.Series, col: str) -> bool:
        return col in query.index and col in row.index and pd.notna(query[col]) and query[col] == row[col]

    def _context_score(self, query: pd.Series, row: pd.Series) -> float:
        cols = ["vertical", "format", "language"]
        present = [c for c in cols if c in query.index and c in row.index]
        if not present:
            return 0.0
        return float(np.mean([self._eq(query, row, c) for c in present]))

    @staticmethod
    def _data_quality_score(row: pd.Series) -> float:
        missing_penalty = float(row.get("missing_row_ratio", 0.0) or 0.0)
        volume = 0.0
        for col in ("lifecycle_impressions", "first_3d_impressions", "first_7d_impressions", "overall_impressions"):
            if col in row and pd.notna(row[col]):
                volume = max(volume, min(1.0, np.log1p(float(row[col])) / 12.0))
        return float(np.clip(1.0 - missing_penalty + 0.15 * volume, 0.0, 1.0))

    def _reason_codes(self, query: pd.Series, row: pd.Series, block_sims: dict[str, float], agreement: float, quality: float) -> list[str]:
        codes: list[str] = []
        if block_sims.get("clip", 0.0) >= 0.75:
            codes.append("HIGH_CLIP_SIMILARITY")
        if block_sims.get("cnn", 0.0) >= 0.75:
            codes.append("HIGH_CNN_SIMILARITY")
        if self._eq(query, row, "vertical"):
            codes.append("SAME_VERTICAL")
        if self._eq(query, row, "format"):
            codes.append("SAME_FORMAT")
        if quality < 0.45:
            codes.append("LOW_DATA_QUALITY")
        if agreement < 0.35:
            codes.append("LOW_BLOCK_AGREEMENT")
        if self._eq(query, row, "campaign_id"):
            codes.append("POSSIBLE_CAMPAIGN_DUPLICATE")
        return codes

