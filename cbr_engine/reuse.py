from __future__ import annotations

import numpy as np
import pandas as pd


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


class ReuseAggregator:
    def __init__(self, target_column: str = "perf_score"):
        self.target_column = target_column

    def summarize(self, neighbors: pd.DataFrame) -> dict[str, object]:
        if neighbors.empty:
            return {"neighbor_count": 0, "confidence_score": 0.0}
        score = _num(neighbors, "final_score").fillna(_num(neighbors, "global_similarity")).clip(lower=0)
        weights = score / score.sum() if score.sum() > 0 else pd.Series(np.ones(len(neighbors)) / len(neighbors), index=neighbors.index)
        perf = _num(neighbors, self.target_column)
        status = neighbors.get("creative_status", pd.Series("", index=neighbors.index)).astype(str)
        out = {
            "neighbor_count": int(len(neighbors)),
            "avg_similarity": float(_num(neighbors, "global_similarity").mean()),
            "avg_final_score": float(_num(neighbors, "final_score").mean()),
            "median_perf_score": _finite_median(perf),
            "mean_perf_score": _finite_mean(perf),
            "median_roas": _finite_median(_num(neighbors, "overall_roas")),
            "median_ipm": _finite_median(_num(neighbors, "overall_ipm")),
            "median_ctr": _finite_median(_num(neighbors, "overall_ctr")),
            "median_cvr": _finite_median(_num(neighbors, "overall_cvr")),
            "top_performer_ratio": float(status.eq("top_performer").mean()),
            "fatigued_ratio": float((status.eq("fatigued") | status.eq("fatigue")).mean()) if len(status) else 0.0,
            "stable_ratio": float(status.eq("stable").mean()),
            "weighted_perf_score": _weighted(perf, weights),
            "weighted_roas": _weighted(_num(neighbors, "overall_roas"), weights),
            "weighted_ipm": _weighted(_num(neighbors, "overall_ipm"), weights),
        }
        sim = max(0.0, min(1.0, out["avg_similarity"]))
        agreement = float(_num(neighbors, "block_agreement_score").fillna(sim).mean())
        dispersion = float(np.nanstd(perf)) if perf.notna().sum() > 1 else 0.0
        dispersion_penalty = min(0.35, dispersion / (abs(float(np.nanmean(perf))) + 1e-6) * 0.1) if perf.notna().any() else 0.1
        count_score = min(1.0, len(neighbors) / 10.0)
        out["confidence_score"] = float(np.clip(0.35 * count_score + 0.35 * sim + 0.25 * agreement - dispersion_penalty, 0.0, 1.0))
        return out


def _weighted(values: pd.Series, weights: pd.Series) -> float | None:
    mask = values.notna()
    if not mask.any():
        return None
    w = weights[mask]
    if w.sum() <= 0:
        return float(values[mask].mean())
    return float((values[mask] * (w / w.sum())).sum())


def _finite_median(values: pd.Series) -> float | None:
    return None if values.dropna().empty else float(values.median())


def _finite_mean(values: pd.Series) -> float | None:
    return None if values.dropna().empty else float(values.mean())
