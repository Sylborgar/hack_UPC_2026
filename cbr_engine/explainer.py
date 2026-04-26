from __future__ import annotations

import pandas as pd


class CBRExplainer:
    def explain(self, neighbors: pd.DataFrame, reuse_summary: dict[str, object], recommendation: dict[str, object], warnings: list[str] | None = None) -> dict[str, object]:
        warnings = warnings or []
        block_cols = [c for c in neighbors.columns if c.startswith("similarity_") and c != "similarity_global"]
        means = {c.replace("similarity_", ""): float(pd.to_numeric(neighbors[c], errors="coerce").mean()) for c in block_cols}
        top_blocks = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:3]
        drivers = [{"block": k, "mean_similarity": round(v, 4)} for k, v in top_blocks]
        n = reuse_summary.get("neighbor_count", 0)
        avg = reuse_summary.get("avg_similarity")
        roas = reuse_summary.get("median_roas")
        ipm = reuse_summary.get("median_ipm")
        top = reuse_summary.get("top_performer_ratio", 0.0)
        action = recommendation.get("action", "INVESTIGATE")
        conf = recommendation.get("confidence", 0.0)
        driver_text = ", ".join(k for k, _ in top_blocks) or "los bloques disponibles"
        marketer = (
            f"Esta creatividad se parece a {n} casos historicos. "
            f"La similitud media es {avg:.2f}. " if avg is not None else f"Esta creatividad tiene {n} vecinos historicos. "
        )
        marketer += (
            f"La similitud viene sobre todo de {driver_text}. "
            f"Los vecinos tuvieron ROAS mediano {roas if roas is not None else 'n/d'} e IPM mediano {ipm if ipm is not None else 'n/d'}. "
            f"El {float(top) * 100:.0f}% fueron top performers. Recomendacion: {action} con confianza {float(conf):.2f}."
        )
        patterns = self._performance_patterns(neighbors)
        technical = (
            f"Top bloques por similitud media: {drivers}. "
            f"Patrones diferenciales detectados: {patterns[:5]}. "
            f"Warnings: {warnings}."
        )
        return {
            "marketer_explanation": marketer,
            "technical_explanation": technical,
            "top_similarity_drivers": drivers,
            "top_performance_patterns": patterns,
            "warnings": warnings,
        }

    @staticmethod
    def _performance_patterns(neighbors: pd.DataFrame) -> list[dict[str, object]]:
        if "creative_status" not in neighbors.columns:
            return []
        high = neighbors[neighbors["creative_status"].astype(str).eq("top_performer")]
        low = neighbors[~neighbors.index.isin(high.index)]
        patterns = []
        for col in ["vertical", "format", "language", "theme", "hook_type", "cta_text"]:
            if col in neighbors.columns and not high.empty:
                patterns.append({"feature": col, "top_values_high": high[col].astype(str).value_counts().head(3).to_dict(), "top_values_other": low[col].astype(str).value_counts().head(3).to_dict()})
        for col in ["overall_roas", "overall_ipm", "overall_ctr", "perf_score"]:
            if col in neighbors.columns and not high.empty:
                patterns.append({"feature": col, "high_mean": _mean(high[col]), "other_mean": _mean(low[col])})
        return patterns


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    return None if values.dropna().empty else float(values.mean())

