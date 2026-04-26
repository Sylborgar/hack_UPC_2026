from __future__ import annotations


class CBRRecommender:
    def recommend(self, reuse_summary: dict[str, object]) -> dict[str, object]:
        n = int(reuse_summary.get("neighbor_count", 0) or 0)
        conf = float(reuse_summary.get("confidence_score", 0.0) or 0.0)
        top = float(reuse_summary.get("top_performer_ratio", 0.0) or 0.0)
        fatigued = float(reuse_summary.get("fatigued_ratio", 0.0) or 0.0)
        perf = reuse_summary.get("weighted_perf_score")
        perf_val = float(perf) if perf is not None else 0.0
        reasons: list[str] = []
        risks: list[str] = []
        next_steps: list[str] = []
        if n < 3 or conf < 0.35:
            action = "INVESTIGATE"
            reasons.append("Pocos vecinos robustos o baja confianza.")
            risks.append("La evidencia historica es debil para automatizar una decision.")
            next_steps.append("Relajar filtros o revisar manualmente los creativos mas cercanos.")
        elif top >= 0.55 and perf_val >= 0 and fatigued <= 0.35 and conf >= 0.55:
            action = "SCALE"
            reasons.append("Vecinos similares concentran top performers y baja fatiga.")
            next_steps.append("Escalar presupuesto gradualmente y monitorizar early metrics.")
        elif fatigued >= 0.55 and perf_val <= 0 and conf >= 0.5:
            action = "PAUSE"
            reasons.append("Vecinos similares muestran fatiga o bajo performance.")
            risks.append("Puede repetir patrones creativos agotados.")
            next_steps.append("Pausar o limitar delivery mientras se prepara una variante.")
        elif top < 0.45 and conf >= 0.45:
            action = "PIVOT"
            reasons.append("La similitud es razonable pero los outcomes historicos son mixtos o bajos.")
            next_steps.append("Cambiar hook, layout o propuesta visual antes de invertir mas.")
        else:
            action = "INVESTIGATE"
            reasons.append("Senales historicas conflictivas entre bloques y outcomes.")
            next_steps.append("Comparar vecinos high-performing vs low-performing.")
        return {"action": action, "confidence": conf, "summary": f"{action} con confianza {conf:.2f}", "reasons": reasons, "risks": risks, "suggested_next_steps": next_steps}

