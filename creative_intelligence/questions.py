from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cbr import CreativeMemory, load_cases
from .paths import (
    BEST_CREATIVES_OUTPUT,
    CBR_CASES_PATH,
    FATIGUE_REPETITION_OUTPUT,
    NEXT_TESTS_OUTPUT,
)


METRIC_WEIGHTS = {
    "perf_score": 0.40,
    "overall_roas": 0.25,
    "overall_ipm": 0.20,
    "overall_cvr": 0.10,
    "overall_ctr": 0.05,
}

STATUS_BONUS = 0.03

RANKING_OUTPUT_COLS = [
    "creative_id",
    "campaign_id",
    "advertiser_id",
    "advertiser_name",
    "app_name",
    "vertical",
    "format",
    "language",
    "theme",
    "hook_type",
    "cta_text",
    "headline",
    "subhead",
    "dominant_color",
    "emotional_tone",
    "objective",
    "primary_theme",
    "target_age_segment",
    "target_os",
    "creative_launch_date",
    "asset_file",
    "creative_status",
    "best_score",
    "winner_segment",
    "action_hint",
    "global_rank",
    "global_pct",
    "campaign_rank",
    "campaign_pct",
    "vertical_rank",
    "vertical_pct",
    "format_rank",
    "format_pct",
    "vertical_format_rank",
    "vertical_format_pct",
    "perf_score",
    "overall_roas",
    "overall_ipm",
    "overall_ctr",
    "overall_cvr",
    "lifecycle_spend_usd",
    "lifecycle_impressions",
    "lifecycle_revenue_usd",
    "fatigue_day",
    "has_fatigue",
    "ctr_decay_pct",
    "cvr_decay_pct",
    "evidence_ok",
    "low_evidence",
    "perf_score_pct",
    "overall_roas_pct",
    "overall_ipm_pct",
    "overall_cvr_pct",
    "overall_ctr_pct",
    "overall_roas_vs_campaign_median",
    "overall_ipm_vs_vertical_median",
    "perf_score_vs_vertical_format_median",
]

NUMERIC_TEST_TRAITS = [
    "text_density",
    "copy_length_chars",
    "readability_score",
    "brand_visibility_score",
    "clutter_score",
    "novelty_score",
    "motion_score",
    "faces_count",
    "product_count",
    "has_gameplay",
    "has_ugc_style",
    "vis_empty_space_ratio",
    "vis_visual_complexity",
    "vis_edge_density",
    "prompt_text_to_visual_ratio",
    "prompt_visual_hierarchy_score",
    "prompt_brand_clarity_score",
    "prompt_badge_visibility_score",
    "prompt_whitespace_balance",
    "cv_product_zone_coverage",
    "cv_text_area_total_rel",
    "prompt_cta_zone_ratio",
]

CATEGORICAL_TEST_TRAITS = [
    "format",
    "theme",
    "hook_type",
    "cta_text",
    "dominant_color",
    "emotional_tone",
    "prompt_layout_template",
    "prompt_layout_pattern_type",
]

TRAIT_LABELS = {
    "text_density": "text density",
    "copy_length_chars": "copy length",
    "readability_score": "readability",
    "brand_visibility_score": "brand visibility",
    "clutter_score": "visual clutter",
    "novelty_score": "novelty",
    "motion_score": "motion",
    "faces_count": "faces",
    "product_count": "product presence",
    "has_gameplay": "gameplay signal",
    "has_ugc_style": "UGC style",
    "vis_empty_space_ratio": "empty space",
    "vis_visual_complexity": "visual complexity",
    "vis_edge_density": "edge density",
    "prompt_text_to_visual_ratio": "text-to-visual ratio",
    "prompt_visual_hierarchy_score": "visual hierarchy",
    "prompt_brand_clarity_score": "brand clarity",
    "prompt_badge_visibility_score": "badge visibility",
    "prompt_whitespace_balance": "whitespace balance",
    "cv_product_zone_coverage": "product zone coverage",
    "cv_text_area_total_rel": "detected text area",
    "prompt_cta_zone_ratio": "CTA zone size",
}


def build_winner_ranking(cases: pd.DataFrame) -> pd.DataFrame:
    ranking = cases.copy()
    ranking["total_spend_usd"] = ranking["lifecycle_spend_usd"]
    ranking["total_impressions"] = ranking["lifecycle_impressions"]

    spend_p25 = ranking["total_spend_usd"].quantile(0.25)
    impressions_p25 = ranking["total_impressions"].quantile(0.25)
    ranking["evidence_ok"] = ranking["total_spend_usd"].ge(spend_p25) | ranking[
        "total_impressions"
    ].ge(impressions_p25)
    ranking["low_evidence"] = ~ranking["evidence_ok"]

    eligible = ranking["evidence_ok"]
    for metric in METRIC_WEIGHTS:
        ranking[f"{metric}_pct"] = np.nan
        ranking.loc[eligible, f"{metric}_pct"] = ranking.loc[eligible, metric].rank(pct=True)

    ranking["status_bonus"] = np.where(
        ranking["creative_status"].eq("top_performer"), STATUS_BONUS, 0.0
    )
    ranking["best_score"] = sum(
        ranking[f"{metric}_pct"].fillna(0.0) * weight
        for metric, weight in METRIC_WEIGHTS.items()
    ) + ranking["status_bonus"]
    ranking.loc[~eligible, "best_score"] = np.nan

    ranking = _add_rank_columns(ranking, None, "global")
    ranking = _add_rank_columns(ranking, ["campaign_id"], "campaign")
    ranking = _add_rank_columns(ranking, ["vertical"], "vertical")
    ranking = _add_rank_columns(ranking, ["format"], "format")
    ranking = _add_rank_columns(ranking, ["vertical", "format"], "vertical_format")

    top_score_threshold = ranking.loc[eligible, "best_score"].quantile(0.90)
    ranking["is_top_score"] = ranking["best_score"].ge(top_score_threshold)
    ranking["winner_segment"] = np.select(
        [
            ~ranking["evidence_ok"],
            ranking["is_top_score"] & ranking["creative_status"].eq("fatigued"),
            ranking["is_top_score"] & ranking["creative_status"].eq("top_performer"),
            ranking["is_top_score"],
        ],
        ["low_evidence", "strong_but_fatigued", "top_winner", "strong_candidate"],
        default="not_top",
    )
    ranking["action_hint"] = np.select(
        [
            ranking["winner_segment"].eq("low_evidence"),
            ranking["winner_segment"].eq("strong_but_fatigued"),
            ranking["winner_segment"].eq("top_winner"),
            ranking["winner_segment"].eq("strong_candidate"),
        ],
        ["Gather more data", "Refresh", "Scale", "Keep"],
        default="Review",
    )

    ranking = _add_contextual_benchmarks(ranking)
    return ranking


def _add_rank_columns(df: pd.DataFrame, group_cols: list[str] | None, prefix: str) -> pd.DataFrame:
    out = df.copy()
    if group_cols is None:
        out[f"{prefix}_rank"] = out["best_score"].rank(ascending=False, method="min")
        out[f"{prefix}_pct"] = out["best_score"].rank(pct=True)
        out[f"{prefix}_group_size"] = out["best_score"].notna().sum()
        return out

    grouped = out.groupby(group_cols, dropna=False)["best_score"]
    out[f"{prefix}_rank"] = grouped.rank(ascending=False, method="min")
    out[f"{prefix}_pct"] = grouped.rank(pct=True)
    out[f"{prefix}_group_size"] = grouped.transform("count")
    return out


def _add_contextual_benchmarks(ranking: pd.DataFrame) -> pd.DataFrame:
    benchmark_metrics = ["perf_score", "overall_roas", "overall_ipm", "overall_ctr", "overall_cvr"]
    for group_cols, group_name in [
        (["campaign_id"], "campaign"),
        (["vertical"], "vertical"),
        (["vertical", "format"], "vertical_format"),
    ]:
        medians = ranking.groupby(group_cols, dropna=False)[benchmark_metrics].transform("median")
        for metric in benchmark_metrics:
            ranking[f"{metric}_vs_{group_name}_median"] = ranking[metric] - medians[metric]
            ranking[f"{metric}_above_{group_name}_median"] = ranking[metric].gt(medians[metric])
    return ranking


def write_question1_outputs(
    ranking: pd.DataFrame,
    memory: CreativeMemory,
    output_dir: Path = BEST_CREATIVES_OUTPUT,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [col for col in RANKING_OUTPUT_COLS if col in ranking.columns]
    ranking_out = ranking[output_cols].sort_values("best_score", ascending=False, na_position="last")

    paths = {
        "winner_ranking_csv": output_dir / "question1_winner_ranking.csv",
        "winner_ranking_parquet": output_dir / "question1_winner_ranking.parquet",
        "top_global_csv": output_dir / "question1_top_global.csv",
        "top_by_campaign_csv": output_dir / "question1_top_by_campaign.csv",
        "top_by_vertical_csv": output_dir / "question1_top_by_vertical.csv",
        "top_by_vertical_format_csv": output_dir / "question1_top_by_vertical_format.csv",
        "winner_explanations_jsonl": output_dir / "question1_winner_explanations.jsonl",
    }

    ranking_out.to_csv(paths["winner_ranking_csv"], index=False)
    ranking_out.to_parquet(paths["winner_ranking_parquet"], index=False)
    ranking_out.head(20).to_csv(paths["top_global_csv"], index=False)
    _top_by_group(ranking_out, ["campaign_id"], 3).to_csv(paths["top_by_campaign_csv"], index=False)
    _top_by_group(ranking_out, ["vertical"], 10).to_csv(paths["top_by_vertical_csv"], index=False)
    _top_by_group(ranking_out, ["vertical", "format"], 5).to_csv(
        paths["top_by_vertical_format_csv"], index=False
    )

    explanation_rows = []
    for row in ranking_out.head(50).itertuples(index=False):
        creative_id = int(row.creative_id)
        neighbor_summary = memory.summarize_neighbors(creative_id, k=10)
        explanation_rows.append(
            {
                "creative_id": creative_id,
                "question": "Which creatives are working best?",
                "answer_type": "winner_explanation",
                "winner_segment": getattr(row, "winner_segment"),
                "action_hint": getattr(row, "action_hint"),
                "best_score": _round_or_none(getattr(row, "best_score")),
                "global_rank": _int_or_none(getattr(row, "global_rank")),
                "vertical_rank": _int_or_none(getattr(row, "vertical_rank")),
                "format_rank": _int_or_none(getattr(row, "format_rank")),
                "evidence": {
                    "perf_score": _round_or_none(getattr(row, "perf_score")),
                    "overall_roas": _round_or_none(getattr(row, "overall_roas")),
                    "overall_ipm": _round_or_none(getattr(row, "overall_ipm")),
                    "overall_ctr": _round_or_none(getattr(row, "overall_ctr")),
                    "overall_cvr": _round_or_none(getattr(row, "overall_cvr")),
                    "spend_usd": _round_or_none(getattr(row, "lifecycle_spend_usd")),
                    "impressions": _int_or_none(getattr(row, "lifecycle_impressions")),
                },
                "similar_cases": neighbor_summary,
            }
        )
    _write_jsonl(paths["winner_explanations_jsonl"], explanation_rows)
    return paths


def _top_by_group(df: pd.DataFrame, group_cols: list[str], n: int) -> pd.DataFrame:
    return (
        df[df["evidence_ok"]]
        .sort_values(group_cols + ["best_score"], ascending=[True] * len(group_cols) + [False])
        .groupby(group_cols, dropna=False)
        .head(n)
        .reset_index(drop=True)
    )


def build_health_table(cases: pd.DataFrame, memory: CreativeMemory, k: int = 20) -> pd.DataFrame:
    health = cases[
        [
            col
            for col in [
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
                "fatigue_day",
                "has_fatigue",
                "perf_score",
                "overall_roas",
                "overall_ipm",
                "overall_ctr",
                "overall_cvr",
                "ctr_decay_pct",
                "cvr_decay_pct",
                "peak_rolling_ctr_5",
            ]
            if col in cases.columns
        ]
    ].copy()

    fatigue_label = (
        health["creative_status"].astype(str).eq("fatigued").astype(float)
        if "creative_status" in health
        else 0.0
    )
    if "has_fatigue" in health:
        fatigue_label = np.maximum(fatigue_label, health["has_fatigue"].fillna(0).astype(float))

    ctr_decay_severity = _positive_quantile_score(-health["ctr_decay_pct"])
    cvr_decay_severity = _positive_quantile_score(-health["cvr_decay_pct"])
    fatigue_day_severity = _fatigue_day_score(health["fatigue_day"])

    health["fatigue_label_score"] = fatigue_label
    health["ctr_decay_severity"] = ctr_decay_severity
    health["cvr_decay_severity"] = cvr_decay_severity
    health["early_fatigue_severity"] = fatigue_day_severity
    health["tired_score"] = (
        0.35 * health["fatigue_label_score"]
        + 0.25 * health["ctr_decay_severity"]
        + 0.20 * health["cvr_decay_severity"]
        + 0.20 * health["early_fatigue_severity"]
    ).clip(0, 1)

    repetition_rows = []
    for creative_id in health["creative_id"]:
        neighbors = memory.similar_cases(int(creative_id), k=k, same_vertical=True)
        repetition_rows.append(_repetition_features(cases, int(creative_id), neighbors))

    repetition = pd.DataFrame(repetition_rows)
    health = health.merge(repetition, on="creative_id", how="left")
    health["similarity_density_pct"] = health["avg_top10_similarity"].rank(pct=True).fillna(0)
    health["neighbor_density_pct"] = health["similar_neighbor_count_080"].rank(pct=True).fillna(0)
    health["repetition_score"] = (
        0.50 * health["similarity_density_pct"]
        + 0.25 * health["same_metadata_ratio"].fillna(0)
        + 0.25 * health["neighbor_density_pct"]
    ).clip(0, 1)
    health["creative_health_risk"] = np.maximum(health["tired_score"], health["repetition_score"])

    health["fatigue_status"] = np.select(
        [
            health["tired_score"].ge(0.65),
            health["tired_score"].ge(0.40),
        ],
        ["tired", "at_risk"],
        default="healthy",
    )
    health["repetition_status"] = np.select(
        [
            health["repetition_score"].ge(0.75),
            health["repetition_score"].ge(0.55),
        ],
        ["high_repetition", "moderate_repetition"],
        default="distinct",
    )
    health["health_action_hint"] = np.select(
        [
            health["fatigue_status"].eq("tired") | health["repetition_status"].eq("high_repetition"),
            health["fatigue_status"].eq("at_risk")
            | health["repetition_status"].eq("moderate_repetition"),
        ],
        ["Refresh", "Monitor"],
        default="Keep",
    )
    return health.sort_values("creative_health_risk", ascending=False)


def _positive_quantile_score(values: pd.Series) -> pd.Series:
    positive = pd.to_numeric(values, errors="coerce").clip(lower=0).fillna(0)
    scale = positive.quantile(0.95)
    if pd.isna(scale) or scale <= 0:
        return pd.Series(0.0, index=values.index)
    return (positive / scale).clip(0, 1)


def _fatigue_day_score(values: pd.Series) -> pd.Series:
    days = pd.to_numeric(values, errors="coerce")
    observed = days.dropna()
    if observed.empty:
        return pd.Series(0.0, index=values.index)
    max_day = observed.quantile(0.95)
    if pd.isna(max_day) or max_day <= 0:
        return pd.Series(0.0, index=values.index)
    score = 1 - (days / max_day)
    return score.clip(0, 1).fillna(0)


def _repetition_features(
    cases: pd.DataFrame, creative_id: int, neighbors: pd.DataFrame
) -> dict[str, object]:
    current = cases.loc[cases["creative_id"].astype(int).eq(int(creative_id))].iloc[0]
    if neighbors.empty:
        return {
            "creative_id": creative_id,
            "avg_top5_similarity": 0.0,
            "avg_top10_similarity": 0.0,
            "max_similarity": 0.0,
            "similar_neighbor_count_080": 0,
            "same_metadata_ratio": 0.0,
            "fatigued_neighbor_ratio": 0.0,
            "top_neighbor_status_mix": {},
        }

    top10 = neighbors.head(10)
    metadata_cols = [col for col in ["format", "theme", "hook_type", "cta_text"] if col in neighbors]
    metadata_matches = []
    for col in metadata_cols:
        metadata_matches.append(top10[col].astype(str).eq(str(current[col])).mean())
    status_mix = top10["creative_status"].astype(str).value_counts(normalize=True).round(4).to_dict()
    return {
        "creative_id": creative_id,
        "avg_top5_similarity": round(float(neighbors.head(5)["similarity"].mean()), 4),
        "avg_top10_similarity": round(float(top10["similarity"].mean()), 4),
        "max_similarity": round(float(neighbors["similarity"].max()), 4),
        "similar_neighbor_count_080": int(neighbors["similarity"].ge(0.80).sum()),
        "same_metadata_ratio": round(float(np.mean(metadata_matches)) if metadata_matches else 0.0, 4),
        "fatigued_neighbor_ratio": round(float(top10["creative_status"].astype(str).eq("fatigued").mean()), 4),
        "top_neighbor_status_mix": status_mix,
    }


def write_question2_outputs(
    health: pd.DataFrame,
    output_dir: Path = FATIGUE_REPETITION_OUTPUT,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "creative_health_csv": output_dir / "question2_creative_health.csv",
        "creative_health_parquet": output_dir / "question2_creative_health.parquet",
        "tired_creatives_csv": output_dir / "question2_tired_creatives.csv",
        "repetitive_creatives_csv": output_dir / "question2_repetitive_creatives.csv",
        "health_summary_csv": output_dir / "question2_health_summary.csv",
        "health_explanations_jsonl": output_dir / "question2_health_explanations.jsonl",
    }
    health.to_csv(paths["creative_health_csv"], index=False)
    health.to_parquet(paths["creative_health_parquet"], index=False)
    health[health["fatigue_status"].isin(["tired", "at_risk"])].head(100).to_csv(
        paths["tired_creatives_csv"], index=False
    )
    health[health["repetition_status"].isin(["high_repetition", "moderate_repetition"])].head(
        100
    ).to_csv(paths["repetitive_creatives_csv"], index=False)
    summary = (
        health.groupby(["fatigue_status", "repetition_status", "health_action_hint"], dropna=False)
        .size()
        .reset_index(name="creatives")
        .sort_values("creatives", ascending=False)
    )
    summary.to_csv(paths["health_summary_csv"], index=False)
    explanation_rows = []
    for row in health.head(200).itertuples(index=False):
        explanation_rows.append(
            {
                "creative_id": int(row.creative_id),
                "question": "Which creatives look repetitive or tired?",
                "answer_type": "fatigue_repetition_explanation",
                "fatigue_status": row.fatigue_status,
                "repetition_status": row.repetition_status,
                "action_hint": row.health_action_hint,
                "evidence": {
                    "tired_score": _round_or_none(row.tired_score),
                    "repetition_score": _round_or_none(row.repetition_score),
                    "fatigue_day": _round_or_none(row.fatigue_day),
                    "has_fatigue": _int_or_none(row.has_fatigue),
                    "ctr_decay_pct": _round_or_none(row.ctr_decay_pct),
                    "cvr_decay_pct": _round_or_none(row.cvr_decay_pct),
                    "avg_top10_similarity": _round_or_none(row.avg_top10_similarity),
                    "same_metadata_ratio": _round_or_none(row.same_metadata_ratio),
                    "fatigued_neighbor_ratio": _round_or_none(row.fatigued_neighbor_ratio),
                    "top_neighbor_status_mix": row.top_neighbor_status_mix,
                },
                "llm_instruction": (
                    "Explain whether the creative is tired, repetitive, both, or safe to keep. "
                    "Use only the evidence fields."
                ),
            }
        )
    _write_jsonl(paths["health_explanations_jsonl"], explanation_rows)
    return paths


def build_next_test_table(
    cases: pd.DataFrame,
    ranking: pd.DataFrame,
    health: pd.DataFrame,
    memory: CreativeMemory,
    k: int = 20,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranking_lookup = ranking.set_index("creative_id", drop=False)
    health_lookup = health.set_index("creative_id", drop=False)
    population_std = cases[[c for c in NUMERIC_TEST_TRAITS if c in cases.columns]].std(numeric_only=True)

    rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for creative_id in cases["creative_id"].astype(int):
        current = cases.loc[cases["creative_id"].astype(int).eq(creative_id)].iloc[0]
        rank_row = ranking_lookup.loc[creative_id]
        health_row = health_lookup.loc[creative_id]
        neighbors = memory.similar_cases(creative_id, k=k, same_vertical=True)
        winner_neighbors = neighbors[neighbors["creative_status"].astype(str).eq("top_performer")]

        if len(winner_neighbors) < 3:
            winner_neighbors = _fallback_winner_pool(cases, current, ranking)

        suggestions = _build_test_suggestions(current, winner_neighbors, population_std)
        action = _choose_action(rank_row, health_row)
        next_test = _compose_next_test(action, suggestions)
        evidence = _recommendation_evidence(rank_row, health_row, neighbors, winner_neighbors)

        row = {
            "creative_id": creative_id,
            "campaign_id": current.get("campaign_id"),
            "vertical": current.get("vertical"),
            "format": current.get("format"),
            "creative_status": current.get("creative_status"),
            "recommended_action": action,
            "next_test": next_test,
            "priority_score": _priority_score(rank_row, health_row),
            "best_score": _round_or_none(rank_row.get("best_score")),
            "winner_segment": rank_row.get("winner_segment"),
            "fatigue_status": health_row.get("fatigue_status"),
            "repetition_status": health_row.get("repetition_status"),
            "tired_score": _round_or_none(health_row.get("tired_score")),
            "repetition_score": _round_or_none(health_row.get("repetition_score")),
            "similar_case_count": int(len(neighbors)),
            "similar_top_performer_count": int(
                neighbors["creative_status"].astype(str).eq("top_performer").sum()
            ),
            "similar_fatigued_count": int(neighbors["creative_status"].astype(str).eq("fatigued").sum()),
            "avg_similarity": _round_or_none(neighbors["similarity"].mean()),
            "suggestion_1": suggestions[0] if len(suggestions) > 0 else None,
            "suggestion_2": suggestions[1] if len(suggestions) > 1 else None,
            "suggestion_3": suggestions[2] if len(suggestions) > 2 else None,
        }
        rows.append(row)
        cards.append(
            {
                "creative_id": creative_id,
                "question": "What should we test next?",
                "recommended_action": action,
                "next_test": next_test,
                "evidence": evidence,
                "suggestions": suggestions,
                "llm_instruction": (
                    "Use only this structured evidence. Do not invent causes; write a concise "
                    "marketer-facing explanation and next-test recommendation."
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values("priority_score", ascending=False)
    return table, cards


def _fallback_winner_pool(cases: pd.DataFrame, current: pd.Series, ranking: pd.DataFrame) -> pd.DataFrame:
    candidates = cases[cases["creative_status"].astype(str).eq("top_performer")]
    same_vertical = candidates[candidates["vertical"].eq(current["vertical"])]
    if len(same_vertical) >= 3:
        candidates = same_vertical
    same_format = candidates[candidates["format"].eq(current["format"])]
    if len(same_format) >= 3:
        candidates = same_format
    top_ids = (
        ranking[ranking["creative_id"].isin(candidates["creative_id"])]
        .sort_values("best_score", ascending=False)
        .head(20)["creative_id"]
    )
    return candidates[candidates["creative_id"].isin(top_ids)]


def _build_test_suggestions(
    current: pd.Series,
    winner_neighbors: pd.DataFrame,
    population_std: pd.Series,
) -> list[str]:
    suggestions: list[str] = []
    if winner_neighbors.empty:
        return ["test a clearer hook with a fresh visual angle"]

    for col in CATEGORICAL_TEST_TRAITS:
        if col not in winner_neighbors.columns or col not in current.index:
            continue
        mode = winner_neighbors[col].dropna().astype(str).mode()
        if mode.empty:
            continue
        candidate = mode.iloc[0]
        mode_share = winner_neighbors[col].astype(str).eq(candidate).mean()
        if candidate != str(current[col]) and mode_share >= 0.35:
            suggestions.append(f"test {col}='{candidate}'")

    for col in NUMERIC_TEST_TRAITS:
        if col not in winner_neighbors.columns or col not in current.index:
            continue
        current_value = pd.to_numeric(current[col], errors="coerce")
        winner_value = pd.to_numeric(winner_neighbors[col], errors="coerce").median()
        std = population_std.get(col, np.nan)
        if pd.isna(current_value) or pd.isna(winner_value) or pd.isna(std) or std == 0:
            continue
        delta = winner_value - current_value
        if abs(delta) < 0.35 * std:
            continue
        direction = "increase" if delta > 0 else "reduce"
        suggestions.append(f"{direction} {TRAIT_LABELS.get(col, col)}")

    if not suggestions:
        suggestions.append("keep the strongest pattern but refresh hook, CTA, or layout")
    return suggestions[:5]


def _choose_action(rank_row: pd.Series, health_row: pd.Series) -> str:
    winner_segment = str(rank_row.get("winner_segment"))
    fatigue_status = str(health_row.get("fatigue_status"))
    repetition_status = str(health_row.get("repetition_status"))
    best_score = rank_row.get("best_score")

    if winner_segment == "low_evidence":
        return "Gather evidence"
    if fatigue_status == "tired" or repetition_status == "high_repetition":
        return "Refresh"
    if winner_segment == "top_winner":
        return "Scale"
    if winner_segment == "strong_candidate":
        return "Keep"
    if pd.notna(best_score) and best_score < 0.35:
        return "Pause or rebuild"
    return "Test next variation"


def _compose_next_test(action: str, suggestions: list[str]) -> str:
    if action == "Scale":
        return "Scale the creative while testing a controlled variant: " + "; ".join(suggestions[:2])
    if action == "Refresh":
        return "Refresh the creative: " + "; ".join(suggestions[:3])
    if action == "Pause or rebuild":
        return "Rebuild the concept before scaling: " + "; ".join(suggestions[:3])
    if action == "Gather evidence":
        return "Keep collecting data before deciding; test a small variant if budget allows."
    if action == "Keep":
        return "Keep running, and queue a controlled test: " + "; ".join(suggestions[:2])
    return "Test next variation: " + "; ".join(suggestions[:3])


def _recommendation_evidence(
    rank_row: pd.Series,
    health_row: pd.Series,
    neighbors: pd.DataFrame,
    winner_neighbors: pd.DataFrame,
) -> dict[str, Any]:
    status = neighbors["creative_status"].astype(str) if not neighbors.empty else pd.Series(dtype=str)
    return {
        "ranking": {
            "best_score": _round_or_none(rank_row.get("best_score")),
            "winner_segment": rank_row.get("winner_segment"),
            "global_rank": _int_or_none(rank_row.get("global_rank")),
            "vertical_rank": _int_or_none(rank_row.get("vertical_rank")),
            "format_rank": _int_or_none(rank_row.get("format_rank")),
        },
        "health": {
            "fatigue_status": health_row.get("fatigue_status"),
            "repetition_status": health_row.get("repetition_status"),
            "tired_score": _round_or_none(health_row.get("tired_score")),
            "repetition_score": _round_or_none(health_row.get("repetition_score")),
        },
        "similar_cases": {
            "count": int(len(neighbors)),
            "top_performer_count": int(status.eq("top_performer").sum()),
            "fatigued_count": int(status.eq("fatigued").sum()),
            "avg_similarity": _round_or_none(neighbors["similarity"].mean())
            if not neighbors.empty
            else None,
            "winner_reference_count": int(len(winner_neighbors)),
        },
    }


def _priority_score(rank_row: pd.Series, health_row: pd.Series) -> float:
    health_risk = float(health_row.get("creative_health_risk", 0) or 0)
    if str(rank_row.get("winner_segment")) in {"top_winner", "strong_candidate"}:
        return round(0.40 + 0.60 * health_risk, 4)
    best_score = rank_row.get("best_score")
    opportunity = 1 - float(best_score) if pd.notna(best_score) else 0.50
    return round(0.60 * opportunity + 0.40 * health_risk, 4)


def write_question3_outputs(
    next_tests: pd.DataFrame,
    cards: list[dict[str, Any]],
    output_dir: Path = NEXT_TESTS_OUTPUT,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "next_tests_csv": output_dir / "question3_next_tests.csv",
        "next_tests_parquet": output_dir / "question3_next_tests.parquet",
        "priority_tests_csv": output_dir / "question3_priority_tests.csv",
        "recommendation_cards_jsonl": output_dir / "question3_recommendation_cards.jsonl",
    }
    next_tests.to_csv(paths["next_tests_csv"], index=False)
    next_tests.to_parquet(paths["next_tests_parquet"], index=False)
    next_tests.head(100).to_csv(paths["priority_tests_csv"], index=False)
    _write_jsonl(paths["recommendation_cards_jsonl"], cards)
    return paths


def run_all_questions(
    cases_path: Path | str = CBR_CASES_PATH,
    memory_feature_set: str = "prelaunch_feature_cols",
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    memory = CreativeMemory(cases_path=cases_path, feature_set_name=memory_feature_set).fit()

    ranking = build_winner_ranking(cases)
    q1_paths = write_question1_outputs(ranking, memory)

    health = build_health_table(cases, memory, k=20)
    q2_paths = write_question2_outputs(health)

    next_tests, cards = build_next_test_table(cases, ranking, health, memory, k=20)
    q3_paths = write_question3_outputs(next_tests, cards)

    return {
        "cases_shape": list(cases.shape),
        "memory_feature_set": memory_feature_set,
        "memory_features": {
            "context_cols": memory.context_cols,
            "numeric_cols": memory.numeric_cols,
            "embedding_cols": memory.embedding_cols,
            "total": len(memory.feature_cols),
        },
        "question1": {key: str(value) for key, value in q1_paths.items()},
        "question2": {key: str(value) for key, value in q2_paths.items()},
        "question3": {key: str(value) for key, value in q3_paths.items()},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _round_or_none(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _int_or_none(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)
