from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cbr import CreativeMemory
from .landscape import build_creative_landscape
from .paths import CREATIVE_MEMORY_DB_PATH
from .questions import RANKING_OUTPUT_COLS


NEIGHBOR_EXTRA_COLS = [
    "lifecycle_spend_usd",
    "lifecycle_impressions",
    "lifecycle_revenue_usd",
    "dominant_color",
    "emotional_tone",
    "objective",
    "primary_theme",
    "target_age_segment",
    "target_os",
]


def create_creative_memory_db(
    *,
    cases: pd.DataFrame,
    ranking: pd.DataFrame,
    health: pd.DataFrame,
    next_tests: pd.DataFrame,
    recommendation_cards: list[dict[str, Any]],
    memory: CreativeMemory,
    q1_explanations_path: Path,
    q2_explanations_path: Path,
    landscape: pd.DataFrame | None = None,
    db_path: Path = CREATIVE_MEMORY_DB_PATH,
    neighbor_k: int = 20,
) -> Path:
    """Create a SQLite case memory DB for the dashboard/API layer."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    q1_rows = _load_jsonl(q1_explanations_path)
    q2_rows = _load_jsonl(q2_explanations_path)
    q3_rows = recommendation_cards
    neighbors = build_neighbors_table(memory, k=neighbor_k)
    explanations = build_explanations_table(q1_rows, q2_rows, q3_rows)
    if landscape is None:
        landscape = build_creative_landscape(memory)

    with sqlite3.connect(db_path) as conn:
        _to_sql_ready(cases).to_sql("creatives", conn, index=False, if_exists="replace")
        _to_sql_ready(_ranking_public_view(ranking)).to_sql(
            "question1_winner_ranking", conn, index=False, if_exists="replace"
        )
        _to_sql_ready(health).to_sql("question2_creative_health", conn, index=False, if_exists="replace")
        _to_sql_ready(next_tests).to_sql("question3_next_tests", conn, index=False, if_exists="replace")
        _to_sql_ready(neighbors).to_sql("creative_neighbors", conn, index=False, if_exists="replace")
        _to_sql_ready(explanations).to_sql("creative_explanations", conn, index=False, if_exists="replace")
        _to_sql_ready(landscape).to_sql("creative_landscape", conn, index=False, if_exists="replace")

        metadata = pd.DataFrame(
            [
                {"key": "generated_at", "value": datetime.now(UTC).isoformat()},
                {"key": "cases_rows", "value": str(len(cases))},
                {"key": "ranking_rows", "value": str(len(ranking))},
                {"key": "health_rows", "value": str(len(health))},
                {"key": "next_tests_rows", "value": str(len(next_tests))},
                {"key": "neighbors_rows", "value": str(len(neighbors))},
                {"key": "landscape_rows", "value": str(len(landscape))},
                {"key": "landscape_method", "value": str(landscape["landscape_method"].iloc[0])},
                {"key": "neighbor_k", "value": str(neighbor_k)},
                {"key": "memory_feature_set", "value": memory.feature_set_name},
                {"key": "memory_feature_count", "value": str(len(memory.feature_cols))},
                {"key": "recommendation_basis", "value": "aggregate top-k winner pattern"},
            ]
        )
        metadata.to_sql("metadata", conn, index=False, if_exists="replace")

        _create_indexes(conn)

    return db_path


def build_neighbors_table(memory: CreativeMemory, k: int = 20) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cases = memory.cases
    current_lookup = cases.set_index("creative_id", drop=False)

    for creative_id in cases["creative_id"].astype(int):
        current = current_lookup.loc[creative_id]
        neighbors = memory.similar_cases(
            creative_id,
            k=k,
            same_vertical=True,
            extra_columns=NEIGHBOR_EXTRA_COLS,
        )
        for neighbor in neighbors.itertuples(index=False):
            neighbor_id = int(getattr(neighbor, "creative_id"))
            rows.append(
                {
                    "creative_id": creative_id,
                    "neighbor_id": neighbor_id,
                    "neighbor_rank": int(getattr(neighbor, "neighbor_rank")),
                    "distance": _float_or_none(getattr(neighbor, "distance")),
                    "similarity": _float_or_none(getattr(neighbor, "similarity")),
                    "same_vertical": _same(current, neighbor, "vertical"),
                    "same_format": _same(current, neighbor, "format"),
                    "same_theme": _same(current, neighbor, "theme"),
                    "same_hook_type": _same(current, neighbor, "hook_type"),
                    "same_cta_text": _same(current, neighbor, "cta_text"),
                    "neighbor_campaign_id": _int_or_none(getattr(neighbor, "campaign_id", None)),
                    "neighbor_vertical": getattr(neighbor, "vertical", None),
                    "neighbor_format": getattr(neighbor, "format", None),
                    "neighbor_theme": getattr(neighbor, "theme", None),
                    "neighbor_hook_type": getattr(neighbor, "hook_type", None),
                    "neighbor_cta_text": getattr(neighbor, "cta_text", None),
                    "neighbor_status": getattr(neighbor, "creative_status", None),
                    "neighbor_perf_score": _float_or_none(getattr(neighbor, "perf_score", None)),
                    "neighbor_roas": _float_or_none(getattr(neighbor, "overall_roas", None)),
                    "neighbor_ipm": _float_or_none(getattr(neighbor, "overall_ipm", None)),
                    "neighbor_ctr": _float_or_none(getattr(neighbor, "overall_ctr", None)),
                    "neighbor_cvr": _float_or_none(getattr(neighbor, "overall_cvr", None)),
                    "neighbor_has_fatigue": _int_or_none(getattr(neighbor, "has_fatigue", None)),
                    "neighbor_fatigue_day": _float_or_none(getattr(neighbor, "fatigue_day", None)),
                }
            )
    return pd.DataFrame(rows)


def build_explanations_table(
    q1_rows: list[dict[str, Any]],
    q2_rows: list[dict[str, Any]],
    q3_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    q1_by_id = {int(row["creative_id"]): row for row in q1_rows}
    q2_by_id = {int(row["creative_id"]): row for row in q2_rows}
    q3_by_id = {int(row["creative_id"]): row for row in q3_rows}
    creative_ids = sorted(set(q1_by_id) | set(q2_by_id) | set(q3_by_id))

    rows = []
    for creative_id in creative_ids:
        q1 = q1_by_id.get(creative_id, {})
        q2 = q2_by_id.get(creative_id, {})
        q3 = q3_by_id.get(creative_id, {})
        rows.append(
            {
                "creative_id": creative_id,
                "q1_json": _json_dumps(q1),
                "q2_json": _json_dumps(q2),
                "q3_json": _json_dumps(q3),
                "q1_summary": _q1_summary(q1),
                "q2_summary": _q2_summary(q2),
                "q3_summary": _q3_summary(q3),
                "recommended_action": q3.get("recommended_action"),
                "next_test": q3.get("next_test"),
                "recommendation_basis": q3.get("recommendation_basis"),
            }
        )
    return pd.DataFrame(rows)


def _ranking_public_view(ranking: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in RANKING_OUTPUT_COLS if col in ranking.columns]
    return ranking[cols].sort_values("best_score", ascending=False, na_position="last")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _to_sql_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(_sql_value)
    return out.replace({np.nan: None})


def _sql_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)


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
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _q1_summary(row: dict[str, Any]) -> str | None:
    if not row:
        return None
    evidence = row.get("evidence", {})
    similar = row.get("similar_cases", {})
    return (
        f"{row.get('action_hint')} | {row.get('winner_segment')} | "
        f"rank global {row.get('global_rank')} | ROAS {evidence.get('overall_roas')} | "
        f"{similar.get('top_performer_count')}/{similar.get('similar_case_count')} similar cases are top performers"
    )


def _q2_summary(row: dict[str, Any]) -> str | None:
    if not row:
        return None
    evidence = row.get("evidence", {})
    return (
        f"{row.get('action_hint')} | fatigue={row.get('fatigue_status')} | "
        f"repetition={row.get('repetition_status')} | tired_score={evidence.get('tired_score')} | "
        f"repetition_score={evidence.get('repetition_score')}"
    )


def _q3_summary(row: dict[str, Any]) -> str | None:
    if not row:
        return None
    return f"{row.get('recommended_action')} | {row.get('next_test')}"


def _create_indexes(conn: sqlite3.Connection) -> None:
    index_statements = [
        "CREATE INDEX idx_creatives_id ON creatives(creative_id)",
        "CREATE INDEX idx_creatives_vertical_format ON creatives(vertical, format)",
        "CREATE INDEX idx_q1_id ON question1_winner_ranking(creative_id)",
        "CREATE INDEX idx_q2_id ON question2_creative_health(creative_id)",
        "CREATE INDEX idx_q3_id ON question3_next_tests(creative_id)",
        "CREATE INDEX idx_neighbors_id ON creative_neighbors(creative_id)",
        "CREATE INDEX idx_neighbors_neighbor ON creative_neighbors(neighbor_id)",
        "CREATE INDEX idx_explanations_id ON creative_explanations(creative_id)",
        "CREATE INDEX idx_landscape_id ON creative_landscape(creative_id)",
        "CREATE INDEX idx_landscape_vertical_format ON creative_landscape(vertical, format)",
    ]
    for statement in index_statements:
        conn.execute(statement)


def _same(current: pd.Series, neighbor: Any, col: str) -> int | None:
    if col not in current.index or not hasattr(neighbor, col):
        return None
    return int(str(current[col]) == str(getattr(neighbor, col)))


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
