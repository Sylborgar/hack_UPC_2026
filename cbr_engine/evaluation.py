from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, KFold

from .config import CBRConfig
from .data_loader import load_cases, write_dataframe
from .pipeline import build_cbr_index, load_retriever
from .reporting import write_evaluation_report


def evaluate_index(retriever, k_values: list[int] | None = None) -> tuple[dict[str, object], pd.DataFrame]:
    k_values = k_values or [5, 10, 20]
    rows = []
    meta = retriever.metadata
    target = retriever.config.target_column
    ids = retriever.ids
    for i, cid in enumerate(ids):
        if target not in meta.columns or pd.isna(meta.iloc[i][target]):
            continue
        for k in k_values:
            neigh, _ = retriever.retrieve_by_id(cid, k=k, filters={"fallback_relax_filters": False})
            if neigh.empty:
                continue
            vals = pd.to_numeric(neigh.get(target), errors="coerce")
            weights = pd.to_numeric(neigh.get("final_score"), errors="coerce").clip(lower=0)
            pred = float((vals.fillna(vals.mean()) * (weights / weights.sum())).sum()) if weights.sum() > 0 and vals.notna().any() else float(vals.mean())
            rows.append({"creative_id": cid, "k": k, "actual": float(meta.iloc[i][target]), "neighbor_prediction": pred, "self_in_neighbors": bool((neigh["neighbor_creative_id"].astype(str) == str(cid)).any()), **_label_metrics(meta.iloc[i], neigh)})
    per = pd.DataFrame(rows)
    metrics: dict[str, object] = {}
    for k in k_values:
        sub = per[per["k"] == k]
        metrics[f"neighbor_outcome_correlation_pearson_at_{k}"] = _corr(sub, "pearson")
        metrics[f"neighbor_outcome_correlation_spearman_at_{k}"] = _corr(sub, "spearman")
        metrics[f"top_k_label_consistency_at_{k}"] = float(sub["label_consistency"].mean()) if "label_consistency" in sub else None
        metrics[f"hit_rate_top_performer_at_{k}"] = float(sub["hit_top_performer"].mean()) if "hit_top_performer" in sub else None
        metrics[f"fatigue_retrieval_consistency_at_{k}"] = float(sub["fatigue_consistency"].mean()) if "fatigue_consistency" in sub else None
        metrics[f"ndcg_at_{k}"] = float(sub["ndcg"].mean()) if "ndcg" in sub else None
    metrics["self_neighbor_violations"] = int(per["self_in_neighbors"].sum()) if not per.empty else 0
    return metrics, per


def run_offline_evaluation(data_path: str, feature_sets_path: str, config: CBRConfig, output_dir: str, k_values: list[int] | None = None) -> dict[str, object]:
    index_dir = str(Path(output_dir) / "_eval_index")
    build_cbr_index(data_path, feature_sets_path, config, index_dir, force=True)
    retriever = load_retriever(index_dir, config)
    metrics, per = evaluate_index(retriever, k_values)
    temporal = temporal_generalization(data_path, feature_sets_path, config, output_dir, k_values or [5, 10, 20])
    metrics.update(temporal)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_dataframe(per, out / "per_query_results.parquet", index=False)
    write_evaluation_report(out, metrics)
    return metrics


def temporal_generalization(data_path: str, feature_sets_path: str, config: CBRConfig, output_dir: str, k_values: list[int]) -> dict[str, object]:
    df = load_cases(data_path)
    if config.time_column not in df.columns or config.target_column not in df.columns or len(df) < 10:
        return {}
    ordered = df.assign(__time=pd.to_datetime(df[config.time_column], errors="coerce")).sort_values("__time")
    split = max(3, int(len(ordered) * 0.8))
    train = ordered.iloc[:split].drop(columns=["__time"])
    test = ordered.iloc[split:].drop(columns=["__time"])
    if train.empty or test.empty:
        return {}
    out = Path(output_dir)
    train_path = write_dataframe(train, out / "_temporal_train.csv", index=False)
    index_dir = str(out / "_temporal_index")
    build_cbr_index(str(train_path), feature_sets_path, config, index_dir, force=True)
    retriever = load_retriever(index_dir, config)
    rows = []
    for _, row in test.iterrows():
        actual = row.get(config.target_column)
        if pd.isna(actual):
            continue
        for k in k_values:
            neigh, _ = retriever.retrieve_by_row(row, k=k, filters={"fallback_relax_filters": False})
            vals = pd.to_numeric(neigh.get(config.target_column), errors="coerce")
            weights = pd.to_numeric(neigh.get("final_score"), errors="coerce").clip(lower=0)
            if neigh.empty or not vals.notna().any():
                continue
            pred = float((vals.fillna(vals.mean()) * (weights / weights.sum())).sum()) if weights.sum() > 0 else float(vals.mean())
            rows.append({"k": k, "actual": float(actual), "neighbor_prediction": pred})
    per = pd.DataFrame(rows)
    write_dataframe(per, out / "temporal_generalization_results.parquet", index=False)
    return {f"temporal_generalization_pearson_at_{k}": _corr(per[per["k"] == k], "pearson") for k in k_values}


def split_indices(df: pd.DataFrame, config: CBRConfig):
    if config.time_column in df.columns:
        order = pd.to_datetime(df[config.time_column], errors="coerce").sort_values().index.to_numpy()
        return np.array_split(order, config.calibration.n_splits)
    if config.calibration.group_column in df.columns and df[config.calibration.group_column].nunique() >= config.calibration.n_splits:
        return [test for _, test in GroupKFold(config.calibration.n_splits).split(df, groups=df[config.calibration.group_column])]
    return [test for _, test in KFold(config.calibration.n_splits, shuffle=True, random_state=config.calibration.random_state).split(df)]


def _corr(sub: pd.DataFrame, method: str) -> float | None:
    sub = sub[["actual", "neighbor_prediction"]].dropna()
    if len(sub) < 3 or sub["actual"].nunique() < 2 or sub["neighbor_prediction"].nunique() < 2:
        return None
    return float((pearsonr if method == "pearson" else spearmanr)(sub["actual"], sub["neighbor_prediction"]).statistic)


def _label_metrics(query: pd.Series, neigh: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if "creative_status" in neigh.columns and "creative_status" in query.index:
        same = neigh["creative_status"].astype(str).eq(str(query["creative_status"]))
        out["label_consistency"] = float(same.mean())
        out["hit_top_performer"] = float(str(query["creative_status"]) == "top_performer" and neigh["creative_status"].astype(str).eq("top_performer").any())
        rel = neigh["creative_status"].astype(str).eq(str(query["creative_status"])).astype(float).to_numpy()
        out["ndcg"] = _ndcg(rel)
    else:
        out["label_consistency"] = None
        out["hit_top_performer"] = None
        out["ndcg"] = None
    if "has_fatigue" in neigh.columns and "has_fatigue" in query.index:
        out["fatigue_consistency"] = float(neigh["has_fatigue"].astype(str).eq(str(query["has_fatigue"])).mean())
    else:
        out["fatigue_consistency"] = None
    return out


def _ndcg(rel: np.ndarray) -> float:
    if rel.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(rel / denom))
    ideal = float(np.sum(np.sort(rel)[::-1] / denom))
    return dcg / ideal if ideal > 0 else 0.0
