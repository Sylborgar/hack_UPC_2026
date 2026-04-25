"""Pipeline orchestration for Florence-based spatial feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .data_io import (
    append_jsonl,
    ensure_output_dir,
    load_creative_tables,
    parse_creative_id_filter,
    read_existing_features,
    resolve_asset_path,
    write_dataframe,
)
from .devices import select_device
from .florence import FlorenceConfig, FlorenceRunner
from .prompt_json import build_prompt_json_object
from .spatial_features import extract_spatial_fingerprint


DEFAULT_KPI_COLUMNS = [
    "overall_ctr",
    "overall_cvr",
    "overall_ipm",
    "overall_roas",
    "perf_score",
    "ctr_decay_pct",
    "cvr_decay_pct",
]


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _prepare_working_table(full_df: pd.DataFrame, creative_ids: str | None, limit: int | None) -> pd.DataFrame:
    work_df = full_df.copy()

    id_filter = parse_creative_id_filter(creative_ids)
    if id_filter:
        work_df = work_df[work_df["creative_id"].isin(id_filter)]

    work_df = work_df.sort_values("creative_id").reset_index(drop=True)

    if limit is not None:
        work_df = work_df.head(limit)

    return work_df


def _merge_with_kpis(features_df: pd.DataFrame, creative_df: pd.DataFrame) -> pd.DataFrame:
    kpi_cols = [col for col in DEFAULT_KPI_COLUMNS if col in creative_df.columns]
    base_cols = [
        "creative_id",
        "campaign_id",
        "creative_status",
        "vertical",
        "format",
        "asset_file",
    ]
    available_base_cols = [col for col in base_cols if col in creative_df.columns]

    lookup_df = creative_df[available_base_cols + kpi_cols].drop_duplicates(subset=["creative_id"])
    merged = features_df.merge(lookup_df, on="creative_id", how="left", suffixes=("", "_kpi"))

    for shared_col in ["vertical", "format", "asset_file"]:
        alt = f"{shared_col}_kpi"
        if alt in merged.columns:
            merged[shared_col] = merged[shared_col].fillna(merged[alt])
            merged.drop(columns=[alt], inplace=True, errors="ignore")

    return merged


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df[col].dtype == bool:
        return df[col].astype(int)
    return pd.to_numeric(df[col], errors="coerce")


def _build_insight_text(segment: str, feature: str, kpi: str, corr: float, n: int) -> str:
    direction = "higher" if corr > 0 else "lower"
    return (
        f"In {segment}, higher {feature} is associated with {direction} {kpi} "
        f"(spearman={corr:.3f}, n={n})."
    )


def _compute_correlation_table(
    merged_df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kpi_cols = [col for col in DEFAULT_KPI_COLUMNS if col in merged_df.columns]
    if not kpi_cols:
        return pd.DataFrame(), pd.DataFrame()

    numeric_cols = merged_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_cols = [
        col
        for col in numeric_cols
        if col not in set(kpi_cols + ["creative_id", "campaign_id", "image_width", "image_height"])
    ]

    all_rows: list[dict[str, Any]] = []

    segments: list[tuple[str, pd.DataFrame]] = [("all", merged_df)]
    if "vertical" in merged_df.columns:
        for vertical, sub_df in merged_df.groupby("vertical", dropna=False):
            segments.append((f"vertical={vertical}", sub_df))

    for segment_name, segment_df in segments:
        if len(segment_df) < 10:
            continue

        working = segment_df.copy()
        for col in feature_cols + kpi_cols:
            working[col] = _to_numeric_series(working, col)

        corr = working[feature_cols + kpi_cols].corr(method="spearman")

        for feature in feature_cols:
            for kpi in kpi_cols:
                value = corr.loc[feature, kpi]
                if pd.isna(value):
                    continue
                all_rows.append(
                    {
                        "segment": segment_name,
                        "feature": feature,
                        "kpi": kpi,
                        "spearman_corr": float(value),
                        "abs_corr": float(abs(value)),
                        "n": int(len(segment_df)),
                        "is_strong": int(abs(value) >= threshold),
                        "insight": _build_insight_text(segment_name, feature, kpi, float(value), int(len(segment_df))),
                    }
                )

    all_corr_df = pd.DataFrame(all_rows)
    if all_corr_df.empty:
        return all_corr_df, all_corr_df

    all_corr_df = all_corr_df.sort_values(by=["abs_corr", "segment"], ascending=[False, True]).reset_index(drop=True)
    strong_df = all_corr_df[all_corr_df["is_strong"] == 1].reset_index(drop=True)
    return all_corr_df, strong_df


def run_pipeline(args: Any) -> dict[str, Path]:
    ensure_output_dir(args.output_dir)

    _log(f"[1/6] Loading creative data from {args.creative_summary}", verbose=True)
    full_creative_df = load_creative_tables(args.creative_summary, args.creatives)

    work_df = _prepare_working_table(full_creative_df, args.creative_ids, args.limit)
    if work_df.empty:
        raise ValueError("No creatives matched the provided filters")

    spatial_stem = "creative_spatial_features"
    analysis_jsonl_path = args.output_dir / args.analysis_jsonl_name

    existing_df = None
    if args.resume:
        existing_df = read_existing_features(args.output_dir, spatial_stem)
        if existing_df is not None and "creative_id" in existing_df.columns:
            processed_ids = set(existing_df["creative_id"].astype(int).tolist())
            before = len(work_df)
            work_df = work_df[~work_df["creative_id"].isin(processed_ids)]
            skipped = before - len(work_df)
            _log(f"[2/6] Resume active: skipping {skipped} already processed creatives", verbose=True)

    if work_df.empty:
        _log("Nothing left to process after resume filtering", verbose=True)
        if existing_df is None:
            raise ValueError("No output exists to reuse")
        final_df = existing_df
    else:
        _log(f"[3/6] Preparing Florence runner (disable_florence={args.disable_florence})", verbose=True)
        runner = None
        if not args.disable_florence:
            device = select_device(args.device)
            runner = FlorenceRunner(
                FlorenceConfig(
                    model_name=args.model_name,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
            )
            _log(f"Florence model loaded on {device}", verbose=True)

        output_jsonl_path = args.output_dir / "creative_spatial_elements.jsonl"
        if args.save_elements_jsonl and args.overwrite and output_jsonl_path.exists():
            output_jsonl_path.unlink()

        if not args.no_analysis_jsonl and args.overwrite and analysis_jsonl_path.exists():
            analysis_jsonl_path.unlink()

        feature_rows: list[dict[str, Any]] = []
        jsonl_buffer: list[dict[str, Any]] = []
        analysis_jsonl_buffer: list[dict[str, Any]] = []

        total = len(work_df)
        _log(f"[4/6] Extracting spatial fingerprints for {total} creatives", verbose=True)

        for idx, row in enumerate(work_df.to_dict(orient="records"), start=1):
            creative_id = int(row["creative_id"])

            try:
                asset_path = resolve_asset_path(row["asset_file"], args.dataset_root, args.assets_root)
            except Exception as exc:
                feature_rows.append(
                    {
                        "creative_id": creative_id,
                        "asset_file": row.get("asset_file"),
                        "source_mode": "none",
                        "extraction_error": f"asset_resolve_error: {exc}",
                    }
                )
                continue

            try:
                image = Image.open(asset_path).convert("RGB")
            except Exception as exc:
                feature_rows.append(
                    {
                        "creative_id": creative_id,
                        "asset_file": row.get("asset_file"),
                        "source_mode": "none",
                        "extraction_error": f"image_open_error: {exc}",
                    }
                )
                continue

            features, artifacts = extract_spatial_fingerprint(
                creative_row=row,
                image=image,
                florence_runner=runner,
                grounding_text=args.grounding_text,
            )
            features["asset_path"] = str(asset_path)
            feature_rows.append(features)

            if not args.no_analysis_jsonl:
                structured = build_prompt_json_object(
                    creative_row=row,
                    image=image,
                    ocr_elements=artifacts.ocr_elements,
                    grounded_elements=artifacts.grounding_elements,
                )
                analysis_jsonl_buffer.append(structured)
                if len(analysis_jsonl_buffer) >= 25:
                    append_jsonl(analysis_jsonl_path, analysis_jsonl_buffer)
                    analysis_jsonl_buffer = []

            if args.save_elements_jsonl:
                jsonl_buffer.append(
                    {
                        "creative_id": creative_id,
                        "asset_file": row.get("asset_file"),
                        "ocr_elements": artifacts.ocr_elements,
                        "grounding_elements": artifacts.grounding_elements,
                        "ocr_raw": artifacts.ocr_raw,
                        "grounding_raw": artifacts.grounding_raw,
                    }
                )
                if len(jsonl_buffer) >= 25:
                    append_jsonl(output_jsonl_path, jsonl_buffer)
                    jsonl_buffer = []

            if idx % args.log_every == 0 or idx == total:
                _log(f"Processed {idx}/{total} creatives", verbose=True)

        if args.save_elements_jsonl and jsonl_buffer:
            append_jsonl(output_jsonl_path, jsonl_buffer)

        if not args.no_analysis_jsonl and analysis_jsonl_buffer:
            append_jsonl(analysis_jsonl_path, analysis_jsonl_buffer)

        new_df = pd.DataFrame(feature_rows)

        if existing_df is not None and not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
            if "creative_id" in final_df.columns:
                final_df = final_df.drop_duplicates(subset=["creative_id"], keep="last")
        else:
            final_df = new_df

        if "creative_id" in final_df.columns:
            final_df = final_df.sort_values("creative_id").reset_index(drop=True)

    _log("[5/6] Writing extracted features", verbose=True)
    spatial_features_path = write_dataframe(final_df, args.output_dir, spatial_stem)

    merged_df = _merge_with_kpis(final_df, full_creative_df)
    merged_path = write_dataframe(merged_df, args.output_dir, "creative_spatial_with_kpis")

    outputs: dict[str, Path] = {
        "spatial_features": spatial_features_path,
        "spatial_with_kpis": merged_path,
    }

    if not args.no_analysis_jsonl and analysis_jsonl_path.exists():
        outputs["prompt_analysis_jsonl"] = analysis_jsonl_path

    if not args.skip_correlation:
        _log("[6/6] Computing spatial x KPI correlations", verbose=True)
        all_corr_df, strong_corr_df = _compute_correlation_table(
            merged_df=merged_df,
            threshold=args.correlation_threshold,
        )

        corr_path = args.output_dir / "spatial_kpi_correlations.csv"
        strong_corr_path = args.output_dir / "spatial_kpi_correlations_strong.csv"

        if all_corr_df.empty:
            corr_path.write_text("segment,feature,kpi,spearman_corr,abs_corr,n,is_strong,insight\n", encoding="utf-8")
            strong_corr_path.write_text("segment,feature,kpi,spearman_corr,abs_corr,n,is_strong,insight\n", encoding="utf-8")
        else:
            all_corr_df.to_csv(corr_path, index=False)
            strong_corr_df.to_csv(strong_corr_path, index=False)

        outputs["correlations"] = corr_path
        outputs["strong_correlations"] = strong_corr_path

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "total_input_creatives": int(len(full_creative_df)),
                "processed_or_available_creatives": int(len(final_df)),
                "model_name": None if args.disable_florence else args.model_name,
                "device": args.device,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["run_summary"] = summary_path

    return outputs
