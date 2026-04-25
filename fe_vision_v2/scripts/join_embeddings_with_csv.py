"""
join_embeddings_with_csv.py
---------------------------
Join a CSV table with embeddings/features exported by the visual pipeline.

Examples
--------
# 1) Join CLIP embeddings to creative_summary.csv
python scripts/join_embeddings_with_csv.py \
  --csv ../Smadex_Creative_Intelligence_Dataset_FULL/creative_summary.csv \
  --embeddings outputs/creative_clip_embeddings.parquet \
  --output outputs/creative_summary_with_clip.csv

# 2) Join final PCA table to daily stats
python scripts/join_embeddings_with_csv.py \
  --csv ../Smadex_Creative_Intelligence_Dataset_FULL/creative_daily_country_os_stats.csv \
  --embeddings outputs/creative_visual_features_with_embeddings_pca.parquet \
  --output outputs/creative_daily_with_visual_pca.parquet \
  --how left
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUPPORTED_READ_EXT = {".csv", ".parquet"}
SUPPORTED_WRITE_EXT = {".csv", ".parquet"}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_READ_EXT:
        raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .parquet")
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_WRITE_EXT:
        raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join a base CSV/parquet with embeddings table using a key column.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=True, help="Base table (.csv or .parquet).")
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Embeddings/features table (.csv or .parquet).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output path (.csv or .parquet).")
    parser.add_argument("--join_key", type=str, default="creative_id", help="Join key column.")
    parser.add_argument(
        "--how",
        type=str,
        default="left",
        choices=["left", "inner", "right", "outer"],
        help="Merge strategy.",
    )
    parser.add_argument(
        "--keep_only_embedding_columns",
        action="store_true",
        help=(
            "If set, keep only join key + embedding-style columns from embeddings "
            "(prefixes: clip_, cnn_, clip_pca_, cnn_pca_)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_df = _read_table(args.csv)
    emb_df = _read_table(args.embeddings)

    if args.join_key not in base_df.columns:
        raise KeyError(f"Join key '{args.join_key}' not found in base table: {args.csv}")
    if args.join_key not in emb_df.columns:
        raise KeyError(f"Join key '{args.join_key}' not found in embeddings table: {args.embeddings}")

    base_df[args.join_key] = pd.to_numeric(base_df[args.join_key], errors="coerce")
    emb_df[args.join_key] = pd.to_numeric(emb_df[args.join_key], errors="coerce")

    base_missing_key = int(base_df[args.join_key].isna().sum())
    emb_missing_key = int(emb_df[args.join_key].isna().sum())
    if base_missing_key > 0 or emb_missing_key > 0:
        print(f"Warning: missing keys -> base={base_missing_key}, embeddings={emb_missing_key}")

    if args.keep_only_embedding_columns:
        emb_prefixes = ("clip_", "cnn_", "clip_pca_", "cnn_pca_")
        emb_cols = [
            c
            for c in emb_df.columns
            if c == args.join_key or c.startswith(emb_prefixes)
        ]
        emb_df = emb_df[emb_cols]

    emb_before = len(emb_df)
    emb_df = emb_df.drop_duplicates(subset=[args.join_key])
    emb_after = len(emb_df)
    if emb_before != emb_after:
        print(f"Info: dropped duplicated embedding keys: {emb_before - emb_after}")

    merged = base_df.merge(emb_df, on=args.join_key, how=args.how)
    _write_table(merged, args.output)

    matched_rows = int(merged[args.join_key].notna().sum())
    print("Join completed")
    print(f"- base rows: {len(base_df)}")
    print(f"- embeddings rows (unique key): {len(emb_df)}")
    print(f"- merged rows: {len(merged)}")
    print(f"- output: {args.output}")

    # Quick check: percentage of base rows that found at least one embedding value.
    possible_emb_cols = [
        c for c in merged.columns if c.startswith(("clip_", "cnn_", "clip_pca_", "cnn_pca_"))
    ]
    if possible_emb_cols:
        has_emb = merged[possible_emb_cols].notna().any(axis=1)
        print(f"- rows with embedding values: {int(has_emb.sum())} / {len(merged)}")
    else:
        print("- no embedding columns found after merge")

    print(f"- non-null join key rows in output: {matched_rows}")


if __name__ == "__main__":
    main()
