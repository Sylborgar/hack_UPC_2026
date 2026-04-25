"""
parquet_to_csv.py
-----------------
Convert one or more Parquet files to CSV.

Examples
--------
# Convert one file and save next to it
python scripts/parquet_to_csv.py \
  --input outputs/creative_visual_features_with_embeddings_pca.parquet

# Convert one file and choose output path
python scripts/parquet_to_csv.py \
  --input outputs/creative_visual_features_with_embeddings_pca.parquet \
  --output outputs/creative_visual_features_with_embeddings_pca.csv

# Convert all parquet files in a folder (non-recursive)
python scripts/parquet_to_csv.py \
  --input_dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .parquet files to .csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input parquet file (e.g. outputs/table.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path for single file mode.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing parquet files to convert.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set with --input_dir, include parquet files in subfolders.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=",",
        help="CSV separator (',' or ';', etc.).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV encoding.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input and not args.input_dir:
        raise ValueError("Provide --input or --input_dir.")
    if args.input and args.input_dir:
        raise ValueError("Use either --input or --input_dir, not both.")
    if args.output and not args.input:
        raise ValueError("--output can only be used with --input.")


def _convert_one_file(input_path: Path, output_path: Path, sep: str, encoding: str) -> tuple[int, int]:
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"Input must be .parquet: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=sep, encoding=encoding)
    return df.shape


def main() -> None:
    args = parse_args()
    _validate_args(args)

    converted = 0

    if args.input:
        input_path = args.input
        output_path = args.output if args.output else input_path.with_suffix(".csv")
        rows, cols = _convert_one_file(input_path, output_path, args.sep, args.encoding)
        converted += 1
        print(f"Converted: {input_path} -> {output_path} ({rows} rows, {cols} cols)")
    else:
        input_dir = args.input_dir
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        pattern = "**/*.parquet" if args.recursive else "*.parquet"
        parquet_files = sorted(input_dir.glob(pattern))
        if not parquet_files:
            print(f"No parquet files found in: {input_dir}")
            return

        for parquet_file in parquet_files:
            output_path = parquet_file.with_suffix(".csv")
            rows, cols = _convert_one_file(parquet_file, output_path, args.sep, args.encoding)
            converted += 1
            print(f"Converted: {parquet_file} -> {output_path} ({rows} rows, {cols} cols)")

    print(f"Done. Total files converted: {converted}")


if __name__ == "__main__":
    main()
