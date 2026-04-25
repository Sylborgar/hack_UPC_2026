"""Command-line options for the visual feature extraction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "Smadex_Creative_Intelligence_Dataset_FULL"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract spatial fingerprints from creative assets with Florence-2 and "
            "export explainable features for KPI analysis."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to Smadex_Creative_Intelligence_Dataset_FULL",
    )
    parser.add_argument(
        "--creative-summary",
        type=Path,
        default=None,
        help="Path to creative_summary.csv. Defaults to <dataset-root>/creative_summary.csv",
    )
    parser.add_argument(
        "--creatives",
        type=Path,
        default=None,
        help="Optional path to creatives.csv used as metadata fallback",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="Asset directory. Defaults to <dataset-root>/assets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted features and reports are written",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Florence-2-base",
        help="Hugging Face model id for Florence-2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Generation max_new_tokens for Florence tasks",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Beam width for Florence generation",
    )
    parser.add_argument(
        "--grounding-text",
        type=str,
        default=(
            "brand logo, sale badge, discount label, price tag, CTA button, "
            "headline text, subheadline text, product image, product packshot"
        ),
        help="Phrase list used with <CAPTION_TO_PHRASE_GROUNDING>",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N creatives after filtering",
    )
    parser.add_argument(
        "--creative-ids",
        type=str,
        default=None,
        help="Comma-separated creative_id list to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing feature file and skip already processed creative_ids",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N creatives",
    )

    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip Spearman correlation report generation",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.4,
        help="Absolute Spearman threshold for highlighting strong correlations",
    )

    parser.add_argument(
        "--disable-florence",
        action="store_true",
        help=(
            "Disable model inference and compute only metadata-driven fallback features. "
            "Useful for dry-runs and debugging file paths."
        ),
    )
    parser.add_argument(
        "--save-elements-jsonl",
        action="store_true",
        help="Save raw OCR/grounding elements for each creative in JSONL format",
    )
    parser.add_argument(
        "--analysis-jsonl-name",
        type=str,
        default="creative_prompt_analysis.jsonl",
        help="Filename for prompt-compatible structured JSONL output",
    )
    parser.add_argument(
        "--no-analysis-jsonl",
        action="store_true",
        help="Disable prompt-compatible structured JSONL export",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.creative_summary is None:
        args.creative_summary = args.dataset_root / "creative_summary.csv"
    if args.creatives is None:
        args.creatives = args.dataset_root / "creatives.csv"
    if args.assets_root is None:
        args.assets_root = args.dataset_root / "assets"

    return args
