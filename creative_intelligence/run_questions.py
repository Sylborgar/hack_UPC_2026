from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import CBR_CASES_PATH
from .questions import run_all_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate outputs for the three Smadex creative intelligence questions."
    )
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=CBR_CASES_PATH,
        help="Path to creative_cbr_cases_final.parquet.",
    )
    parser.add_argument(
        "--feature-set",
        default="prelaunch_feature_cols",
        choices=[
            "prelaunch_feature_cols",
            "early_3d_feature_cols",
            "early_7d_feature_cols",
            "early_14d_feature_cols",
            "candidate_feature_cols",
        ],
        help="Feature set used by the CBR similarity module.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all_questions(cases_path=args.cases_path, memory_feature_set=args.feature_set)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
