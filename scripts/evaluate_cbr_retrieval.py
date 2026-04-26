from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbr_engine import CBRConfig
from cbr_engine.evaluation import run_offline_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--feature_sets_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()
    cfg = CBRConfig.from_yaml(args.config)
    metrics = run_offline_evaluation(args.data_path, args.feature_sets_path, cfg, args.output_dir, args.k_values)
    print(metrics)


if __name__ == "__main__":
    main()
