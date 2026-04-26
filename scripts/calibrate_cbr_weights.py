from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbr_engine import CBRConfig
from cbr_engine.weight_calibration import WeightCalibrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--feature_sets_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_trials", type=int, default=None)
    args = parser.parse_args()
    cfg = CBRConfig.from_yaml(args.config)
    best = WeightCalibrator(cfg).random_search(args.data_path, args.feature_sets_path, args.output_dir, args.n_trials)
    print(best)


if __name__ == "__main__":
    main()
