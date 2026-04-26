from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbr_engine import CBRConfig, build_cbr_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--feature_sets_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = CBRConfig.from_yaml(args.config)
    manifest = build_cbr_index(args.data_path, args.feature_sets_path, cfg, args.output_dir, args.force)
    print(f"Built CBR index at {Path(args.output_dir or cfg.outputs.index_dir)}")
    print(f"Cases={manifest['num_cases']} dim={manifest['vector_dim']} backend={manifest['backend']}")


if __name__ == "__main__":
    main()
