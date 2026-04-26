from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cbr_engine import CBRConfig, load_retriever
from cbr_engine.data_loader import write_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    cfg = CBRConfig.from_yaml(args.config)
    retriever = load_retriever(args.index_dir, cfg)
    query_ids = pd.read_csv(args.input_path)[cfg.id_column].tolist()
    frames = []
    for cid in query_ids:
        neigh, _ = retriever.retrieve_by_id(cid, k=args.k)
        frames.append(neigh)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(out, args.output_path, index=False)
    print(f"Wrote {len(out)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
