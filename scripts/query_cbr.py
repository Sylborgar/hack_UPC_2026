from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

from cbr_engine import CBRConfig, load_retriever
from cbr_engine.reporting import write_query_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creative_id", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--same_vertical", action="store_true")
    parser.add_argument("--same_format", action="store_true")
    parser.add_argument("--same_language", action="store_true")
    parser.add_argument("--exclude_same_campaign", action="store_true")
    args = parser.parse_args()
    cfg = CBRConfig.from_yaml(args.config)
    retriever = load_retriever(args.index_dir, cfg)
    filters = {
        "same_vertical": args.same_vertical,
        "same_format": args.same_format,
        "same_language": args.same_language,
        "exclude_same_campaign": args.exclude_same_campaign,
    }
    neighbors, trace = retriever.retrieve_by_id(args.creative_id, k=args.k, filters=filters)
    enriched = retriever.enrich(neighbors)
    trace.update(enriched)
    out_dir = Path("outputs/cbr_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_query_trace(out_dir / f"query_{args.creative_id}.json", trace)
    columns = ["rank", "neighbor_creative_id", "global_similarity", "similarity_clip", "similarity_cnn", "similarity_visual_numeric", "similarity_text_numeric", "similarity_categorical_context", "final_score", "creative_status", "perf_score", "overall_roas", "overall_ipm", "reason_codes"]
    if Console is not None and Table is not None:
        console = Console()
        table = Table(title=f"CBR neighbors for {args.creative_id}")
        for col in columns:
            table.add_column(col)
        for _, row in neighbors.iterrows():
            table.add_row(*[_fmt(row.get(col)) for col in columns])
        console.print(table)
        console.print("[bold]Aggregates[/bold]", enriched["reuse_summary"])
        console.print("[bold]Recommendation[/bold]", enriched["recommendation"])
        console.print("[bold]Marketer explanation[/bold]", enriched["explanation"]["marketer_explanation"])
        console.print("[bold]Technical explanation[/bold]", enriched["explanation"]["technical_explanation"])
        if trace.get("warnings"):
            console.print("[yellow]Warnings[/yellow]", trace["warnings"])
    else:
        print(neighbors[[c for c in columns if c in neighbors.columns]].to_string(index=False))
        print("Aggregates:", enriched["reuse_summary"])
        print("Recommendation:", enriched["recommendation"])
        print("Marketer explanation:", enriched["explanation"]["marketer_explanation"])
        print("Technical explanation:", enriched["explanation"]["technical_explanation"])
        if trace.get("warnings"):
            print("Warnings:", trace["warnings"])


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
