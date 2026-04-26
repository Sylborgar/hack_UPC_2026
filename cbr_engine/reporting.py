from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir, write_json


def write_build_report(output_dir: str | Path, manifest: dict[str, object], feature_report: dict[str, object]) -> None:
    out = ensure_dir(output_dir)
    lines = [
        "# CBR Build Report",
        "",
        f"- Mode: {manifest.get('mode')}",
        f"- Feature set: {manifest.get('feature_set_name')}",
        f"- Cases: {manifest.get('num_cases')}",
        f"- Vector dim: {manifest.get('vector_dim')}",
        f"- Backend: {manifest.get('backend')}",
        f"- Blocks: {', '.join(manifest.get('blocks', []))}",
        "",
        "## Weights",
    ]
    for k, v in (manifest.get("weights") or {}).items():
        lines.append(f"- {k}: {v:.4f}")
    lines += ["", "## Warnings"]
    for w in feature_report.get("warnings", []):
        lines.append(f"- {w}")
    (out / "build_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "feature_report.json", feature_report)


def write_query_trace(path: str | Path, payload: dict[str, object]) -> None:
    write_json(path, payload)


def write_evaluation_report(output_dir: str | Path, metrics: dict[str, object]) -> None:
    out = ensure_dir(output_dir)
    write_json(out / "metrics.json", metrics)
    lines = ["# CBR Retrieval Evaluation", ""]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    (out / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_calibration_report(output_dir: str | Path, best: dict[str, object]) -> None:
    out = ensure_dir(output_dir)
    write_json(out / "best_weights.json", best)
    lines = ["# CBR Weight Calibration", "", f"- Objective: {best.get('objective')}", f"- Best score: {best.get('score')}", "", "## Weights"]
    for k, v in (best.get("weights") or {}).items():
        lines.append(f"- {k}: {v:.4f}")
    (out / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")

