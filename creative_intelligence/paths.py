from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Find the repository root from a script, notebook, or cwd."""
    root = (start or Path.cwd()).resolve()
    while root != root.parent:
        if (root / "dataset" / "final").exists() and (root / ".gitignore").exists():
            return root
        root = root.parent
    return (start or Path.cwd()).resolve()


ROOT = repo_root()
DATASET_FINAL = ROOT / "dataset" / "final"
DATASET_OUTPUT = ROOT / "dataset" / "output"

CBR_CASES_PATH = DATASET_FINAL / "creative_cbr_cases_final.parquet"
CBR_FEATURE_SETS_PATH = DATASET_FINAL / "cbr_feature_sets.json"
CREATIVE_MEMORY_DB_PATH = DATASET_FINAL / "creative_memory.db"

BEST_CREATIVES_OUTPUT = ROOT / "best_creatives" / "outputs"
FATIGUE_REPETITION_OUTPUT = ROOT / "fatigue_repetition" / "outputs"
NEXT_TESTS_OUTPUT = ROOT / "next_tests" / "outputs"
