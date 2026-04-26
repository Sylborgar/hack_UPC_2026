from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FeatureBuildReport:
    selected_columns_by_block: dict[str, list[str]] = field(default_factory=dict)
    dropped_missing_columns: list[str] = field(default_factory=list)
    dropped_forbidden_columns: list[str] = field(default_factory=list)
    dropped_absent_columns: list[str] = field(default_factory=list)
    dropped_non_numeric_columns: list[str] = field(default_factory=list)
    dropped_metadata_columns: list[str] = field(default_factory=list)
    dropped_outcome_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltFeatures:
    columns_by_block: dict[str, list[str]]
    report: FeatureBuildReport


@dataclass
class PreprocessArtifacts:
    matrix: Any
    block_matrices: dict[str, Any]
    preprocessors: dict[str, Any]
    block_slices: dict[str, tuple[int, int]]
    missing_row_ratio: Any

