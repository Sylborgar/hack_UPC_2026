from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BlockConfig:
    enabled: bool = True
    weight: float = 0.0
    prefixes: list[str] = field(default_factory=list)
    exclude_prefixes: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)


@dataclass
class IndexConfig:
    backend: str = "faiss"
    faiss_index_type: str = "flat_ip"
    metric: str = "cosine"
    overfetch_multiplier: int = 10


@dataclass
class RetrievalConfig:
    k: int = 10
    min_similarity: float = 0.0
    exclude_self: bool = True
    exclude_same_campaign_default: bool = False
    filters: dict[str, bool] = field(default_factory=lambda: {"same_vertical": False, "same_format": False, "same_language": False})


@dataclass
class MissingValuesConfig:
    numeric_strategy: str = "median"
    categorical_strategy: str = "__missing__"
    max_missing_ratio_per_column: float = 0.8
    max_missing_ratio_per_row: float = 0.7


@dataclass
class CalibrationConfig:
    enabled: bool = True
    method: str = "random_search"
    n_trials: int = 80
    cv_strategy: str = "temporal"
    n_splits: int = 4
    objective: str = "neighbor_outcome_correlation"
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    group_column: str = "campaign_id"
    random_state: int = 42


@dataclass
class OutputConfig:
    index_dir: str = "outputs/cbr_index"
    reports_dir: str = "outputs/cbr_reports"
    calibration_dir: str = "outputs/cbr_calibration"


@dataclass
class ScoringConfig:
    alpha: float = 0.70
    beta: float = 0.15
    gamma: float = 0.10
    delta: float = 0.05


@dataclass
class CBRConfig:
    id_column: str = "creative_id"
    mode: str = "prelaunch"
    feature_set_name: str = "prelaunch_feature_cols"
    target_column: str = "perf_score"
    time_column: str = "creative_launch_date"
    duplicate_id_policy: str = "error"
    scaler: str = "standard"
    metadata_columns: list[str] = field(default_factory=list)
    outcome_columns: list[str] = field(default_factory=list)
    forbidden_feature_prefixes_by_mode: dict[str, list[str]] = field(default_factory=dict)
    embedding_blocks: dict[str, BlockConfig] = field(default_factory=dict)
    numeric_blocks: dict[str, BlockConfig] = field(default_factory=dict)
    categorical_blocks: dict[str, BlockConfig] = field(default_factory=dict)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    missing_values: MissingValuesConfig = field(default_factory=MissingValuesConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CBRConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CBRConfig":
        data = dict(raw)
        for key in ("embedding_blocks", "numeric_blocks", "categorical_blocks"):
            data[key] = {name: BlockConfig(**cfg) for name, cfg in (data.get(key) or {}).items()}
        if "index" in data:
            data["index"] = IndexConfig(**data["index"])
        if "retrieval" in data:
            data["retrieval"] = RetrievalConfig(**data["retrieval"])
        if "missing_values" in data:
            data["missing_values"] = MissingValuesConfig(**data["missing_values"])
        if "calibration" in data:
            data["calibration"] = CalibrationConfig(**data["calibration"])
        if "outputs" in data:
            data["outputs"] = OutputConfig(**data["outputs"])
        if "scoring" in data:
            data["scoring"] = ScoringConfig(**data["scoring"])
        cfg = cls(**data)
        cfg.normalize_weights()
        return cfg

    def all_blocks(self) -> dict[str, BlockConfig]:
        return {**self.embedding_blocks, **self.numeric_blocks, **self.categorical_blocks}

    def active_blocks(self) -> dict[str, BlockConfig]:
        return {name: block for name, block in self.all_blocks().items() if block.enabled and block.weight > 0}

    def block_weights(self) -> dict[str, float]:
        return {name: float(block.weight) for name, block in self.active_blocks().items()}

    def normalize_weights(self) -> None:
        active = [b for b in self.all_blocks().values() if b.enabled and b.weight > 0]
        total = sum(float(b.weight) for b in active)
        if total > 0:
            for block in active:
                block.weight = float(block.weight) / total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=False)

