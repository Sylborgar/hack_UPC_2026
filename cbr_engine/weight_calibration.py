from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CBRConfig
from .data_loader import write_dataframe
from .evaluation import run_offline_evaluation
from .reporting import write_calibration_report


class WeightCalibrator:
    def __init__(self, config: CBRConfig):
        self.config = config

    def random_search(self, data_path: str, feature_sets_path: str, output_dir: str, n_trials: int | None = None) -> dict[str, object]:
        n_trials = int(n_trials or self.config.calibration.n_trials)
        rng = np.random.default_rng(self.config.calibration.random_state)
        active = list(self.config.active_blocks())
        rows = []
        best: dict[str, object] | None = None
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for trial in range(n_trials):
            weights = rng.dirichlet(np.ones(len(active))).astype(float)
            cfg = CBRConfig.from_dict(self.config.to_dict())
            for name, w in zip(active, weights):
                cfg.all_blocks()[name].weight = float(w)
            cfg.normalize_weights()
            trial_dir = out / f"trial_{trial:04d}"
            try:
                metrics = run_offline_evaluation(data_path, feature_sets_path, cfg, str(trial_dir), k_values=[self.config.calibration.k_values[0]])
                score = _objective(metrics, self.config.calibration.objective)
            except Exception as exc:
                metrics = {"error": str(exc)}
                score = -np.inf
            row = {"trial": trial, "score": score, "weights": cfg.block_weights(), **metrics}
            rows.append(row)
            if best is None or score > float(best["score"]):
                best = {"trial": trial, "score": score, "weights": cfg.block_weights(), "objective": self.config.calibration.objective, "metrics": metrics}
        write_dataframe(pd.DataFrame(rows), out / "calibration_trials.parquet", index=False)
        best = best or {"score": None, "weights": self.config.block_weights(), "objective": self.config.calibration.objective}
        write_calibration_report(out, best)
        return best


def _objective(metrics: dict[str, object], name: str) -> float:
    if name == "ndcg_at_k":
        vals = [float(v) for k, v in metrics.items() if k.startswith("ndcg_at_") and v is not None]
    else:
        vals = [float(v) for k, v in metrics.items() if k.startswith("temporal_generalization_") and v is not None]
        if not vals:
            vals = [float(v) for k, v in metrics.items() if "neighbor_outcome_correlation" in k and v is not None]
    return float(np.mean(vals)) if vals else -np.inf
