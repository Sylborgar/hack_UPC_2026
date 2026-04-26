from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    start = time.perf_counter()
    box: dict[str, float] = {}
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms == 0)] = 1.0
    return (matrix / norms).astype(np.float32)


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default

