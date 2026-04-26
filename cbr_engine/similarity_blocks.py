from __future__ import annotations

import numpy as np


def block_similarity(query_blocks: dict[str, np.ndarray], candidate_blocks: dict[str, np.ndarray], weights: dict[str, float]) -> dict[str, object]:
    sims: dict[str, float] = {}
    contributions: dict[str, float] = {}
    weighted_sum = 0.0
    for name, q in query_blocks.items():
        if name not in candidate_blocks or q.size == 0:
            continue
        sim = float(np.sum(q * candidate_blocks[name], axis=1)[0])
        sim = max(-1.0, min(1.0, sim))
        sims[name] = sim
        contrib = sim * float(weights.get(name, 0.0))
        contributions[name] = contrib
        weighted_sum += contrib
    denom = sum(float(weights.get(k, 0.0)) for k in sims) or 1.0
    return {"global_similarity": float(weighted_sum / denom), "block_similarities": sims, "block_contributions": contributions}


def compute_pairwise_block_similarities(query_idx: int, candidate_indices: list[int], block_matrices: dict[str, np.ndarray], weights: dict[str, float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, matrix in block_matrices.items():
        q = matrix[query_idx : query_idx + 1]
        c = matrix[candidate_indices]
        out[name] = np.sum(q * c, axis=1).astype(float).tolist()
    return out


def block_agreement_score(block_sims: dict[str, float]) -> float:
    vals = np.array([v for v in block_sims.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 0.0
    positive = np.clip(vals, 0.0, 1.0)
    return float(np.mean(positive) * (1.0 - min(np.std(positive), 1.0)))

