from __future__ import annotations

import numpy as np

from cbr_engine.similarity_blocks import block_agreement_score, block_similarity


def test_block_similarity_range_and_contributions():
    q = {"clip": np.array([[1.0, 0.0]], dtype=np.float32), "cnn": np.array([[0.0, 1.0]], dtype=np.float32)}
    c = {"clip": np.array([[0.5, 0.5]], dtype=np.float32), "cnn": np.array([[0.0, 1.0]], dtype=np.float32)}
    out = block_similarity(q, c, {"clip": 0.5, "cnn": 0.5})
    assert -1 <= out["global_similarity"] <= 1
    assert set(out["block_similarities"]) == {"clip", "cnn"}
    assert abs(sum(out["block_contributions"].values()) - 0.75) < 1e-6
    assert block_agreement_score(out["block_similarities"]) > 0

