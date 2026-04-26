from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .cbr import CreativeMemory
from .paths import DATASET_OUTPUT


LANDSCAPE_OUTPUT_PATH = DATASET_OUTPUT / "creative_landscape.csv"


def build_creative_landscape(
    memory: CreativeMemory,
    *,
    output_path: Path = LANDSCAPE_OUTPUT_PATH,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a 2D creative map from the same multimodal CBR matrix.

    UMAP is the preferred reducer for the dashboard. If it is unavailable, the
    fallback is t-SNE over a PCA-compressed version of the CBR matrix.
    """
    memory._require_fit()
    matrix = np.asarray(memory.matrix, dtype=np.float32)

    method = "umap"
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.12,
            metric="cosine",
            random_state=random_state,
        )
        coords = reducer.fit_transform(matrix)
    except Exception:
        method = "tsne_pca_fallback"
        compressed = PCA(n_components=min(50, matrix.shape[1]), random_state=random_state).fit_transform(matrix)
        coords = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=random_state,
            max_iter=1000,
        ).fit_transform(compressed)

    coords = _scale_coords(coords)
    cases = memory.cases
    landscape = pd.DataFrame(
        {
            "creative_id": cases["creative_id"].astype(int).to_numpy(),
            "landscape_x": coords[:, 0],
            "landscape_y": coords[:, 1],
            "landscape_method": method,
            "vertical": cases["vertical"].astype(str).to_numpy(),
            "format": cases["format"].astype(str).to_numpy(),
            "creative_status": cases["creative_status"].astype(str).to_numpy(),
            "perf_score": pd.to_numeric(cases["perf_score"], errors="coerce").to_numpy(),
            "overall_roas": pd.to_numeric(cases["overall_roas"], errors="coerce").to_numpy(),
            "asset_file": cases["asset_file"].astype(str).to_numpy(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    landscape.to_csv(output_path, index=False)
    return landscape


def _scale_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where(maxs - mins == 0, 1, maxs - mins)
    return ((coords - mins) / span * 2) - 1
