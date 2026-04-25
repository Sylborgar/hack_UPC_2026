"""
pca_reducer.py
--------------
Aplica PCA a los embeddings CLIP y CNN.
Persiste los modelos PCA entrenados en outputs/models/.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import VisualFeatureConfig

logger = logging.getLogger(__name__)


class PCAReducer:
    """
    Reduce la dimensionalidad de embeddings con PCA.

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config
        self._pca_clip: PCA | None = None
        self._pca_cnn: PCA | None = None

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def fit_transform_clip(self, clip_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica PCA a los embeddings CLIP."""
        return self._reduce(
            emb_df=clip_df,
            n_components=self.config.clip_pca_components,
            prefix="clip_pca",
            model_name="pca_clip.joblib",
            attr="_pca_clip",
        )

    def fit_transform_cnn(self, cnn_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica PCA a los embeddings CNN."""
        return self._reduce(
            emb_df=cnn_df,
            n_components=self.config.cnn_pca_components,
            prefix="cnn_pca",
            model_name="pca_cnn.joblib",
            attr="_pca_cnn",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────

    def _reduce(
        self,
        emb_df: pd.DataFrame,
        n_components: int,
        prefix: str,
        model_name: str,
        attr: str,
    ) -> pd.DataFrame:
        cid_col = self.config.creative_id_column
        feat_cols = [c for c in emb_df.columns if c != cid_col]
        X = emb_df[feat_cols].values.astype(np.float32)

        # Ajustar n_components al mínimo posible
        n_components = min(n_components, X.shape[0], X.shape[1])
        logger.info(f"PCA ({prefix}): {X.shape[1]}D → {n_components}D")

        pca = PCA(n_components=n_components, random_state=self.config.random_seed)
        X_reduced = pca.fit_transform(X)
        setattr(self, attr, pca)

        var_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"  Varianza explicada: {var_explained:.2%}")

        pca_df = pd.DataFrame(
            X_reduced,
            columns=[f"{prefix}_{i}" for i in range(n_components)],
        )
        pca_df.insert(0, cid_col, emb_df[cid_col].values)

        self._save_model(pca, model_name)
        return pca_df

    def _save_model(self, pca: PCA, filename: str) -> None:
        path = self.config.models_dir / filename
        joblib.dump(pca, path)
        logger.info(f"Modelo PCA guardado: {path}")
