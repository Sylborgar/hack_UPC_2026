"""
visualizer.py
-------------
Genera visualizaciones exploratorias de embeddings y features visuales.
Soporta UMAP y t-SNE para reducción a 2D, y distribuciones de features.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import VisualFeatureConfig

logger = logging.getLogger(__name__)

# Intentar importar UMAP (opcional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn no disponible. Se usará t-SNE en su lugar.")

from sklearn.manifold import TSNE


class EmbeddingVisualizer:
    """
    Genera y guarda plots de exploración de embeddings y features.

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config
        sns.set_theme(style="darkgrid", palette="husl")

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def plot_embeddings(
        self,
        emb_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        prefix: str,
        method: str = "umap",
    ) -> None:
        """
        Proyecta embeddings a 2D y colorea por distintas columnas de metadatos.

        Parameters
        ----------
        emb_df   : DataFrame con creative_id + columnas de embedding
        meta_df  : DataFrame con creative_id + metadatos (campaign_id, etc.)
        prefix   : "clip" o "cnn"
        method   : "umap" | "tsne"
        """
        cid = self.config.creative_id_column
        feat_cols = [c for c in emb_df.columns if c != cid]
        X = emb_df[feat_cols].values.astype(np.float32)

        coords = self._reduce_2d(X, method)
        plot_df = pd.DataFrame(coords, columns=["x", "y"])
        plot_df[cid] = emb_df[cid].values

        # Unir metadatos
        plot_df = plot_df.merge(meta_df[[cid] + self._meta_columns(meta_df)], on=cid, how="left")

        # Colorear por varias columnas
        color_cols = self._meta_columns(plot_df)
        if not color_cols:
            color_cols = [None]

        for color_col in color_cols[:3]:  # máximo 3 plots por embedding
            self._scatter_plot(
                plot_df=plot_df,
                prefix=prefix,
                method=method,
                color_col=color_col,
            )

    def plot_feature_distributions(self, features_df: pd.DataFrame) -> None:
        """Distribuciones de features visuales clave."""
        cols_to_plot = [
            "brightness_mean", "saturation_mean", "edge_density",
            "visual_complexity", "colorfulness", "sharpness_laplacian_var",
            "symmetry_horizontal_score", "symmetry_vertical_score",
            "saliency_center_bias", "empty_space_ratio",
        ]
        existing = [c for c in cols_to_plot if c in features_df.columns]

        if not existing:
            logger.warning("No se encontraron features de distribución para plotear.")
            return

        fig, axes = plt.subplots(
            nrows=(len(existing) + 1) // 2,
            ncols=2,
            figsize=(14, 4 * ((len(existing) + 1) // 2)),
        )
        axes = axes.flatten()

        for i, col in enumerate(existing):
            data = features_df[col].dropna()
            sns.histplot(data, kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel("")

        for j in range(len(existing), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Distribución de features visuales", fontsize=14, y=1.02)
        plt.tight_layout()
        path = self.config.figures_dir / "feature_distributions.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot guardado: {path}")

    def plot_correlation_heatmap(self, features_df: pd.DataFrame) -> None:
        """Mapa de correlación entre features numéricas."""
        numeric = features_df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return

        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(16, 13))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            center=0,
            linewidths=0.3,
            annot=False,
        )
        ax.set_title("Correlación entre features visuales")
        path = self.config.figures_dir / "feature_correlation_heatmap.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot guardado: {path}")

    # ─────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────

    def _reduce_2d(self, X: np.ndarray, method: str) -> np.ndarray:
        """Reduce X a 2D con UMAP o t-SNE."""
        if method == "umap" and UMAP_AVAILABLE:
            logger.info("Reduciendo con UMAP…")
            reducer = umap.UMAP(
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                n_components=2,
                random_state=self.config.random_seed,
            )
        else:
            logger.info("Reduciendo con t-SNE…")
            perplexity = min(30, max(5, X.shape[0] // 5))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=self.config.random_seed,
            )
        return reducer.fit_transform(X)

    def _scatter_plot(
        self,
        plot_df: pd.DataFrame,
        prefix: str,
        method: str,
        color_col: Optional[str],
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        if color_col and color_col in plot_df.columns:
            categories = plot_df[color_col].astype(str).fillna("unknown")
            unique_cats = categories.unique()
            palette = sns.color_palette("husl", len(unique_cats))
            cat_to_color = dict(zip(unique_cats, palette))
            colors = [cat_to_color[c] for c in categories]
            scatter = ax.scatter(
                plot_df["x"], plot_df["y"],
                c=colors, alpha=0.6, s=15
            )
            # Leyenda (solo si hay pocas categorías)
            if len(unique_cats) <= 20:
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=cat_to_color[cat], markersize=8, label=cat)
                    for cat in unique_cats
                ]
                ax.legend(handles=handles, title=color_col, bbox_to_anchor=(1.05, 1),
                          loc="upper left", fontsize=7)
            title = f"{prefix.upper()} embeddings [{method.upper()}] — color: {color_col}"
            fname = f"{prefix}_{method}_{color_col}.png"
        else:
            ax.scatter(plot_df["x"], plot_df["y"], alpha=0.5, s=15, color="steelblue")
            title = f"{prefix.upper()} embeddings [{method.upper()}]"
            fname = f"{prefix}_{method}.png"

        ax.set_title(title)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        plt.tight_layout()
        path = self.config.figures_dir / fname
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot guardado: {path}")

    def _meta_columns(self, df: pd.DataFrame) -> List[str]:
        """Columnas de metadatos disponibles para colorear."""
        candidates = [
            self.config.campaign_id_column,
            "creative_status",
            "vertical",
            "format",
            "creative_type",
        ]
        return [c for c in candidates if c in df.columns]
