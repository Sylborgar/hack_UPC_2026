"""
pipeline.py
-----------
Orquestador del pipeline completo de feature engineering visual.
Coordina todos los módulos en el orden correcto y persiste los outputs.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.config import VisualFeatureConfig
from src.data_loader import CreativeDataLoader
from src.embedding_extractors import CLIPEmbeddingExtractor, CNNEmbeddingExtractor
from src.image_validator import ImageValidator
from src.pca_reducer import PCAReducer
from src.visual_features import HandcraftedVisualFeatureExtractor
from src.visualizer import EmbeddingVisualizer

logger = logging.getLogger(__name__)


class VisualFeaturePipeline:
    """
    Pipeline completo:
      1. Carga de datos
      2. Validación de imágenes
      3. Extracción de features interpretables
      4. Embeddings CLIP
      5. Embeddings CNN
      6. PCA de embeddings
      7. Guardado de outputs
      8. Visualizaciones

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config
        config.setup_output_dirs()

    # ─────────────────────────────────────────────────────────────────────
    # Ejecución principal
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("SMADEX VISUAL FEATURE PIPELINE — INICIO")
        logger.info("=" * 60)

        # ── 1. Cargar datos ──────────────────────────────────────────────
        logger.info("[1/7] Cargando datos…")
        loader = CreativeDataLoader(self.config)
        df = loader.load()

        # ── 2. Validar imágenes ──────────────────────────────────────────
        logger.info("[2/7] Validando imágenes…")
        validator = ImageValidator(self.config)
        valid_df, _ = validator.validate(df)
        logger.info(f"  → {len(valid_df)} imágenes válidas para procesar.")

        if valid_df.empty:
            logger.error("No hay imágenes válidas. Abortando pipeline.")
            return

        # ── 3. Features interpretables ───────────────────────────────────
        logger.info("[3/7] Extrayendo features visuales interpretables…")
        feat_extractor = HandcraftedVisualFeatureExtractor(self.config)
        visual_feats_df = feat_extractor.extract_all(valid_df)

        # Unir con metadatos completos
        meta_cols = [c for c in df.columns if c != "image_path"]
        full_features_df = valid_df[meta_cols].merge(
            visual_feats_df, on=self.config.creative_id_column, how="left"
        )
        self._save_parquet(full_features_df, "creative_visual_features.parquet")

        # ── 4. Embeddings CLIP ───────────────────────────────────────────
        logger.info("[4/7] Extrayendo embeddings CLIP…")
        clip_extractor = CLIPEmbeddingExtractor(self.config)
        clip_df = clip_extractor.extract(valid_df)
        self._save_parquet(clip_df, "creative_clip_embeddings.parquet")

        # ── 5. Embeddings CNN ────────────────────────────────────────────
        logger.info("[5/7] Extrayendo embeddings CNN…")
        cnn_extractor = CNNEmbeddingExtractor(self.config)
        cnn_df = cnn_extractor.extract(valid_df)
        self._save_parquet(cnn_df, "creative_cnn_embeddings.parquet")

        # ── 6. PCA ──────────────────────────────────────────────────────
        logger.info("[6/7] Aplicando PCA…")
        reducer = PCAReducer(self.config)
        clip_pca_df = reducer.fit_transform_clip(clip_df)
        cnn_pca_df = reducer.fit_transform_cnn(cnn_df)

        # Unir todo en un único parquet
        cid = self.config.creative_id_column
        combined_df = (
            full_features_df
            .merge(clip_pca_df, on=cid, how="left")
            .merge(cnn_pca_df, on=cid, how="left")
        )
        self._save_parquet(combined_df, "creative_visual_features_with_embeddings_pca.parquet")

        # ── 7. Visualizaciones ───────────────────────────────────────────
        logger.info("[7/7] Generando visualizaciones…")
        visualizer = EmbeddingVisualizer(self.config)

        # Elegir método según disponibilidad de umap
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "tsne"

        # CLIP
        visualizer.plot_embeddings(
            emb_df=clip_df,
            meta_df=df[[c for c in df.columns if c != "image_path"]],
            prefix="clip",
            method=method,
        )
        # CNN
        visualizer.plot_embeddings(
            emb_df=cnn_df,
            meta_df=df[[c for c in df.columns if c != "image_path"]],
            prefix="cnn",
            method=method,
        )
        # Distribuciones de features
        visualizer.plot_feature_distributions(full_features_df)
        visualizer.plot_correlation_heatmap(full_features_df)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETADO en {elapsed:.1f}s")
        logger.info(f"Outputs en: {self.config.output_dir.resolve()}")
        logger.info("=" * 60)

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────

    def _save_parquet(self, df: pd.DataFrame, filename: str) -> None:
        path = self.config.output_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"  Guardado: {path}  ({len(df)} filas × {len(df.columns)} cols)")
