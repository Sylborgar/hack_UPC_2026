"""
data_loader.py
--------------
Carga y fusiona creatives.csv + creative_summary.csv.
Resuelve las rutas de imágenes y expone un DataFrame unificado.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import VisualFeatureConfig

logger = logging.getLogger(__name__)


class CreativeDataLoader:
    """
    Carga los metadatos de creatividades y resuelve las rutas de imágenes.

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config
        self._df: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────
    # Carga
    # ─────────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Lee, fusiona y enriquece los datos.

        Returns
        -------
        pd.DataFrame
            DataFrame unificado con columna ``image_path`` resuelta.
        """
        creatives = self._read_creatives()
        summary = self._read_summary()
        df = self._merge(creatives, summary)
        df = self._resolve_image_paths(df)
        self._df = df
        logger.info(f"Dataset cargado: {len(df)} creatividades.")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────

    def _read_creatives(self) -> pd.DataFrame:
        data_dir = self._resolve_data_dir()
        path = data_dir / self.config.creatives_csv

        if not path.exists():
            raise FileNotFoundError(f"No se encontró creatives.csv en: {path}")

        df = pd.read_csv(path)
        logger.info(f"creatives.csv leído: {len(df)} filas, columnas: {list(df.columns)}")
        return df

    def _resolve_data_dir(self) -> Path:
        """
        Asegura que data_dir apunte al dataset correcto.

        Prioriza el data_dir configurado. Si no contiene creatives.csv,
        intenta localizar automáticamente una carpeta Smadex hermana de fe_vision_v2.
        """
        configured_dir = self.config.data_dir
        configured_creatives = configured_dir / self.config.creatives_csv
        if configured_creatives.exists():
            return configured_dir

        project_root = Path(__file__).resolve().parents[2]
        smadex_dirs = sorted(
            d for d in project_root.glob("Smadex_Creative_Intelligence*") if d.is_dir()
        )

        for dataset_dir in smadex_dirs:
            creatives_candidate = dataset_dir / self.config.creatives_csv
            if creatives_candidate.exists():
                self.config.data_dir = dataset_dir
                logger.warning(
                    "data_dir no contenía creatives.csv. "
                    f"Usando dataset detectado automáticamente: {dataset_dir}"
                )
                return dataset_dir

        return configured_dir

    def _read_summary(self) -> pd.DataFrame:
        path = self.config.data_dir / self.config.creative_summary_csv
        if not path.exists():
            logger.warning(f"creative_summary.csv no encontrado en: {path}. Se omite.")
            return pd.DataFrame()
        df = pd.read_csv(path)
        logger.info(f"creative_summary.csv leído: {len(df)} filas, columnas: {list(df.columns)}")
        return df

    def _merge(self, creatives: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
        """Left-join de creatives con summary por creative_id."""
        if summary.empty:
            return creatives.copy()

        cid = self.config.creative_id_column

        # Evitar columnas duplicadas: descartar las de summary que ya existen en creatives
        # (excepto la clave de join)
        overlap = [c for c in summary.columns if c in creatives.columns and c != cid]
        if overlap:
            logger.debug(f"Columnas duplicadas eliminadas de summary: {overlap}")
            summary = summary.drop(columns=overlap)

        merged = creatives.merge(summary, on=cid, how="left")
        logger.info(f"Merge completado: {len(merged)} filas, {len(merged.columns)} columnas.")
        return merged

    def _resolve_image_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade la columna ``image_path`` con la ruta absoluta a cada imagen.
        Intenta varias estrategias para localizar el asset.
        """
        candidate_cols = [self.config.asset_path_column, "asset_file", "asset_path", "asset"]
        col = next((c for c in candidate_cols if c in df.columns), None)
        assets_dir = self.config.assets_dir

        if col is None:
            logger.warning(
                f"Ninguna columna de asset encontrada ({candidate_cols}). "
                "Se intentará buscar por creative_id en assets/."
            )
            df["image_path"] = df[self.config.creative_id_column].apply(
                lambda cid: self._guess_path(cid, assets_dir)
            )
        else:
            if col != self.config.asset_path_column:
                logger.info(
                    f"Usando columna de asset detectada automáticamente: '{col}'."
                )
            df["image_path"] = df[col].apply(
                lambda rel: self._resolve_single(rel, assets_dir)
            )

        return df

    @staticmethod
    def _resolve_single(relative: str, assets_dir: Path) -> Optional[Path]:
        """Resuelve un path relativo probando distintas bases."""
        if pd.isna(relative):
            return None
        rel = Path(str(relative).strip())

        candidates = [
            assets_dir / rel,           # base/assets/relative
            assets_dir / rel.name,      # base/assets/<filename>
            assets_dir.parent / rel,    # base/relative (ej: assets/foo.png)
            rel,                        # path absoluto tal cual
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    @staticmethod
    def _guess_path(creative_id: str, assets_dir: Path) -> Optional[Path]:
        """Intenta encontrar la imagen buscando por creative_id (sin columna asset_path)."""
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = assets_dir / f"{creative_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades públicas
    # ─────────────────────────────────────────────────────────────────────

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Llama a load() primero.")
        return self._df
