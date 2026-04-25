"""
image_validator.py
------------------
Valida la existencia y legibilidad de cada imagen del dataset.
Genera un informe de calidad (image_quality_report.csv).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import VisualFeatureConfig

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Comprueba si cada imagen es accesible y se puede abrir correctamente.

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Valida todas las imágenes del DataFrame.

        Returns
        -------
        valid_df : pd.DataFrame
            Filas con imágenes accesibles y válidas.
        report_df : pd.DataFrame
            Informe completo (una fila por creative).
        """
        records: List[Dict] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Validando imágenes"):
            cid = row.get(self.config.creative_id_column, "unknown")
            img_path: Optional[Path] = row.get("image_path")
            status, reason, width, height = self._check_image(img_path)
            records.append(
                {
                    self.config.creative_id_column: cid,
                    "image_path": str(img_path) if img_path else None,
                    "status": status,
                    "reason": reason,
                    "width": width,
                    "height": height,
                }
            )

        report_df = pd.DataFrame(records)
        self._log_summary(report_df)
        self._save_report(report_df)

        valid_ids = report_df.loc[report_df["status"] == "ok", self.config.creative_id_column]
        valid_df = df[df[self.config.creative_id_column].isin(valid_ids)].copy()
        return valid_df, report_df

    # ─────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_image(
        img_path: Optional[Path],
    ) -> Tuple[str, str, Optional[int], Optional[int]]:
        """
        Intenta abrir la imagen con PIL.

        Returns
        -------
        status  : "ok" | "missing" | "corrupt"
        reason  : descripción del error o ""
        width   : ancho en píxeles o None
        height  : alto en píxeles o None
        """
        if img_path is None:
            return "missing", "path no resuelto", None, None

        if not Path(img_path).exists():
            return "missing", "archivo no encontrado", None, None

        try:
            with Image.open(img_path) as img:
                img.verify()          # comprueba integridad sin cargar píxeles

            # Segundo open necesario tras verify()
            with Image.open(img_path) as img:
                w, h = img.size
            return "ok", "", w, h

        except Exception as exc:
            return "corrupt", str(exc), None, None

    def _log_summary(self, report: pd.DataFrame) -> None:
        total = len(report)
        ok = (report["status"] == "ok").sum()
        missing = (report["status"] == "missing").sum()
        corrupt = (report["status"] == "corrupt").sum()
        logger.info("=" * 50)
        logger.info("INFORME DE VALIDACIÓN DE IMÁGENES")
        logger.info(f"  Total creatividades : {total}")
        logger.info(f"  Imágenes válidas    : {ok}")
        logger.info(f"  Imágenes faltantes  : {missing}")
        logger.info(f"  Imágenes corruptas  : {corrupt}")
        logger.info("=" * 50)

    def _save_report(self, report: pd.DataFrame) -> None:
        out = self.config.output_dir / "image_quality_report.csv"
        report.to_csv(out, index=False)
        logger.info(f"Informe guardado en: {out}")
