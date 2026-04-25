"""
visual_features.py
------------------
Extrae features visuales interpretables (sin embeddings de red neuronal) de cada imagen.
Usa PIL y OpenCV con heurísticas explicables.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import VisualFeatureConfig

logger = logging.getLogger(__name__)


class HandcraftedVisualFeatureExtractor:
    """
    Extrae ~40 features visuales interpretables para cada imagen.

    Incluye features básicas de color/textura y features de layout/localización.

    Parameters
    ----------
    config : VisualFeatureConfig
    """

    def __init__(self, config: VisualFeatureConfig) -> None:
        self.config = config

    # ─────────────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────────────

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae features de todas las imágenes y las devuelve como DataFrame.

        Parameters
        ----------
        df : pd.DataFrame con columna 'image_path'

        Returns
        -------
        pd.DataFrame con una fila por creative.
        """
        records: List[Dict] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extrayendo features visuales"):
            cid = row[self.config.creative_id_column]
            img_path = row["image_path"]
            try:
                feats = self._extract_single(img_path)
            except Exception as exc:
                logger.warning(f"Error en {cid}: {exc}")
                feats = self._empty_features()
            feats[self.config.creative_id_column] = cid
            records.append(feats)

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────
    # Extracción de una imagen
    # ─────────────────────────────────────────────────────────────────────

    def _extract_single(self, img_path: Path) -> Dict:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"cv2 no pudo leer: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        feats: Dict = {}
        feats.update(self._basic_geometry(img_rgb))
        feats.update(self._color_features(img_rgb, img_hsv))
        feats.update(self._texture_features(gray))
        feats.update(self._layout_features(img_rgb, gray, img_hsv))

        return feats

    # ─────────────────────────────────────────────────────────────────────
    # Sub-extractores
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _basic_geometry(img: np.ndarray) -> Dict:
        h, w = img.shape[:2]
        return {
            "width": w,
            "height": h,
            "aspect_ratio": w / h if h > 0 else 0.0,
            "image_area": w * h,
        }

    @staticmethod
    def _color_features(img_rgb: np.ndarray, img_hsv: np.ndarray) -> Dict:
        # Brillo sobre canal Value de HSV
        v = img_hsv[:, :, 2]
        s = img_hsv[:, :, 1]
        h = img_hsv[:, :, 0]

        # Contraste: desviación estándar del canal Value
        contrast = float(v.std())

        # Colorfulness: basado en Hasler & Süsstrunk (2003)
        # rg = R - G,  yb = 0.5*(R+G) - B
        rf = img_rgb[:, :, 0].astype(float)
        gf = img_rgb[:, :, 1].astype(float)
        bf = img_rgb[:, :, 2].astype(float)
        rg = rf - gf
        yb = 0.5 * (rf + gf) - bf
        colorfulness = float(
            np.sqrt(rg.std() ** 2 + yb.std() ** 2)
            + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        )

        # Color dominante (K-Means con K=1 simplificado: media de los píxeles más saturados)
        sat_mask = s > (s.mean() + s.std())
        if sat_mask.sum() > 10:
            dom_pixels = img_rgb[sat_mask]
        else:
            dom_pixels = img_rgb.reshape(-1, 3)

        dom_rgb = dom_pixels.mean(axis=0).astype(int).tolist()
        dom_hsv = img_hsv[sat_mask if sat_mask.sum() > 10 else np.ones_like(sat_mask, bool)].mean(axis=0).tolist()

        return {
            "brightness_mean": float(v.mean()),
            "brightness_std": float(v.std()),
            "contrast": contrast,
            "saturation_mean": float(s.mean()),
            "saturation_std": float(s.std()),
            "hue_mean": float(h.mean()),
            "hue_std": float(h.std()),
            "dominant_color_r": dom_rgb[0],
            "dominant_color_g": dom_rgb[1],
            "dominant_color_b": dom_rgb[2],
            "dominant_color_h": float(dom_hsv[0]),
            "dominant_color_s": float(dom_hsv[1]),
            "dominant_color_v": float(dom_hsv[2]),
            "colorfulness": colorfulness,
        }

    @staticmethod
    def _texture_features(gray: np.ndarray) -> Dict:
        # Densidad de bordes con Canny
        edges = cv2.Canny(gray.astype(np.uint8), threshold1=50, threshold2=150)
        edge_density = float(edges.mean() / 255.0)

        # Complejidad visual: entropía aproximada de la imagen en escala de grises
        # Usamos el histograma de 256 bins para estimar la entropía de Shannon
        hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]
        visual_complexity = float(-np.sum(hist * np.log2(hist)))

        # Sharpness: varianza del laplaciano (alta = imagen nítida)
        lap_var = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())

        return {
            "edge_density": edge_density,
            "visual_complexity": visual_complexity,
            "sharpness_laplacian_var": lap_var,
        }

    def _layout_features(
        self,
        img_rgb: np.ndarray,
        gray: np.ndarray,
        img_hsv: np.ndarray,
    ) -> Dict:
        """
        Heurísticas de distribución visual y localización de contenido.

        La "masa visual" se aproxima como la energía de bordes en cada región,
        lo que captura dónde está el contenido visualmente activo.
        """
        h, w = gray.shape
        total_px = h * w

        # ── Mapa de activación (proxy de saliencia con bordes + saturación) ──
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        sat_norm = img_hsv[:, :, 1] / 255.0
        # Combinamos bordes y saturación como proxy de "actividad visual"
        activation = 0.6 * edges + 0.4 * sat_norm

        total_activation = activation.sum() + 1e-9  # evitar div/0

        # ── Centroides de masa visual ──
        # Calculamos el centroide pesado por el mapa de activación
        ys, xs = np.mgrid[0:h, 0:w]
        cx = float((xs * activation).sum() / total_activation) / w
        cy = float((ys * activation).sum() / total_activation) / h

        # ── Ratios por región ──
        half_h, half_w = h // 2, w // 2
        center_margin_h = int(h * 0.25)
        center_margin_w = int(w * 0.25)

        top_mass = activation[:half_h, :].sum() / total_activation
        bottom_mass = activation[half_h:, :].sum() / total_activation
        left_mass = activation[:, :half_w].sum() / total_activation
        right_mass = activation[:, half_w:].sum() / total_activation

        center_region = activation[
            center_margin_h : h - center_margin_h,
            center_margin_w : w - center_margin_w,
        ]
        center_mass = center_region.sum() / total_activation
        border_mask = np.ones_like(activation, dtype=bool)
        border_mask[center_margin_h : h - center_margin_h, center_margin_w : w - center_margin_w] = False
        border_mass = activation[border_mask].sum() / total_activation

        # ── Cuadrantes ──
        q_tl = activation[:half_h, :half_w].sum() / total_activation
        q_tr = activation[:half_h, half_w:].sum() / total_activation
        q_bl = activation[half_h:, :half_w].sum() / total_activation
        q_br = activation[half_h:, half_w:].sum() / total_activation

        # ── Simetría ──
        # Diferencia normalizada entre mitades
        sym_h = 1.0 - float(
            np.abs(activation[:half_h, :] - activation[half_h : 2 * half_h, :]).mean()
        )
        sym_v = 1.0 - float(
            np.abs(activation[:, :half_w] - activation[:, half_w : 2 * half_w]).mean()
        )

        # ── Bounding box del objeto principal ──
        # Umbral adaptativo sobre el mapa de activación
        thresh = float(activation.mean() + activation.std())
        active_binary = (activation > thresh).astype(np.uint8)
        contours, _ = cv2.findContours(active_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest)
            bbox_area_ratio = float((bw * bh) / total_px)
            obj_cx = float((bx + bw / 2) / w)
            obj_cy = float((by + bh / 2) / h)
            is_centered = int(
                abs(obj_cx - 0.5) < 0.15 and abs(obj_cy - 0.5) < 0.15
            )
        else:
            bx = by = bw = bh = 0
            bbox_area_ratio = 0.0
            obj_cx = obj_cy = 0.5
            is_centered = 0

        # ── Sesgo de saliencia hacia el centro ──
        # Comparamos activación en la zona central vs periferia
        saliency_center_bias = float(center_mass - border_mass)

        # ── Uniformidad del fondo ──
        # El fondo se aproxima como los píxeles con baja activación
        bg_mask = activation < activation.mean() * 0.5
        if bg_mask.sum() > 10:
            bg_gray = gray[bg_mask]
            bg_uniformity = float(1.0 - (bg_gray.std() / 128.0))
        else:
            bg_uniformity = 0.0

        # ── Espacio vacío ──
        # Porcentaje de píxeles con baja activación (proxy de espacio en blanco)
        empty_space_ratio = float((activation < 0.05).mean())

        return {
            "centroid_x_visual_mass": cx,
            "centroid_y_visual_mass": cy,
            "visual_mass_top_ratio": float(top_mass),
            "visual_mass_bottom_ratio": float(bottom_mass),
            "visual_mass_left_ratio": float(left_mass),
            "visual_mass_right_ratio": float(right_mass),
            "visual_mass_center_ratio": float(center_mass),
            "visual_mass_border_ratio": float(border_mass),
            "quadrant_top_left_density": float(q_tl),
            "quadrant_top_right_density": float(q_tr),
            "quadrant_bottom_left_density": float(q_bl),
            "quadrant_bottom_right_density": float(q_br),
            "symmetry_horizontal_score": float(sym_h),
            "symmetry_vertical_score": float(sym_v),
            "main_object_bbox_x_min": bx / w,
            "main_object_bbox_y_min": by / h,
            "main_object_bbox_x_max": (bx + bw) / w,
            "main_object_bbox_y_max": (by + bh) / h,
            "main_object_bbox_area_ratio": bbox_area_ratio,
            "main_object_center_x": obj_cx,
            "main_object_center_y": obj_cy,
            "main_object_is_centered": is_centered,
            "saliency_center_bias": saliency_center_bias,
            "background_uniformity_score": bg_uniformity,
            "empty_space_ratio": empty_space_ratio,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Fallback para imágenes que fallan
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_features() -> Dict:
        """Devuelve NaN para todas las features cuando hay error."""
        feature_names = [
            "width", "height", "aspect_ratio", "image_area",
            "brightness_mean", "brightness_std", "contrast",
            "saturation_mean", "saturation_std", "hue_mean", "hue_std",
            "dominant_color_r", "dominant_color_g", "dominant_color_b",
            "dominant_color_h", "dominant_color_s", "dominant_color_v",
            "colorfulness", "edge_density", "visual_complexity",
            "sharpness_laplacian_var",
            "centroid_x_visual_mass", "centroid_y_visual_mass",
            "visual_mass_top_ratio", "visual_mass_bottom_ratio",
            "visual_mass_left_ratio", "visual_mass_right_ratio",
            "visual_mass_center_ratio", "visual_mass_border_ratio",
            "quadrant_top_left_density", "quadrant_top_right_density",
            "quadrant_bottom_left_density", "quadrant_bottom_right_density",
            "symmetry_horizontal_score", "symmetry_vertical_score",
            "main_object_bbox_x_min", "main_object_bbox_y_min",
            "main_object_bbox_x_max", "main_object_bbox_y_max",
            "main_object_bbox_area_ratio",
            "main_object_center_x", "main_object_center_y",
            "main_object_is_centered",
            "saliency_center_bias", "background_uniformity_score",
            "empty_space_ratio",
        ]
        return {k: float("nan") for k in feature_names}
