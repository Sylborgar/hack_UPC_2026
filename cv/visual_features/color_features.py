"""Color analysis features for ad creatives.

Extracts dominant color, palette, warmth, saturation, contrast,
and background type using pure OpenCV/PIL — no model downloads needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _rgb_to_hsv_single(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert a single RGB color to HSV (H: 0-360, S: 0-1, V: 0-1)."""
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    c_max = max(r_f, g_f, b_f)
    c_min = min(r_f, g_f, b_f)
    delta = c_max - c_min

    # Hue
    if delta == 0:
        h = 0.0
    elif c_max == r_f:
        h = 60.0 * (((g_f - b_f) / delta) % 6)
    elif c_max == g_f:
        h = 60.0 * (((b_f - r_f) / delta) + 2)
    else:
        h = 60.0 * (((r_f - g_f) / delta) + 4)

    # Saturation
    s = 0.0 if c_max == 0 else delta / c_max

    # Value
    v = c_max

    return h, s, v


def _classify_color_name(h: float, s: float, v: float) -> str:
    """Classify an HSV color into a human-readable name."""
    if v < 0.15:
        return "black"
    if s < 0.10 and v > 0.85:
        return "white"
    if s < 0.15:
        return "gray"

    # Classify by hue
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 160:
        return "green"
    if h < 195:
        return "teal"
    if h < 255:
        return "blue"
    if h < 290:
        return "purple"
    return "pink"


def _is_warm(h: float) -> bool:
    """Check if a hue is warm (red/orange/yellow range)."""
    return h < 70 or h >= 290


def extract_color_features(
    image: Image.Image,
    n_colors: int = 5,
    sample_size: int = 5000,
) -> dict[str, Any]:
    """
    Extract color-based features from a creative image.

    Args:
        image: PIL Image (RGB)
        n_colors: Number of dominant colors to extract
        sample_size: Pixel sample size for k-means (performance)

    Returns:
        Dictionary with color features.
    """
    img = image.convert("RGB")
    width, height = img.size

    # Downsample for performance
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)
    if len(pixels) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), size=sample_size, replace=False)
        pixels = pixels[indices]

    # K-means clustering for dominant colors
    import cv2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # Sort by cluster frequency (most common first)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_centers = centers[sorted_indices]
    sorted_proportions = counts[sorted_indices] / counts.sum()

    # Extract palette
    palette = []
    for i, (center, proportion) in enumerate(zip(sorted_centers, sorted_proportions)):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        h, s, v = _rgb_to_hsv_single(r, g, b)
        palette.append({
            "rgb": [r, g, b],
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "color_name": _classify_color_name(h, s, v),
            "proportion": round(float(proportion), 4),
            "hue": round(h, 1),
            "saturation": round(s, 3),
            "value": round(v, 3),
        })

    # Dominant color (largest cluster)
    dominant = palette[0]

    # Average saturation and value across all pixels
    hsv_img = np.array(img.convert("HSV")).reshape(-1, 3)
    sat_mean = float(np.mean(hsv_img[:, 1])) / 255.0
    val_mean = float(np.mean(hsv_img[:, 2])) / 255.0

    # Color warmth: proportion of warm pixels
    warm_count = sum(
        float(p["proportion"]) for p in palette if _is_warm(p["hue"]) and p["saturation"] > 0.1
    )

    # Background color: sample edges of the image (top/bottom/left/right strips)
    img_arr = np.array(img)
    edge_strip = max(3, min(width, height) // 20)
    edge_pixels = np.concatenate([
        img_arr[:edge_strip, :, :].reshape(-1, 3),       # top
        img_arr[-edge_strip:, :, :].reshape(-1, 3),      # bottom
        img_arr[:, :edge_strip, :].reshape(-1, 3),       # left
        img_arr[:, -edge_strip:, :].reshape(-1, 3),      # right
    ]).astype(np.float32)

    # K-means on edges for background color
    _, _, bg_centers = cv2.kmeans(
        edge_pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    bg_rgb = bg_centers[0]
    bg_r, bg_g, bg_b = int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2])
    bg_h, bg_s, bg_v = _rgb_to_hsv_single(bg_r, bg_g, bg_b)

    # Check if background is solid (low edge variance)
    bg_std = float(np.std(edge_pixels, axis=0).mean())
    bg_is_solid = bg_std < 30.0

    # Contrast: luminance difference between dominant text area and background
    # Simple approximation: std of luminance across image
    gray = np.array(img.convert("L")).flatten()
    luminance_std = float(np.std(gray))
    contrast_score = min(1.0, luminance_std / 80.0)

    return {
        "dominant_color_rgb": dominant["rgb"],
        "dominant_color_hex": dominant["hex"],
        "dominant_color_name": dominant["color_name"],
        "dominant_color_proportion": dominant["proportion"],
        "palette": palette[:n_colors],
        "saturation_mean": round(sat_mean, 4),
        "brightness_mean": round(val_mean, 4),
        "color_warmth": round(warm_count, 4),
        "background_color_rgb": [bg_r, bg_g, bg_b],
        "background_color_hex": f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}",
        "background_color_name": _classify_color_name(bg_h, bg_s, bg_v),
        "background_is_solid": bg_is_solid,
        "contrast_score": round(contrast_score, 4),
    }
