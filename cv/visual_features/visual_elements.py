"""Detect visual elements (product cards, thumbnails, icons) using OpenCV contour analysis.

These are the rectangular/rounded shapes that appear in most ad creatives as
product showcases, app screenshots, or thumbnail grids. Florence-2 cannot
detect them reliably because they are abstract geometric shapes, not semantic objects.

Approach:
1. Edge detection (Canny) on the image
2. Contour finding with hierarchy analysis
3. Filter contours by: rectangularity, minimum size, aspect ratio
4. Cluster nearby detections to avoid duplicates
5. Classify each shape (card, circle, icon) based on geometry
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _pil_to_gray(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale numpy array."""
    return np.array(image.convert("L"))


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to RGB numpy array."""
    return np.array(image.convert("RGB"))


def _rectangularity(contour: np.ndarray) -> float:
    """How close the contour area is to its bounding rectangle area (1.0 = perfect rect)."""
    area = abs(float(np.array(cv2.contourArea(contour))))
    if area < 1:
        return 0.0
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    if rect_area < 1:
        return 0.0
    return area / rect_area


def _circularity(contour: np.ndarray) -> float:
    """How close the contour is to a circle (1.0 = perfect circle)."""
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter < 1:
        return 0.0
    return (4 * np.pi * area) / (perimeter * perimeter)


def _contour_overlaps_existing(
    bbox: tuple[int, int, int, int],
    existing: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> bool:
    """Check if a bounding box significantly overlaps with any existing detection."""
    x1, y1, x2, y2 = bbox
    area_new = (x2 - x1) * (y2 - y1)

    for item in existing:
        eb = item["bbox_pixels"]
        ex1, ey1, ex2, ey2 = eb

        ix1 = max(x1, ex1)
        iy1 = max(y1, ey1)
        ix2 = min(x2, ex2)
        iy2 = min(y2, ey2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_existing = (ex2 - ex1) * (ey2 - ey1)
        union = area_new + area_existing - intersection
        iou = intersection / max(union, 1)

        if iou > iou_threshold:
            return True

    return False


def _classify_shape(contour: np.ndarray) -> str:
    """Classify a contour as card, circle, or icon based on geometry."""
    circ = _circularity(contour)
    rect = _rectangularity(contour)

    if circ > 0.85:
        return "circle"
    if rect > 0.75:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if 0.7 <= aspect <= 1.4:
            return "card_square"
        elif aspect > 1.4:
            return "card_landscape"
        else:
            return "card_portrait"
    if rect > 0.5:
        return "rounded_card"

    return "shape"


def _overlaps_exclusion_zone(
    bbox_norm: list[float],
    exclude_zones: list[list[float]],
    overlap_threshold: float = 0.3,
) -> bool:
    """Check if a detection overlaps with a known text/UI zone.
    
    Checks from both directions: if the exclusion zone is mostly inside
    the detection, or the detection is mostly inside the exclusion zone.
    """
    x1, y1, x2, y2 = bbox_norm
    area_det = max(1e-9, (x2 - x1) * (y2 - y1))

    for ez in exclude_zones:
        ex1, ey1, ex2, ey2 = ez
        area_ez = max(1e-9, (ex2 - ex1) * (ey2 - ey1))

        ix1 = max(x1, ex1)
        iy1 = max(y1, ey1)
        ix2 = min(x2, ex2)
        iy2 = min(y2, ey2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        intersection = (ix2 - ix1) * (iy2 - iy1)
        # Either the detection is mostly covered by the exclusion zone
        # OR the exclusion zone is mostly inside the detection (= the detection IS the UI element bg)
        if intersection / area_det > overlap_threshold or intersection / area_ez > overlap_threshold:
            return True
    return False


def detect_visual_elements(
    image: Image.Image,
    min_area_fraction: float = 0.008,
    max_area_fraction: float = 0.35,
    min_dimension_px: int = 25,
    exclude_zones: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    """
    Detect rectangular/card-like visual elements in a creative image.

    Args:
        image: PIL Image (RGB)
        min_area_fraction: Minimum element area as fraction of image area
        max_area_fraction: Maximum element area as fraction of image area
        min_dimension_px: Minimum width or height in pixels
        exclude_zones: List of normalized bboxes [x1,y1,x2,y2] to exclude
                       (e.g. known text elements, CTA buttons, badges)

    Returns:
        List of detected elements with normalized bboxes and metadata.
    """
    global cv2
    import cv2 as _cv2
    cv2 = _cv2

    if exclude_zones is None:
        exclude_zones = []

    width, height = image.size
    img_area = width * height
    min_area = img_area * min_area_fraction
    max_area = img_area * max_area_fraction

    gray = _pil_to_gray(image)

    # Bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Canny edge detection with two sensitivity levels
    edges_tight = cv2.Canny(blurred, 50, 150)
    edges_loose = cv2.Canny(blurred, 30, 100)

    # Combine edge maps
    edges_combined = cv2.bitwise_or(edges_tight, edges_loose)

    # Dilate slightly to close small gaps in contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges_combined, kernel, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(
        edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        return []

    detected: list[dict[str, Any]] = []

    for i, contour in enumerate(contours):
        area = abs(float(cv2.contourArea(contour)))

        # Filter by area
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter by minimum dimension
        if w < min_dimension_px or h < min_dimension_px:
            continue

        # Filter by rectangularity — we want card-like shapes
        rect_score = _rectangularity(contour)
        circ_score = _circularity(contour)

        # Must be reasonably rectangular OR circular
        if rect_score < 0.45 and circ_score < 0.7:
            continue

        # Skip very wide/flat shapes (UI bars, buttons, text labels)
        aspect = w / max(h, 1)
        if aspect > 3.5 or aspect < 0.28:
            continue

        # Compute bounding box
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Skip if it overlaps with an existing detection
        if _contour_overlaps_existing((x1, y1, x2, y2), detected):
            continue

        # Skip if it's nearly the full image (background contour)
        coverage = (w * h) / img_area
        if coverage > max_area_fraction:
            continue

        # Normalized bbox
        bbox_norm = [
            round(x1 / width, 4),
            round(y1 / height, 4),
            round(x2 / width, 4),
            round(y2 / height, 4),
        ]

        # Skip if it overlaps a known text/UI zone
        if _overlaps_exclusion_zone(bbox_norm, exclude_zones):
            continue

        shape_type = _classify_shape(contour)

        # Approximate the contour to check if it's a clean polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertex_count = len(approx)

        detected.append({
            "bbox_pixels": (x1, y1, x2, y2),
            "bbox_normalized": bbox_norm,
            "area_relative": round(coverage, 6),
            "shape_type": shape_type,
            "rectangularity": round(rect_score, 3),
            "circularity": round(circ_score, 3),
            "vertex_count": vertex_count,
            "aspect_ratio": round(aspect, 3),
            "center_normalized": [
                round((x1 + x2) / 2 / width, 4),
                round((y1 + y2) / 2 / height, 4),
            ],
        })

    # Sort by area (largest first) for consistent ordering
    detected.sort(key=lambda d: d["area_relative"], reverse=True)

    return detected


def summarize_visual_elements(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize detected visual elements into features for the spatial fingerprint.

    Returns counts and spatial stats useful for ad layout analysis.
    """
    if not elements:
        return {
            "visual_element_count": 0,
            "card_count": 0,
            "circle_count": 0,
            "visual_elements_area_total": 0.0,
            "visual_elements_grid_like": False,
            "visual_elements_center_y": None,
        }

    cards = [e for e in elements if "card" in e["shape_type"] or e["shape_type"] == "rounded_card"]
    circles = [e for e in elements if e["shape_type"] == "circle"]

    total_area = sum(e["area_relative"] for e in elements)

    # Check if elements form a grid pattern
    # (multiple elements at similar y-coordinates)
    grid_like = False
    if len(elements) >= 2:
        centers_y = [e["center_normalized"][1] for e in elements]
        # Group by similar y (within 5% tolerance)
        rows: dict[int, int] = {}
        for cy in centers_y:
            bucket = int(cy * 20)  # 5% buckets
            rows[bucket] = rows.get(bucket, 0) + 1
        # Grid-like if at least one row has 2+ elements
        grid_like = any(count >= 2 for count in rows.values())

    avg_center_y = sum(e["center_normalized"][1] for e in elements) / len(elements)

    return {
        "visual_element_count": len(elements),
        "card_count": len(cards),
        "circle_count": len(circles),
        "visual_elements_area_total": round(total_area, 6),
        "visual_elements_grid_like": grid_like,
        "visual_elements_center_y": round(avg_center_y, 4),
    }
