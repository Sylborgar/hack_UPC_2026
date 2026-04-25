"""Layout pattern classification and content zone analysis.

Classifies ad creatives into template types based on detected visual elements
and text positions, and computes vertical content zone ratios.
"""

from __future__ import annotations

from typing import Any


def classify_layout_pattern(
    visual_elements: list[dict[str, Any]],
    text_elements: list[dict[str, Any]],
    detected_elements: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """
    Classify the ad creative into a layout template type.

    Templates identified from analysis of 20+ creatives:
    - grid_2x2: 4 product cards in a 2×2 grid
    - carousel_3: 3 cards in a horizontal row
    - dual_product: 2 cards side by side
    - single_product: 1 centered card
    - hero_shapes: large circle + rectangle overlapping (app showcase)
    - minimal: no product showcase, text-focused
    """
    # Separate product cards from pagination dots
    # Pagination dots are very small circles (< 1.5% of image area)
    product_cards = []
    pagination_dots = []

    for elem in visual_elements:
        if elem["area_relative"] < 0.0025:
            # Too tiny — likely pagination dot or noise
            if elem["shape_type"] == "circle":
                pagination_dots.append(elem)
            continue
        product_cards.append(elem)

    card_count = len(product_cards)
    dot_count = len(pagination_dots)

    # Check for grid pattern (cards at similar y-coordinates)
    has_grid = False
    grid_rows = 0
    grid_cols = 0
    if card_count >= 4:
        centers_y = [e["center_normalized"][1] for e in product_cards]
        centers_x = [e["center_normalized"][0] for e in product_cards]
        # Group by y (5% buckets)
        y_buckets: dict[int, list] = {}
        for i, cy in enumerate(centers_y):
            bucket = int(cy * 20)
            y_buckets.setdefault(bucket, []).append(i)
        rows_with_multiple = [indices for indices in y_buckets.values() if len(indices) >= 2]
        if len(rows_with_multiple) >= 2:
            has_grid = True
            grid_rows = len(rows_with_multiple)
            grid_cols = max(len(r) for r in rows_with_multiple)

    # Check for horizontal row (all cards at same y)
    has_horizontal_row = False
    if card_count >= 2:
        centers_y = [e["center_normalized"][1] for e in product_cards]
        y_range = max(centers_y) - min(centers_y)
        if y_range < 0.08:  # Within 8% vertical tolerance
            has_horizontal_row = True

    # Check for hero shapes (large circle + rectangle overlapping)
    has_hero_shapes = False
    circles = [e for e in product_cards if e["shape_type"] == "circle"]
    rects = [e for e in product_cards if "card" in e["shape_type"] or e["shape_type"] == "rounded_card"]
    if len(circles) >= 1 and len(rects) >= 1:
        # Check if they overlap or are adjacent
        for circ in circles:
            for rect in rects:
                if circ["area_relative"] > 0.02 and rect["area_relative"] > 0.02:
                    # Check proximity
                    cx_diff = abs(circ["center_normalized"][0] - rect["center_normalized"][0])
                    cy_diff = abs(circ["center_normalized"][1] - rect["center_normalized"][1])
                    if cx_diff < 0.3 and cy_diff < 0.2:
                        has_hero_shapes = True
                        break

    # Classify template
    if has_grid and card_count >= 4:
        template = f"grid_{grid_rows}x{grid_cols}"
    elif has_horizontal_row and card_count >= 3:
        template = "carousel_3"
    elif has_horizontal_row and card_count == 2:
        template = "dual_product"
    elif has_hero_shapes:
        template = "hero_shapes"
    elif card_count == 1:
        template = "single_product"
    elif card_count == 0:
        template = "minimal"
    else:
        template = f"multi_product_{card_count}"

    return {
        "layout_template": template,
        "product_card_count": card_count,
        "pagination_dot_count": dot_count,
        "has_pagination_dots": dot_count >= 2,
        "has_grid_layout": has_grid,
        "has_carousel_indicator": _has_carousel_arrow(visual_elements, image_width, image_height),
    }


def _has_carousel_arrow(
    visual_elements: list[dict[str, Any]],
    width: int,
    height: int,
) -> bool:
    """Detect if there's a carousel/play arrow on the right edge of the image."""
    for elem in visual_elements:
        # Look for a triangular shape on the right side
        cx = elem["center_normalized"][0]
        area = elem["area_relative"]
        # Arrow is typically: right side (x > 0.8), small (< 3% area), triangle-like (few vertices)
        if cx > 0.8 and 0.002 < area < 0.03:
            if elem.get("vertex_count", 0) <= 5 and elem["shape_type"] in ("shape", "card_portrait"):
                return True
    return False


def compute_content_zones(
    text_elements: list[dict[str, Any]],
    visual_elements: list[dict[str, Any]],
    detected_elements: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute vertical content zone ratios.

    Standard ad layout zones:
    - Brand zone:    top area with brand name + badges
    - Showcase zone: middle area with product cards/images
    - Copy zone:     text area with headline + subheadline
    - CTA zone:      bottom area with CTA button + price

    Returns normalized heights for each zone.
    """
    # Find the vertical extent of each zone based on element positions
    brand_y = []
    showcase_y = []
    copy_y = []
    cta_y = []

    for te in text_elements:
        bbox = te.get("bbox_normalized", [0, 0, 0, 0])
        y_top, y_bot = bbox[1], bbox[3]
        text_type = te.get("type", "other")

        if text_type in ("brand_name", "social_proof"):
            brand_y.extend([y_top, y_bot])
        elif text_type in ("headline", "subheadline"):
            copy_y.extend([y_top, y_bot])
        elif text_type in ("cta_button", "price_tag"):
            cta_y.extend([y_top, y_bot])
        elif text_type == "discount_badge":
            # Badge can be in brand zone or copy zone
            if y_top < 0.3:
                brand_y.extend([y_top, y_bot])
            else:
                copy_y.extend([y_top, y_bot])

    # Product cards define the showcase zone
    for ve in visual_elements:
        if ve["area_relative"] >= 0.005:  # Not a dot
            bbox = ve["bbox_normalized"]
            showcase_y.extend([bbox[1], bbox[3]])

    # Compute zone boundaries
    brand_end = max(brand_y) if brand_y else 0.0
    showcase_start = min(showcase_y) if showcase_y else brand_end
    showcase_end = max(showcase_y) if showcase_y else brand_end
    copy_start = min(copy_y) if copy_y else showcase_end
    copy_end = max(copy_y) if copy_y else showcase_end
    cta_start = min(cta_y) if cta_y else copy_end

    # Compute zone height ratios
    brand_zone = max(0.0, brand_end)
    showcase_zone = max(0.0, showcase_end - showcase_start)
    copy_zone = max(0.0, copy_end - copy_start)
    cta_zone = max(0.0, 1.0 - cta_start) if cta_y else 0.0

    total = brand_zone + showcase_zone + copy_zone + cta_zone
    if total < 0.01:
        total = 1.0

    # Text area ratios
    total_text_area = sum(te.get("area_relative", 0) for te in text_elements)
    headline_area = sum(
        te.get("area_relative", 0) for te in text_elements
        if te.get("type") == "headline"
    )

    return {
        "brand_zone_ratio": round(brand_zone / total, 4) if total > 0 else 0.0,
        "showcase_zone_ratio": round(showcase_zone / total, 4) if total > 0 else 0.0,
        "copy_zone_ratio": round(copy_zone / total, 4) if total > 0 else 0.0,
        "cta_zone_ratio": round(cta_zone / total, 4) if total > 0 else 0.0,
        "total_text_area_ratio": round(total_text_area, 6),
        "headline_area_ratio": round(headline_area, 6),
        "headline_is_large": headline_area > 0.05,
        "vertical_center_of_mass": _vertical_center_of_mass(text_elements, visual_elements),
    }


def _vertical_center_of_mass(
    text_elements: list[dict[str, Any]],
    visual_elements: list[dict[str, Any]],
) -> float:
    """Compute the vertical center of mass of all content."""
    total_weight = 0.0
    weighted_y = 0.0

    for te in text_elements:
        area = te.get("area_relative", 0.001)
        bbox = te.get("bbox_normalized", [0, 0, 0, 0])
        cy = (bbox[1] + bbox[3]) / 2
        weighted_y += cy * area
        total_weight += area

    for ve in visual_elements:
        area = ve.get("area_relative", 0.001)
        cy = ve["center_normalized"][1]
        weighted_y += cy * area
        total_weight += area

    if total_weight < 1e-9:
        return 0.5
    return round(weighted_y / total_weight, 4)


def compute_background_complexity(
    image: "Image.Image",
    exclude_zones: list[list[float]] | None = None,
) -> dict[str, Any]:
    """
    Measure background complexity by computing edge density
    outside of known content zones.

    High density → decorative patterns/textures
    Low density → solid/gradient background
    """
    import cv2
    import numpy as np

    width, height = image.size
    gray = np.array(image.convert("L"))

    # Create mask for content zones (set to 0 = excluded)
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    if exclude_zones:
        for zone in exclude_zones:
            x1 = int(zone[0] * width)
            y1 = int(zone[1] * height)
            x2 = int(zone[2] * width)
            y2 = int(zone[3] * height)
            mask[y1:y2, x1:x2] = 0

    # Edge detection on full image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Apply mask — only count edges in background
    bg_edges = cv2.bitwise_and(edges, edges, mask=mask)
    bg_pixel_count = np.sum(mask > 0)

    if bg_pixel_count < 100:
        return {
            "background_edge_density": 0.0,
            "has_decorative_pattern": False,
        }

    edge_density = float(np.sum(bg_edges > 0)) / float(bg_pixel_count)

    return {
        "background_edge_density": round(edge_density, 6),
        "has_decorative_pattern": edge_density > 0.035,
    }
