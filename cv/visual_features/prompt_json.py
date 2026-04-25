"""Build prompt-compatible structured JSON objects per creative."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .color_features import extract_color_features
from .visual_elements import detect_visual_elements, summarize_visual_elements
from .layout_features import (
    classify_layout_pattern,
    compute_content_zones,
    compute_background_complexity,
)


CTA_KEYWORDS = {
    "install",
    "shop",
    "start",
    "watch",
    "book",
    "order",
    "claim",
    "get",
    "download",
    "learn more",
    "try",
}

PROMO_KEYWORDS = {
    "sale",
    "discount",
    "off",
    "free",
    "save",
    "limited",
    "stock",
    "%",
    "2-for-1",
    "up to",
}

PRICE_TOKENS = {"$", "EUR", "GBP", "USD", "MXN", "CAD", "BRL", "JPY"}
LEGAL_KEYWORDS = {"terms", "conditions", "privacy", "legal", "sponsored", "ad"}
SOCIAL_KEYWORDS = {"ugc", "review", "reviews", "rating", "ratings", "testimonial", "trusted", "stars"}

TEXT_TYPES = {
    "headline",
    "subheadline",
    "cta_button",
    "price_tag",
    "discount_badge",
    "brand_name",
    "social_proof",
    "legal_text",
    "other",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _clean_ocr_text(text: str) -> str:
    """Remove Florence-2 OCR artifacts like </s> prefix."""
    text = re.sub(r"</?s>", "", text)
    text = text.strip()
    return text


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = _clean_ocr_text(str(text))
    lowered = cleaned.strip().lower()
    lowered = re.sub(r"[^a-z0-9%$]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _bbox_center_and_area(bbox: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    area = width * height
    return x_center, y_center, area


def _vertical_zone(y_center: float) -> str:
    if y_center < 0.33:
        return "top"
    if y_center < 0.66:
        return "middle"
    return "bottom"


def _horizontal_zone(x_center: float) -> str:
    if x_center < 0.33:
        return "left"
    if x_center < 0.66:
        return "center"
    return "right"


def _token_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a.intersection(tokens_b)
    return len(overlap) / max(len(tokens_b), 1)


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                curr_row[j] + 1,        # insert
                prev_row[j + 1] + 1,    # delete
                prev_row[j] + cost,     # replace
            ))
        prev_row = curr_row
    return prev_row[-1]


def _fuzzy_match(a: str, b: str, threshold: float = 0.75) -> bool:
    """Check if two strings are similar enough (handles OCR typos like insertions/deletions)."""
    if not a or not b:
        return False
    if a == b:
        return True
    # Check substring containment
    if a in b or b in a:
        return True
    # Use edit distance ratio for short strings
    longer = max(len(a), len(b))
    if longer == 0:
        return False
    dist = _edit_distance(a, b)
    ratio = 1.0 - (dist / longer)
    return ratio >= threshold


def _text_matches_reference(candidate: str, reference: str) -> tuple[bool, float]:
    """Check if candidate text matches a reference, requiring substantial overlap."""
    if not candidate or not reference:
        return False, 0.0
    if candidate == reference:
        return True, 1.0
    # Exact containment: only match if the candidate is the whole reference
    # or nearly so (not a single common word in a multi-word ref)
    if reference in candidate:
        return True, 0.9
    if candidate in reference:
        # Only match if candidate covers a significant portion of reference
        ratio = len(candidate) / len(reference)
        if ratio >= 0.6:
            return True, 0.8
        return False, 0.0
    overlap = _token_overlap(candidate, reference)
    return overlap >= 0.6, overlap


def _classify_text_type(
    text_norm: str,
    text_raw: str,
    advertiser_norm: str,
    cta_norm: str,
    headline_norm: str,
    subhead_norm: str,
    y_center: float,
    area_relative: float,
) -> tuple[str, float]:
    # --- 1. BRAND NAME: check first, handles OCR typos ---
    if advertiser_norm:
        if _fuzzy_match(text_norm, advertiser_norm):
            return "brand_name", 0.92
        if _token_overlap(text_norm, advertiser_norm) >= 0.5:
            return "brand_name", 0.82

    # --- 2. CTA BUTTON: exact match against CSV ---
    is_match, score = _text_matches_reference(text_norm, cta_norm)
    if is_match:
        return "cta_button", _clamp(0.80 + 0.15 * score)

    # --- 3. HEADLINE: exact match against CSV ---
    # Check BEFORE promo keywords so "Flash sale live" matches headline, not badge
    is_match, score = _text_matches_reference(text_norm, headline_norm)
    if is_match:
        return "headline", _clamp(0.75 + 0.2 * score)

    # --- 4. SUBHEADLINE: exact match against CSV ---
    is_match, score = _text_matches_reference(text_norm, subhead_norm)
    if is_match:
        return "subheadline", _clamp(0.7 + 0.2 * score)

    # --- 5. SOCIAL PROOF: short keywords like "UGC", "review", etc. ---
    if any(keyword in text_norm for keyword in SOCIAL_KEYWORDS):
        return "social_proof", 0.88

    # --- 6. PROMO / DISCOUNT BADGE: "SALE", "50% off", etc. ---
    if any(keyword in text_norm for keyword in PROMO_KEYWORDS):
        return "discount_badge", 0.88

    # --- 7. PRICE TAG ---
    if "$" in text_norm or re.search(r"\b\d+[\.,]?\d*\s*(%|off)\b", text_norm):
        return "price_tag", 0.85
    if any(tok in text_norm for tok in {"usd", "eur", "gbp", "mxn", "cad", "brl", "jpy"}):
        return "price_tag", 0.85

    # --- 8. CTA by keywords (fallback) ---
    if len(text_norm.split()) <= 3 and any(kw in text_norm for kw in CTA_KEYWORDS):
        return "cta_button", 0.75

    # --- 9. LEGAL TEXT ---
    if any(keyword in text_norm for keyword in LEGAL_KEYWORDS) or (y_center > 0.9 and area_relative < 0.01):
        return "legal_text", 0.7

    return "other", 0.55


def _element_bbox_normalized(element: dict[str, Any], width: int, height: int) -> list[float]:
    x1 = _clamp(float(element.get("x1", 0.0)) / max(width, 1))
    y1 = _clamp(float(element.get("y1", 0.0)) / max(height, 1))
    x2 = _clamp(float(element.get("x2", 0.0)) / max(width, 1))
    y2 = _clamp(float(element.get("y2", 0.0)) / max(height, 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [_round_float(x1), _round_float(y1), _round_float(x2), _round_float(y2)]


def _build_text_elements_from_ocr(
    creative_row: dict[str, Any],
    ocr_elements: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    advertiser_norm = _normalize_text(str(creative_row.get("advertiser_name", "")))
    cta_norm = _normalize_text(str(creative_row.get("cta_text", "")))
    headline_norm = _normalize_text(str(creative_row.get("headline", "")))
    subhead_norm = _normalize_text(str(creative_row.get("subhead", "")))

    text_elements: list[dict[str, Any]] = []

    for element in ocr_elements:
        text_raw = str(element.get("label", "")).strip()
        text = _clean_ocr_text(text_raw)  # Remove </s> and other artifacts
        text_norm = _normalize_text(text)
        bbox = _element_bbox_normalized(element, width=width, height=height)
        x_center, y_center, area = _bbox_center_and_area(bbox)

        text_type, confidence = _classify_text_type(
            text_norm=text_norm,
            text_raw=text,
            advertiser_norm=advertiser_norm,
            cta_norm=cta_norm,
            headline_norm=headline_norm,
            subhead_norm=subhead_norm,
            y_center=y_center,
            area_relative=area,
        )

        if text_type not in TEXT_TYPES:
            text_type = "other"

        text_elements.append(
            {
                "text": text,
                "text_norm": text_norm,
                "type": text_type,
                "type_confidence": _round_float(confidence),
                "bbox_normalized": bbox,
                "vertical_zone": _vertical_zone(y_center),
                "horizontal_zone": _horizontal_zone(x_center),
                "area_relative": _round_float(area),
            }
        )

    text_elements.sort(key=lambda item: (item["bbox_normalized"][1], item["bbox_normalized"][0]))
    return text_elements



def _best_text_element(text_elements: list[dict[str, Any]], wanted_types: set[str]) -> dict[str, Any] | None:
    filtered = [item for item in text_elements if item.get("type") in wanted_types]
    if not filtered:
        return None
    return max(
        filtered,
        key=lambda item: (float(item.get("type_confidence", 0.0)), float(item.get("area_relative", 0.0))),
    )


def _best_text_by_keywords(text_elements: list[dict[str, Any]], keywords: set[str]) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in text_elements:
        text_norm = str(item.get("text_norm", ""))
        score = 0.0
        for keyword in keywords:
            if keyword in text_norm:
                score += 1.0
        if score > 0:
            score += 0.2 * float(item.get("area_relative", 0.0))
            scored.append((score, item))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _best_grounding_element(grounded_elements: list[dict[str, Any]], keywords: set[str]) -> dict[str, Any] | None:
    for element in grounded_elements:
        label = _normalize_text(str(element.get("label", "")))
        if any(keyword in label for keyword in keywords):
            return element
    return None


def _grounding_to_detected(element: dict[str, Any], width: int, height: int, confidence: float, text: str | None = None) -> dict[str, Any]:
    bbox = _element_bbox_normalized(element, width=width, height=height)
    detected: dict[str, Any] = {
        "bbox_normalized": bbox,
        "confidence": _round_float(confidence),
    }
    if text is not None:
        detected["text"] = text
    return detected


def _text_to_detected(text_element: dict[str, Any], confidence: float, include_text: bool = False) -> dict[str, Any]:
    detected: dict[str, Any] = {
        "bbox_normalized": list(text_element["bbox_normalized"]),
        "confidence": _round_float(confidence),
    }
    if include_text:
        detected["text"] = str(text_element.get("text", ""))
    return detected


def _build_detected_elements(
    creative_row: dict[str, Any],
    text_elements: list[dict[str, Any]],
    grounded_elements: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    brand_ground = _best_grounding_element(grounded_elements, {"logo", "brand"})
    cta_ground = _best_grounding_element(grounded_elements, {"cta", "button"})
    badge_ground = _best_grounding_element(grounded_elements, {"badge", "discount", "price", "sale"})
    product_ground = _best_grounding_element(grounded_elements, {"product", "packshot", "food", "item"})

    brand_text = _best_text_element(text_elements, {"brand_name"})
    cta_text = _best_text_element(text_elements, {"cta_button"})
    badge_text = _best_text_element(text_elements, {"discount_badge", "price_tag"})
    social_text = _best_text_element(text_elements, {"social_proof"})
    # Also check by keyword for social proof if classifier missed it
    if social_text is None:
        social_text = _best_text_by_keywords(text_elements, SOCIAL_KEYWORDS)

    # --- Brand logo: prefer OCR brand_text position, then grounding ---
    # Florence-2 grounding often labels the SALE badge as "brand logo",
    # so we check for overlap with badge_text and reject if they overlap
    brand_logo = None
    if brand_text is not None:
        brand_logo = _text_to_detected(brand_text, confidence=max(0.7, float(brand_text.get("type_confidence", 0.7))))
    elif brand_ground is not None:
        # Only use grounding if it doesn't overlap with a detected badge
        use_grounding = True
        if badge_text is not None:
            ground_bbox = _element_bbox_normalized(brand_ground, width=width, height=height)
            badge_bbox = list(badge_text["bbox_normalized"])
            overlap = _bbox_overlap(ground_bbox, badge_bbox)
            badge_area = max(1e-6, (badge_bbox[2] - badge_bbox[0]) * (badge_bbox[3] - badge_bbox[1]))
            if overlap / badge_area > 0.3:
                use_grounding = False  # Grounding is pointing at the badge, not the logo
        if use_grounding:
            brand_logo = _grounding_to_detected(brand_ground, width=width, height=height, confidence=0.84)

    primary_cta = None
    if cta_text is not None:
        primary_cta = _text_to_detected(
            cta_text,
            confidence=max(0.6, float(cta_text.get("type_confidence", 0.65))),
            include_text=True,
        )
    elif cta_ground is not None:
        primary_cta = _grounding_to_detected(
            cta_ground,
            width=width,
            height=height,
            confidence=0.76,
            text=str(creative_row.get("cta_text", "")),
        )

    promo_badge = None
    if badge_text is not None:
        promo_badge = _text_to_detected(
            badge_text,
            confidence=max(0.58, float(badge_text.get("type_confidence", 0.62))),
            include_text=True,
        )
    elif badge_ground is not None:
        promo_badge = _grounding_to_detected(badge_ground, width=width, height=height, confidence=0.74, text="")

    product_area = None
    if product_ground is not None:
        product_area = _grounding_to_detected(product_ground, width=width, height=height, confidence=0.73)
        # Discard if it covers >80% of the image — Florence-2 often returns the
        # entire image as "product" for ad creatives, which is uninformative.
        pb = product_area["bbox_normalized"]
        coverage = (pb[2] - pb[0]) * (pb[3] - pb[1])
        if coverage > 0.80:
            product_area = None

    social_proof = None
    if social_text is not None:
        social_proof = _text_to_detected(social_text, confidence=0.7, include_text=True)

    return {
        "brand_logo": brand_logo,
        "primary_cta": primary_cta,
        "promo_badge": promo_badge,
        "product_area": product_area,
        "social_proof": social_proof,
    }


def _complexity_metrics(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if arr.size == 0:
        return {
            "visual_weight_top": 0.0,
            "visual_weight_bottom": 0.0,
            "visual_active_fraction": 0.0,
            "whitespace_fraction": 1.0,
        }

    gray = np.mean(arr, axis=2)
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    edges = 0.5 * (gx + gy)
    color_var = np.std(arr, axis=2)
    complexity = 0.6 * edges + 0.4 * color_var

    total = float(np.sum(complexity)) + 1e-9
    height = complexity.shape[0]
    top_end = max(1, height // 3)
    bottom_start = max(0, (2 * height) // 3)

    top_sum = float(np.sum(complexity[:top_end, :]))
    bottom_sum = float(np.sum(complexity[bottom_start:, :]))

    threshold = float(np.percentile(complexity, 40.0))
    visual_active = float(np.mean(complexity > threshold))
    whitespace = float(np.mean(complexity <= threshold))

    return {
        "visual_weight_top": _clamp(top_sum / total),
        "visual_weight_bottom": _clamp(bottom_sum / total),
        "visual_active_fraction": _clamp(visual_active),
        "whitespace_fraction": _clamp(whitespace),
    }


def _bbox_overlap(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _get_center_from_detected(detected: dict[str, Any] | None) -> tuple[float | None, float | None, float]:
    if not detected:
        return None, None, 0.0
    bbox = detected.get("bbox_normalized")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, None, 0.0
    x_center, y_center, area = _bbox_center_and_area([float(v) for v in bbox])
    return x_center, y_center, area


def _safe_bool(value: bool) -> bool:
    return bool(value)


def _layout_pattern(features: dict[str, Any], detected_elements: dict[str, Any]) -> dict[str, Any]:
    top_count = int(features["top_zone_element_count"])
    middle_count = int(features["middle_zone_element_count"])
    bottom_count = int(features["bottom_zone_element_count"])
    cta_bottom = bool(features["cta_in_bottom_third"])
    cta_y = features["cta_y_relative"]
    headline_y = features["headline_y_relative"]
    product_coverage = float(features["product_zone_coverage"])

    product_area = detected_elements.get("product_area")
    product_area_rel = 0.0
    if product_area is not None:
        _, _, product_area_rel = _bbox_center_and_area([float(v) for v in product_area["bbox_normalized"]])

    if top_count > max(1, int(1.2 * bottom_count)) and cta_bottom:
        confidence = _clamp(0.65 + 0.15 * min(1.0, (top_count - bottom_count) / 5.0))
        return {"type": "top_heavy", "confidence": _round_float(confidence)}

    if bottom_count > max(1, int(1.2 * top_count)):
        confidence = _clamp(0.62 + 0.15 * min(1.0, (bottom_count - top_count) / 5.0))
        return {"type": "bottom_heavy", "confidence": _round_float(confidence)}

    if cta_y is not None and headline_y is not None and 0.33 <= cta_y < 0.66 and 0.33 <= headline_y < 0.66:
        confidence = _clamp(0.64 + 0.1 * (1.0 - abs(cta_y - headline_y)))
        return {"type": "centered", "confidence": _round_float(confidence)}

    if product_coverage > 0.35 and top_count > 0 and bottom_count > 0:
        confidence = _clamp(0.63 + 0.2 * min(1.0, product_coverage))
        return {"type": "split", "confidence": _round_float(confidence)}

    if product_area_rel > 0.55 and (top_count + middle_count + bottom_count) > 0:
        confidence = _clamp(0.6 + 0.3 * min(1.0, product_area_rel))
        return {"type": "full_bleed", "confidence": _round_float(confidence)}

    if cta_bottom:
        return {"type": "top_heavy", "confidence": 0.55}
    if middle_count >= top_count and middle_count >= bottom_count:
        return {"type": "centered", "confidence": 0.52}
    return {"type": "split", "confidence": 0.5}


def _design_conventions(
    features: dict[str, Any],
    detected_elements: dict[str, Any],
    aspect_ratio: float,
    whitespace_fraction: float,
    creative_row: dict[str, Any],
) -> dict[str, float]:
    cta_y = features["cta_y_relative"]
    headline_y = features["headline_y_relative"]
    cta_in_bottom = bool(features["cta_in_bottom_third"])
    badge_in_top_right = bool(features["badge_in_top_right"])

    if cta_y is None:
        thumb_zone_cta = 0.0
    else:
        if aspect_ratio >= 1.0:
            thumb_zone_cta = _clamp((float(cta_y) - 0.6) / 0.4)
        else:
            thumb_zone_cta = _clamp((float(cta_y) - 0.5) / 0.5)

    if headline_y is not None and cta_y is not None and float(headline_y) < float(cta_y):
        distance = float(cta_y) - float(headline_y)
        visual_hierarchy_score = _clamp(1.0 - abs(distance - 0.35) / 0.35)
    elif cta_in_bottom:
        visual_hierarchy_score = 0.45
    else:
        visual_hierarchy_score = 0.25

    if detected_elements.get("promo_badge") is None:
        badge_visibility_score = 0.0
    elif badge_in_top_right:
        badge_visibility_score = 1.0
    else:
        badge_y = features["badge_y_relative"]
        badge_visibility_score = 0.72 if badge_y is not None and float(badge_y) < 0.33 else 0.45

    brand_logo = detected_elements.get("brand_logo")
    if brand_logo is not None:
        _, _, logo_area = _bbox_center_and_area([float(v) for v in brand_logo["bbox_normalized"]])
        brand_clarity_score = _clamp(0.6 * float(brand_logo.get("confidence", 0.6)) + 0.4 * min(1.0, logo_area * 8.0))
    else:
        meta_brand = creative_row.get("brand_visibility_score")
        try:
            brand_clarity_score = _clamp(float(meta_brand))
        except Exception:
            brand_clarity_score = 0.0

    whitespace_balance = _clamp(1.0 - abs(float(whitespace_fraction) - 0.38) / 0.38)

    return {
        "thumb_zone_cta": _round_float(thumb_zone_cta) or 0.0,
        "visual_hierarchy_score": _round_float(visual_hierarchy_score) or 0.0,
        "badge_visibility_score": _round_float(badge_visibility_score) or 0.0,
        "brand_clarity_score": _round_float(brand_clarity_score) or 0.0,
        "whitespace_balance": _round_float(whitespace_balance) or 0.0,
    }


def _distinct_zone_counts(text_elements: list[dict[str, Any]], detected_elements: dict[str, Any]) -> tuple[int, int, int]:
    centers: set[tuple[float, float]] = set()

    for item in text_elements:
        x_center, y_center, _ = _bbox_center_and_area([float(v) for v in item["bbox_normalized"]])
        centers.add((round(x_center, 3), round(y_center, 3)))

    for value in detected_elements.values():
        if value is None:
            continue
        x_center, y_center, _ = _bbox_center_and_area([float(v) for v in value["bbox_normalized"]])
        centers.add((round(x_center, 3), round(y_center, 3)))

    top = 0
    middle = 0
    bottom = 0
    for _, y_center in centers:
        if y_center < 0.33:
            top += 1
        elif y_center < 0.66:
            middle += 1
        else:
            bottom += 1
    return top, middle, bottom


def _compute_spatial_features(
    text_elements: list[dict[str, Any]],
    detected_elements: dict[str, Any],
    image: Image.Image,
) -> dict[str, Any]:
    cta = detected_elements.get("primary_cta")
    badge = detected_elements.get("promo_badge")
    logo = detected_elements.get("brand_logo")

    headline = _best_text_element(text_elements, {"headline"})

    cta_x, cta_y, cta_area = _get_center_from_detected(cta)
    badge_x, badge_y, _ = _get_center_from_detected(badge)
    logo_x, logo_y, _ = _get_center_from_detected(logo)

    headline_y = None
    headline_area = 0.0
    if headline is not None:
        hx, hy, h_area = _bbox_center_and_area([float(v) for v in headline["bbox_normalized"]])
        _ = hx
        headline_y = hy
        headline_area = h_area

    cta_headline_vertical_distance = None
    if cta_y is not None and headline_y is not None:
        cta_headline_vertical_distance = abs(float(cta_y) - float(headline_y))

    top_count, middle_count, bottom_count = _distinct_zone_counts(text_elements, detected_elements)

    product_zone_coverage = 0.0
    product_area = detected_elements.get("product_area")
    if product_area is not None:
        product_bbox = [float(v) for v in product_area["bbox_normalized"]]
        middle_zone = [0.0, 0.33, 1.0, 0.66]
        overlap = _bbox_overlap(product_bbox, middle_zone)
        middle_zone_area = (middle_zone[2] - middle_zone[0]) * (middle_zone[3] - middle_zone[1])
        if middle_zone_area > 0:
            product_zone_coverage = _clamp(overlap / middle_zone_area)

    complexity = _complexity_metrics(image)

    text_area_total = 0.0
    for item in text_elements:
        text_area_total += float(item.get("area_relative", 0.0))
    text_area_total = _clamp(text_area_total, 0.0, 1.0)

    visual_area = max(1e-4, float(complexity["visual_active_fraction"]))
    text_to_visual_ratio = text_area_total / visual_area

    return {
        "cta_y_relative": _round_float(cta_y),
        "cta_x_relative": _round_float(cta_x),
        "cta_in_bottom_third": _safe_bool(cta_y is not None and float(cta_y) > 0.66),
        "cta_area_relative": _round_float(cta_area) or 0.0,
        "badge_in_top_right": _safe_bool(badge_x is not None and badge_y is not None and float(badge_x) > 0.6 and float(badge_y) < 0.2),
        "badge_y_relative": _round_float(badge_y),
        "badge_x_relative": _round_float(badge_x),
        "headline_y_relative": _round_float(headline_y),
        "headline_area_relative": _round_float(headline_area) or 0.0,
        "logo_in_top_left": _safe_bool(logo_x is not None and logo_y is not None and float(logo_x) < 0.4 and float(logo_y) < 0.15),
        "cta_headline_vertical_distance": _round_float(cta_headline_vertical_distance),
        "top_zone_element_count": int(top_count),
        "middle_zone_element_count": int(middle_count),
        "bottom_zone_element_count": int(bottom_count),
        "product_zone_coverage": _round_float(product_zone_coverage) or 0.0,
        "visual_weight_top": _round_float(float(complexity["visual_weight_top"])) or 0.0,
        "visual_weight_bottom": _round_float(float(complexity["visual_weight_bottom"])) or 0.0,
        "text_to_visual_ratio": _round_float(text_to_visual_ratio) or 0.0,
        "whitespace_fraction": _round_float(float(complexity["whitespace_fraction"])) or 0.0,
    }


def _clean_text_elements_for_output(text_elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for item in text_elements:
        clean.append(
            {
                "text": str(item.get("text", "")),
                "type": str(item.get("type", "other")),
                "bbox_normalized": [float(v) for v in item.get("bbox_normalized", [0.0, 0.0, 0.0, 0.0])],
                "vertical_zone": str(item.get("vertical_zone", "middle")),
                "horizontal_zone": str(item.get("horizontal_zone", "center")),
                "area_relative": float(item.get("area_relative", 0.0)),
            }
        )
    return clean


def _creative_id_from_asset(asset_file: str | None, fallback_id: Any) -> str:
    if asset_file:
        return Path(str(asset_file)).stem
    return str(fallback_id)


def build_prompt_json_object(
    creative_row: dict[str, Any],
    image: Image.Image,
    ocr_elements: list[dict[str, Any]],
    grounded_elements: list[dict[str, Any]],
) -> dict[str, Any]:
    width, height = image.size

    text_elements = _build_text_elements_from_ocr(
        creative_row=creative_row,
        ocr_elements=ocr_elements,
        width=width,
        height=height,
    )


    detected_elements = _build_detected_elements(
        creative_row=creative_row,
        text_elements=text_elements,
        grounded_elements=grounded_elements,
        width=width,
        height=height,
    )

    spatial_features = _compute_spatial_features(
        text_elements=text_elements,
        detected_elements=detected_elements,
        image=image,
    )

    layout_pattern = _layout_pattern(features=spatial_features, detected_elements=detected_elements)

    design_conventions = _design_conventions(
        features=spatial_features,
        detected_elements=detected_elements,
        aspect_ratio=float(width / max(height, 1)),
        whitespace_fraction=float(spatial_features.pop("whitespace_fraction", 0.0)),
        creative_row=creative_row,
    )

    # --- New feature extractors (OpenCV-based, no model needed) ---

    # Collect exclusion zones for visual element detection
    exclude_zones = []
    for te in text_elements:
        exclude_zones.append(te["bbox_normalized"])
    for key, det in detected_elements.items():
        if det and "bbox_normalized" in det:
            exclude_zones.append(det["bbox_normalized"])

    # Visual elements (product cards, thumbnails)
    visual_elements = detect_visual_elements(image, exclude_zones=exclude_zones)
    visual_summary = summarize_visual_elements(visual_elements)

    # Color analysis
    color_features = extract_color_features(image)

    # Layout pattern classification
    layout_classification = classify_layout_pattern(
        visual_elements=visual_elements,
        text_elements=text_elements,
        detected_elements=detected_elements,
        image_width=width,
        image_height=height,
    )

    # Content zone ratios
    content_zones = compute_content_zones(
        text_elements=text_elements,
        visual_elements=visual_elements,
        detected_elements=detected_elements,
    )

    # Background complexity
    bg_complexity = compute_background_complexity(image, exclude_zones=exclude_zones)

    creative_id = _creative_id_from_asset(creative_row.get("asset_file"), creative_row.get("creative_id"))

    output_obj = {
        "creative_id": creative_id,
        "image_dimensions": {
            "width": int(width),
            "height": int(height),
            "aspect_ratio": _round_float(width / max(height, 1)) or 0.0,
        },
        "text_elements": _clean_text_elements_for_output(text_elements),
        "detected_elements": {
            "brand_logo": detected_elements.get("brand_logo"),
            "primary_cta": detected_elements.get("primary_cta"),
            "promo_badge": detected_elements.get("promo_badge"),
            "product_area": detected_elements.get("product_area"),
            "social_proof": detected_elements.get("social_proof"),
        },
        "visual_elements": {
            "count": visual_summary["visual_element_count"],
            "card_count": visual_summary["card_count"],
            "circle_count": visual_summary["circle_count"],
            "total_area": visual_summary["visual_elements_area_total"],
            "grid_like": visual_summary["visual_elements_grid_like"],
            "center_y": visual_summary["visual_elements_center_y"],
            "elements": [
                {
                    "shape_type": ve["shape_type"],
                    "bbox_normalized": ve["bbox_normalized"],
                    "area_relative": ve["area_relative"],
                    "aspect_ratio": ve["aspect_ratio"],
                }
                for ve in visual_elements[:10]  # Cap to 10 elements
            ],
        },
        "color_features": {
            "dominant_color_name": color_features["dominant_color_name"],
            "dominant_color_hex": color_features["dominant_color_hex"],
            "dominant_color_proportion": color_features["dominant_color_proportion"],
            "background_color_name": color_features["background_color_name"],
            "background_color_hex": color_features["background_color_hex"],
            "background_is_solid": color_features["background_is_solid"],
            "saturation_mean": color_features["saturation_mean"],
            "brightness_mean": color_features["brightness_mean"],
            "color_warmth": color_features["color_warmth"],
            "contrast_score": color_features["contrast_score"],
            "palette": [
                {"hex": p["hex"], "name": p["color_name"], "proportion": p["proportion"]}
                for p in color_features["palette"][:3]  # Top 3 colors
            ],
        },
        "layout_classification": {
            "template": layout_classification["layout_template"],
            "product_card_count": layout_classification["product_card_count"],
            "pagination_dot_count": layout_classification["pagination_dot_count"],
            "has_pagination_dots": layout_classification["has_pagination_dots"],
            "has_grid_layout": layout_classification["has_grid_layout"],
            "has_carousel_indicator": layout_classification["has_carousel_indicator"],
        },
        "content_zones": content_zones,
        "background": bg_complexity,
        "spatial_features": {
            "cta_y_relative": spatial_features["cta_y_relative"],
            "cta_x_relative": spatial_features["cta_x_relative"],
            "cta_in_bottom_third": bool(spatial_features["cta_in_bottom_third"]),
            "cta_area_relative": float(spatial_features["cta_area_relative"]),
            "badge_in_top_right": bool(spatial_features["badge_in_top_right"]),
            "badge_y_relative": spatial_features["badge_y_relative"],
            "badge_x_relative": spatial_features["badge_x_relative"],
            "headline_y_relative": spatial_features["headline_y_relative"],
            "headline_area_relative": float(spatial_features["headline_area_relative"]),
            "logo_in_top_left": bool(spatial_features["logo_in_top_left"]),
            "cta_headline_vertical_distance": spatial_features["cta_headline_vertical_distance"],
            "top_zone_element_count": int(spatial_features["top_zone_element_count"]),
            "middle_zone_element_count": int(spatial_features["middle_zone_element_count"]),
            "bottom_zone_element_count": int(spatial_features["bottom_zone_element_count"]),
            "product_zone_coverage": float(spatial_features["product_zone_coverage"]),
            "visual_weight_top": float(spatial_features["visual_weight_top"]),
            "visual_weight_bottom": float(spatial_features["visual_weight_bottom"]),
            "text_to_visual_ratio": float(spatial_features["text_to_visual_ratio"]),
        },
        "layout_pattern": {
            "type": str(layout_pattern["type"]),
            "confidence": float(layout_pattern["confidence"]),
        },
        "design_conventions": {
            "thumb_zone_cta": float(design_conventions["thumb_zone_cta"]),
            "visual_hierarchy_score": float(design_conventions["visual_hierarchy_score"]),
            "badge_visibility_score": float(design_conventions["badge_visibility_score"]),
            "brand_clarity_score": float(design_conventions["brand_clarity_score"]),
            "whitespace_balance": float(design_conventions["whitespace_balance"]),
        },
    }

    return output_obj
