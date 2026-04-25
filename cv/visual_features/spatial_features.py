"""Spatial feature extraction logic for Florence outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from PIL import Image

from .florence import FlorenceRunner


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

PRICE_TOKENS = {"$", "€", "£", "¥", "usd", "eur", "mxn", "cad", "brl", "jpy"}

LOGO_LABEL_KEYWORDS = {"logo", "brand logo", "brand"}
PRODUCT_LABEL_KEYWORDS = {"product", "packshot", "item", "food", "product image", "product placeholder"}
BADGE_LABEL_KEYWORDS = {"badge", "discount label", "sale badge", "price tag"}


@dataclass
class ExtractionArtifacts:
    ocr_raw: dict[str, Any]
    grounding_raw: dict[str, Any]
    ocr_elements: list[dict[str, Any]]
    grounding_elements: list[dict[str, Any]]


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9%$€£¥]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _coerce_number(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_bbox(coords: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(coords, (list, tuple)):
        return None

    values: list[float] = []
    for item in coords:
        number = _coerce_number(item)
        if number is None:
            return None
        values.append(number)

    if len(values) == 4:
        x1, y1, x2, y2 = values
    elif len(values) >= 8:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        return None

    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _compute_region_fields(
    label: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    x_rel = x_center / width
    y_rel = y_center / height
    w_rel = bw / width
    h_rel = bh / height
    area_rel = (bw * bh) / float(width * height)

    vertical_zone = "top" if y_rel < 1 / 3 else "middle" if y_rel < 2 / 3 else "bottom"
    horizontal_zone = "left" if x_rel < 1 / 3 else "center" if x_rel < 2 / 3 else "right"

    return {
        "label": label,
        "label_norm": normalize_text(label),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x_rel": x_rel,
        "y_rel": y_rel,
        "w_rel": w_rel,
        "h_rel": h_rel,
        "area_rel": area_rel,
        "zone_vertical": vertical_zone,
        "zone_horizontal": horizontal_zone,
        "zone_3x3": f"{vertical_zone}_{horizontal_zone}",
    }


def _unwrap_task_payload(raw: Any) -> Any:
    if isinstance(raw, dict):
        if len(raw) == 1:
            only_key = next(iter(raw))
            if isinstance(only_key, str) and only_key.startswith("<"):
                return raw[only_key]
    return raw


def _extract_list_by_keys(payload: dict[str, Any], keys: set[str]) -> Any:
    lower_to_original = {str(k).lower(): k for k in payload.keys()}
    for wanted in keys:
        if wanted in lower_to_original:
            return payload[lower_to_original[wanted]]
    return None


def parse_ocr_regions(raw_output: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    payload = _unwrap_task_payload(raw_output)
    elements: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        quad_boxes = _extract_list_by_keys(payload, {"quad_boxes", "quad_box", "polygons", "boxes", "bboxes"})
        labels = _extract_list_by_keys(payload, {"labels", "texts", "text", "words"})

        if isinstance(quad_boxes, list) and isinstance(labels, list):
            for idx, box in enumerate(quad_boxes):
                label = str(labels[idx]) if idx < len(labels) else ""
                bbox = _coerce_bbox(box, width=width, height=height)
                if bbox is None:
                    continue
                elements.append(_compute_region_fields(label=label, bbox=bbox, width=width, height=height))

    if not elements and isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", item.get("text", "")))
            raw_box = item.get("quad_box", item.get("quad_boxes", item.get("bbox", item.get("box"))))
            bbox = _coerce_bbox(raw_box, width=width, height=height)
            if bbox is None:
                continue
            elements.append(_compute_region_fields(label=label, bbox=bbox, width=width, height=height))

    return elements


def parse_grounding_regions(raw_output: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    payload = _unwrap_task_payload(raw_output)
    elements: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        boxes = _extract_list_by_keys(payload, {"bboxes", "bbox", "boxes", "quad_boxes"})
        labels = _extract_list_by_keys(payload, {"labels", "phrases", "text", "texts"})

        if isinstance(boxes, list):
            for idx, box in enumerate(boxes):
                label = ""
                if isinstance(labels, list) and idx < len(labels):
                    label = str(labels[idx])
                bbox = _coerce_bbox(box, width=width, height=height)
                if bbox is None:
                    continue
                elements.append(_compute_region_fields(label=label, bbox=bbox, width=width, height=height))

    if not elements and isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", item.get("phrase", "")))
            raw_box = item.get("bbox", item.get("box", item.get("bboxes", item.get("quad_box"))))
            bbox = _coerce_bbox(raw_box, width=width, height=height)
            if bbox is None:
                continue
            elements.append(_compute_region_fields(label=label, bbox=bbox, width=width, height=height))

    return elements


def _token_overlap_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0

    overlap = tokens_a.intersection(tokens_b)
    return len(overlap) / max(len(tokens_b), 1)


def find_best_text_match(elements: list[dict[str, Any]], reference_text: str | None) -> dict[str, Any] | None:
    target = normalize_text(reference_text)
    if not target:
        return None

    best_item = None
    best_score = -math.inf

    for item in elements:
        label = item.get("label_norm", "")
        score = 0.0

        if label == target:
            score += 4.0
        elif target and target in label:
            score += 2.5
        elif label and label in target:
            score += 2.0

        score += 2.0 * _token_overlap_score(label, target)
        score += 0.1 * item.get("area_rel", 0.0)

        if score > best_score:
            best_item = item
            best_score = score

    if best_score < 1.0:
        return None

    return best_item


def find_text_by_keywords(elements: list[dict[str, Any]], keywords: set[str]) -> dict[str, Any] | None:
    best_item = None
    best_score = -math.inf

    for item in elements:
        label = item.get("label_norm", "")
        score = 0.0
        for keyword in keywords:
            if keyword in label:
                score += 1.0

        if score <= 0:
            continue

        score += 0.2 * item.get("area_rel", 0.0)

        if score > best_score:
            best_item = item
            best_score = score

    return best_item


def find_largest_text_element(elements: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not elements:
        return None
    return max(elements, key=lambda item: item.get("area_rel", 0.0))


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _find_grounding_by_keywords(elements: list[dict[str, Any]], keywords: set[str]) -> dict[str, Any] | None:
    for element in elements:
        label = element.get("label_norm", "")
        if _contains_any(label, keywords):
            return element
    return None


def _count_zone(elements: list[dict[str, Any]], zone: str) -> int:
    return sum(1 for item in elements if item.get("zone_vertical") == zone)


def _safe_rel_value(item: dict[str, Any] | None, key: str) -> float | None:
    if not item:
        return None
    value = item.get(key)
    return float(value) if value is not None else None


def _as_int_bool(value: bool) -> int:
    return int(bool(value))


def _metadata_flag(creative_row: dict[str, Any], key: str) -> bool:
    value = creative_row.get(key)
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(int(value))

    value_str = str(value).strip().lower()
    return value_str in {"1", "true", "yes", "y"}


def _metadata_float(creative_row: dict[str, Any], key: str) -> float | None:
    value = creative_row.get(key)
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _estimate_center_product_coverage(
    grounded_elements: list[dict[str, Any]],
    width: int,
    height: int,
) -> float:
    center = (0.25 * width, 0.33 * height, 0.75 * width, 0.66 * height)
    center_area = (center[2] - center[0]) * (center[3] - center[1])
    if center_area <= 0:
        return 0.0

    total_overlap = 0.0
    for item in grounded_elements:
        label = item.get("label_norm", "")
        if not _contains_any(label, PRODUCT_LABEL_KEYWORDS):
            continue

        bbox = (item["x1"], item["y1"], item["x2"], item["y2"])
        total_overlap += _intersection_area(bbox, center)

    return min(1.0, total_overlap / center_area)


def _has_promo_text(elements: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    has_discount = False
    has_price = False
    has_promo = False

    for item in elements:
        label = item.get("label_norm", "")
        if not label:
            continue

        if any(tok in label for tok in PRICE_TOKENS):
            has_price = True
            has_promo = True

        if _contains_any(label, PROMO_KEYWORDS):
            has_discount = True
            has_promo = True

    return has_promo, has_discount, has_price


def compute_spatial_features(
    creative_row: dict[str, Any],
    image: Image.Image,
    ocr_elements: list[dict[str, Any]],
    grounded_elements: list[dict[str, Any]],
    source_mode: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    width, height = image.size

    cta_ref = creative_row.get("cta_text")
    headline_ref = creative_row.get("headline")
    subhead_ref = creative_row.get("subhead")

    cta = find_best_text_match(ocr_elements, cta_ref)
    if cta is None:
        cta = find_text_by_keywords(ocr_elements, CTA_KEYWORDS)

    headline = find_best_text_match(ocr_elements, headline_ref)
    if headline is None:
        headline = find_largest_text_element(ocr_elements)

    subhead = find_best_text_match(ocr_elements, subhead_ref)

    badge_by_text = find_text_by_keywords(ocr_elements, PROMO_KEYWORDS.union(PRICE_TOKENS))
    badge_by_grounding = _find_grounding_by_keywords(grounded_elements, BADGE_LABEL_KEYWORDS)
    badge = badge_by_text if badge_by_text is not None else badge_by_grounding

    logo = _find_grounding_by_keywords(grounded_elements, LOGO_LABEL_KEYWORDS)

    text_area_total = sum(item.get("area_rel", 0.0) for item in ocr_elements)
    avg_text_area = text_area_total / len(ocr_elements) if ocr_elements else 0.0

    has_promo_text, has_discount_text, has_price_text = _has_promo_text(ocr_elements)

    meta_has_discount_badge = _metadata_flag(creative_row, "has_discount_badge")
    meta_has_price = _metadata_flag(creative_row, "has_price")
    meta_brand_visibility = _metadata_float(creative_row, "brand_visibility_score")
    meta_text_density = _metadata_float(creative_row, "text_density")
    meta_product_count = _metadata_float(creative_row, "product_count")

    if not ocr_elements and meta_text_density is not None:
        text_area_total = max(0.0, min(1.0, meta_text_density))
        avg_text_area = text_area_total

    top_count = _count_zone(ocr_elements, "top")
    middle_count = _count_zone(ocr_elements, "middle")
    bottom_count = _count_zone(ocr_elements, "bottom")

    bottom_ratio = 0.0
    if ocr_elements:
        bottom_ratio = sum(1 for item in ocr_elements if item["y_rel"] > 0.66) / len(ocr_elements)

    cta_y = _safe_rel_value(cta, "y_rel")
    headline_y = _safe_rel_value(headline, "y_rel")

    cta_headline_distance = None
    if cta_y is not None and headline_y is not None:
        cta_headline_distance = abs(cta_y - headline_y)

    badge_x = _safe_rel_value(badge, "x_rel")
    badge_y = _safe_rel_value(badge, "y_rel")

    product_coverage = _estimate_center_product_coverage(grounded_elements, width=width, height=height)
    if not grounded_elements and meta_product_count is not None:
        product_coverage = max(0.0, min(1.0, meta_product_count / 3.0))

    cta_found_flag = (cta is not None) or bool(str(cta_ref).strip())
    headline_found_flag = (headline is not None) or bool(str(headline_ref).strip())
    subhead_found_flag = (subhead is not None) or bool(str(subhead_ref).strip())

    has_price_combined = has_price_text or meta_has_price
    has_discount_combined = has_discount_text or meta_has_discount_badge
    has_promo_combined = has_promo_text or has_price_combined or has_discount_combined

    badge_detected_flag = (badge is not None) or has_promo_combined
    logo_detected_flag = (logo is not None) or (
        meta_brand_visibility is not None and meta_brand_visibility >= 0.5
    )

    logo_in_top_left_flag = logo is not None and logo.get("x_rel", 1.0) < 0.4 and logo.get("y_rel", 1.0) < 0.2

    spatial_signal = 0
    for present in [
        cta_found_flag,
        headline_found_flag,
        badge_detected_flag,
        logo_detected_flag,
        (len(ocr_elements) > 0) or (meta_text_density is not None and meta_text_density > 0),
        (len(grounded_elements) > 0) or (meta_product_count is not None and meta_product_count > 0),
    ]:
        spatial_signal += int(present)

    return {
        "creative_id": int(creative_row["creative_id"]),
        "asset_file": creative_row.get("asset_file"),
        "vertical": creative_row.get("vertical"),
        "format": creative_row.get("format"),
        "advertiser_name": creative_row.get("advertiser_name"),
        "source_mode": source_mode,
        "extraction_error": error_message,
        "image_width": width,
        "image_height": height,
        "image_aspect_ratio": width / height if height else None,
        "ocr_element_count": len(ocr_elements),
        "grounded_element_count": len(grounded_elements),
        "text_area_total_rel": text_area_total,
        "avg_text_area_rel": avg_text_area,
        "top_zone_element_count": top_count,
        "middle_zone_element_count": middle_count,
        "bottom_zone_element_count": bottom_count,
        "bottom_text_ratio": bottom_ratio,
        "cta_found": _as_int_bool(cta_found_flag),
        "cta_y_relative": cta_y,
        "cta_x_relative": _safe_rel_value(cta, "x_rel"),
        "cta_area_relative": _safe_rel_value(cta, "area_rel") or 0.0,
        "cta_in_bottom_third": _as_int_bool(cta_y is not None and cta_y > 0.66),
        "headline_found": _as_int_bool(headline_found_flag),
        "headline_y_relative": headline_y,
        "headline_area_relative": _safe_rel_value(headline, "area_rel") or 0.0,
        "subhead_found": _as_int_bool(subhead_found_flag),
        "subhead_y_relative": _safe_rel_value(subhead, "y_rel"),
        "cta_headline_distance": cta_headline_distance,
        "has_promo_text": _as_int_bool(has_promo_combined),
        "has_discount_text": _as_int_bool(has_discount_combined),
        "has_price_text": _as_int_bool(has_price_combined),
        "has_promo_badge": _as_int_bool(badge_detected_flag),
        "badge_in_top_right": _as_int_bool(badge_x is not None and badge_y is not None and badge_x > 0.6 and badge_y < 0.2),
        "badge_y_relative": badge_y,
        "logo_detected": _as_int_bool(logo_detected_flag),
        "logo_in_top_left": _as_int_bool(logo_in_top_left_flag),
        "product_zone_coverage": product_coverage,
        "text_density_meta": meta_text_density,
        "brand_visibility_meta": meta_brand_visibility,
        "has_discount_badge_meta": _as_int_bool(meta_has_discount_badge),
        "has_price_meta": _as_int_bool(meta_has_price),
        "spatial_signal_score": spatial_signal / 6.0,
    }


def extract_spatial_fingerprint(
    creative_row: dict[str, Any],
    image: Image.Image,
    florence_runner: FlorenceRunner | None,
    grounding_text: str,
) -> tuple[dict[str, Any], ExtractionArtifacts]:
    source_mode = "florence" if florence_runner is not None else "metadata_fallback"

    if florence_runner is None:
        ocr_raw: dict[str, Any] = {}
        grounding_raw: dict[str, Any] = {}
        ocr_elements: list[dict[str, Any]] = []
        grounding_elements: list[dict[str, Any]] = []
        features = compute_spatial_features(
            creative_row=creative_row,
            image=image,
            ocr_elements=ocr_elements,
            grounded_elements=grounding_elements,
            source_mode=source_mode,
            error_message=None,
        )
        return features, ExtractionArtifacts(ocr_raw, grounding_raw, ocr_elements, grounding_elements)

    ocr_raw: dict[str, Any]
    grounding_raw: dict[str, Any]
    ocr_elements: list[dict[str, Any]]
    grounding_elements: list[dict[str, Any]]

    try:
        ocr_raw = florence_runner.run_ocr_with_region(image)
        grounding_raw = florence_runner.run_phrase_grounding(image, text=grounding_text)

        ocr_elements = parse_ocr_regions(ocr_raw, width=image.width, height=image.height)
        grounding_elements = parse_grounding_regions(grounding_raw, width=image.width, height=image.height)

        task_errors: list[str] = []
        if isinstance(ocr_raw, dict) and ocr_raw.get("error"):
            task_errors.append(f"ocr={ocr_raw['error']}")
        if isinstance(grounding_raw, dict) and grounding_raw.get("error"):
            task_errors.append(f"grounding={grounding_raw['error']}")
        error_message = " | ".join(task_errors) if task_errors else None

        features = compute_spatial_features(
            creative_row=creative_row,
            image=image,
            ocr_elements=ocr_elements,
            grounded_elements=grounding_elements,
            source_mode=source_mode,
            error_message=error_message,
        )
        return features, ExtractionArtifacts(ocr_raw, grounding_raw, ocr_elements, grounding_elements)

    except Exception as exc:
        features = compute_spatial_features(
            creative_row=creative_row,
            image=image,
            ocr_elements=[],
            grounded_elements=[],
            source_mode=source_mode,
            error_message=str(exc),
        )
        return features, ExtractionArtifacts({}, {}, [], [])
