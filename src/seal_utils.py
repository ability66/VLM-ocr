from __future__ import annotations

import re
import unicodedata

from src.schema import ParsedLabel

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
_GENERIC_STAMP_TERMS = {
    "公章",
    "印章",
    "红色公章",
    "红色印章",
    "红色圆形公章",
    "红色圆形印章",
    "公司公章",
    "公司印章",
}
_STRUCTURED_IMAGE_TYPES = {"flowchart", "chart", "table"}


def is_stamp_mode(labels: list[ParsedLabel]) -> bool:
    if not labels:
        return False
    if any(label.image_type in _STRUCTURED_IMAGE_TYPES for label in labels):
        return False
    return any(primary_seal_text(label) for label in labels)


def primary_seal_text(label: ParsedLabel) -> str:
    candidates = []
    for region in label.ocr_regions:
        if str(region.role or "").strip() != "seal":
            continue
        extracted = extract_primary_seal_text(str(region.text or ""))
        if not extracted:
            continue
        candidates.append(extracted)
    if candidates:
        return max(candidates, key=_seal_text_quality)

    # Structured images may expose title/body OCR regions; never treat those
    # captions as seal text fallback candidates.
    if label.image_type in _STRUCTURED_IMAGE_TYPES:
        return ""

    caption_candidate = extract_primary_seal_text(str(label.caption or ""))
    if _is_specific_seal_name(caption_candidate):
        return caption_candidate

    brief_candidate = extract_primary_seal_text(str(label.caption_structured.brief or ""))
    if _is_specific_seal_name(brief_candidate):
        return brief_candidate

    return ""


def extract_primary_seal_text(text: str) -> str:
    normalized = _normalize_display_text(text)
    if not normalized:
        return ""

    trimmed = normalized.strip(" ,;:|/\\-_.")
    if _contains_cjk(trimmed):
        trimmed = re.sub(r"^\d+", "", trimmed)
        trimmed = re.sub(r"\d+$", "", trimmed)
        cjk_tokens = [token for token in _TOKEN_PATTERN.findall(trimmed) if _contains_cjk(token)]
        if cjk_tokens:
            return max(cjk_tokens, key=_seal_text_quality)
        return trimmed

    tokens = _TOKEN_PATTERN.findall(trimmed)
    non_numeric_tokens = [token for token in tokens if not token.isdigit()]
    if non_numeric_tokens:
        return max(non_numeric_tokens, key=_seal_text_quality)
    return "" if trimmed.isdigit() else trimmed


def normalized_seal_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", "", normalized).strip().lower()
    return normalized


def primary_seal_signature(label: ParsedLabel) -> str:
    return normalized_seal_text(primary_seal_text(label))


def _normalize_display_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


def _seal_text_quality(text: str) -> tuple[int, int, int]:
    normalized = normalized_seal_text(text)
    contains_cjk = int(_contains_cjk(text))
    has_company_suffix = int(any(text.endswith(suffix) for suffix in ("公司", "有限公司", "研究院", "医院", "大学")))
    return (
        contains_cjk,
        has_company_suffix,
        len(normalized),
    )


def _is_specific_seal_name(text: str) -> bool:
    normalized = normalized_seal_text(text)
    if not normalized:
        return False
    if normalized in {normalized_seal_text(value) for value in _GENERIC_STAMP_TERMS}:
        return False
    if len(normalized) < 4:
        return False
    return _contains_cjk(text)
