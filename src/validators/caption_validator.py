from __future__ import annotations

from src.schema import ParsedLabel
from src.validators.evidence import best_evidence_match, compact_text

_GENERIC_MAIN_SUBJECTS = {
    "图表区域",
    "表格区域",
    "流程步骤",
    "文档版面",
    "界面区域",
    "关系结构",
}


def validate_caption_structured(label: ParsedLabel, evidence_texts: list[str]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    unsupported_fields: list[str] = []
    score = 1.0

    caption = str(label.caption or "").strip()
    if len(caption) > 400:
        warnings.append("caption is unusually long")
        score -= 0.15
    if len(caption) > 800:
        errors.append("caption is excessively long")
        score -= 0.2

    key_visible_text = getattr(label.caption_structured, "key_visible_text", [])
    if not isinstance(key_visible_text, list):
        errors.append("caption_structured.key_visible_text is not a list")
        score -= 0.35

    visible_title = str(label.caption_structured.visible_title or "").strip()
    if visible_title:
        best_match, best_score = best_evidence_match(visible_title, evidence_texts)
        if best_score < 0.65:
            warnings.append("caption_structured.visible_title is not supported by evidence")
            unsupported_fields.append("visible_title")
            score -= 0.2
        elif best_match is None:
            score -= 0.1

    main_subject = str(label.caption_structured.main_subject or "").strip()
    if _is_specific_subject(main_subject):
        _, best_score = best_evidence_match(main_subject, evidence_texts)
        if best_score < 0.65:
            warnings.append("caption_structured.main_subject may be too specific without visible evidence")
            unsupported_fields.append("main_subject")
            score -= 0.15

    return {
        "errors": _deduplicate(errors),
        "warnings": _deduplicate(warnings),
        "score": round(max(0.0, min(1.0, score)), 4),
        "unsupported_fields": unsupported_fields,
    }


def _is_specific_subject(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped in _GENERIC_MAIN_SUBJECTS:
        return False
    compact = compact_text(stripped)
    return len(compact) >= 4


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
