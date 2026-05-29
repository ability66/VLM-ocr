from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from src.schema import CaptionStructured, ModelOutput, OcrRegion, ParsedLabel, StructuredLabel

VALID_IMAGE_TYPES = {
    "natural_image",
    "chart",
    "table",
    "flowchart",
    "document",
    "screenshot",
    "diagram",
    "mixed",
    "unknown",
}
VALID_STRUCTURED_KINDS = {"none", "table", "mermaid", "text"}
VALID_STRUCTURED_FORMATS = {"markdown", "csv", "mermaid", "plain_text", "none"}
VALID_CAPTION_SOURCES = {"generated"}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_FLOWCHART_SHAPES = {"rectangle", "diamond", "ellipse", "rounded", "unknown"}
VALID_OCR_REGION_ROLES = {"seal", "watermark", "footer", "body", "title", "other"}
VISUAL_TYPE_SYNONYMS = {
    "流程图": "flowchart",
    "图表": "chart",
    "表格": "table",
    "文档": "document",
    "截图": "screenshot",
    "示意图": "diagram",
    "自然图像": "natural_image",
    "混合": "mixed",
    "未知": "unknown",
}
OCR_REGION_ROLE_SYNONYMS = {
    "印章": "seal",
    "公章": "seal",
    "stamp": "seal",
    "seal": "seal",
    "水印": "watermark",
    "watermark": "watermark",
    "底部文字": "footer",
    "页脚": "footer",
    "footer": "footer",
    "正文": "body",
    "body": "body",
    "标题": "title",
    "title": "title",
    "其他": "other",
    "other": "other",
}
TITLE_PATTERNS = (
    re.compile(r"^(图|表)\s*\d+", re.IGNORECASE),
    re.compile(r"^(figure|table|chart)\s*\d+", re.IGNORECASE),
)
TITLE_PREFIX_PATTERN = re.compile(
    r"^(?:(图|表)\s*\d+|(?:figure|table|chart)\s*\d+)\s*[:：.．\-]?\s*",
    re.IGNORECASE,
)


def normalize_model_output(
    model_output: ModelOutput,
) -> tuple[ModelOutput, ParsedLabel | None]:
    if not model_output.success:
        return model_output, None

    cleaned_text = _strip_code_fences(model_output.raw_text)
    json_text, recovered_truncated_json = _extract_first_json_object(cleaned_text)
    if json_text is None:
        updated = model_output.model_copy(deep=True)
        updated.error = _merge_errors(updated.error, "Failed to locate JSON object in model output")
        return updated, None

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        updated = model_output.model_copy(deep=True)
        updated.error = _merge_errors(updated.error, f"JSON parse error: {exc}")
        return updated, None

    if not isinstance(payload, dict):
        updated = model_output.model_copy(deep=True)
        updated.error = _merge_errors(updated.error, "Model output JSON is not an object")
        return updated, None

    if recovered_truncated_json:
        payload["warnings"] = _append_warning(payload.get("warnings"), "recovered_truncated_json_output")

    normalized = _normalize_payload(payload)
    updated = model_output.model_copy(deep=True)
    updated.parsed = normalized.model_dump()
    return updated, normalized


def _normalize_payload(payload: dict[str, Any]) -> ParsedLabel:
    warnings = _normalize_string_list(payload.get("warnings"))

    raw_image_type = str(payload.get("image_type", "unknown")).strip().lower()
    image_type = raw_image_type if raw_image_type in VALID_IMAGE_TYPES else "unknown"
    if image_type != raw_image_type:
        warnings.append(f"image_type normalized from '{raw_image_type or 'missing'}' to 'unknown'")

    structured_input = payload.get("structured_label")
    if not isinstance(structured_input, dict):
        structured_input = {}

    raw_kind = str(structured_input.get("kind", "none")).strip().lower()
    if not raw_kind:
        raw_kind = "none"
    kind = raw_kind if raw_kind in VALID_STRUCTURED_KINDS else "text"
    if kind != raw_kind:
        warnings.append(
            f"structured_label.kind normalized from '{raw_kind}' to '{kind}'"
        )

    raw_format = str(structured_input.get("format", "")).strip().lower()
    output_format = (
        raw_format if raw_format in VALID_STRUCTURED_FORMATS else _infer_format_from_kind(kind)
    )
    if raw_format and output_format != raw_format:
        warnings.append(
            f"structured_label.format normalized from '{raw_format}' to '{output_format}'"
        )

    caption = str(payload.get("caption", "") or "").strip()
    visible_text = _normalize_string_list(payload.get("visible_text"))
    caption_structured = _normalize_caption_structured(
        value=payload.get("caption_structured"),
        caption=caption,
        visible_text=visible_text,
        image_type=image_type,
        warnings=warnings,
    )
    flowchart_graph = _normalize_flowchart_graph(
        value=payload.get("flowchart_graph"),
        structured_kind=kind,
        structured_content=str(structured_input.get("content", "") or "").strip(),
        warnings=warnings,
    )
    ocr_regions = _normalize_ocr_regions(payload.get("ocr_regions"), warnings)
    if not caption and caption_structured.brief:
        caption = caption_structured.brief
        warnings.append("caption filled from caption_structured.brief")

    return ParsedLabel(
        image_type=image_type,
        caption=caption,
        caption_structured=caption_structured,
        structured_label=StructuredLabel(
            kind=kind,
            content=str(structured_input.get("content", "") or "").strip(),
            format=output_format,
        ),
        flowchart_graph=flowchart_graph,
        visible_text=visible_text,
        ocr_regions=ocr_regions,
        uncertainty=str(payload.get("uncertainty", "") or "").strip(),
        warnings=warnings,
    )


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _infer_format_from_kind(kind: str) -> str:
    if kind == "mermaid":
        return "mermaid"
    if kind == "table":
        return "markdown"
    if kind == "none":
        return "none"
    return "plain_text"


def _normalize_flowchart_graph(
    value: Any,
    structured_kind: str,
    structured_content: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    if value is None:
        return _derive_flowchart_graph_from_mermaid_if_needed(
            structured_kind=structured_kind,
            structured_content=structured_content,
            warnings=warnings,
        )

    if not isinstance(value, dict):
        warnings.append("flowchart_graph normalized from non-object to None")
        return _derive_flowchart_graph_from_mermaid_if_needed(
            structured_kind=structured_kind,
            structured_content=structured_content,
            warnings=warnings,
        )

    nodes_input = value.get("nodes")
    edges_input = value.get("edges")
    nodes_raw = nodes_input if isinstance(nodes_input, list) else []
    edges_raw = edges_input if isinstance(edges_input, list) else []

    raw_id_map: dict[str, str] = {}
    normalized_nodes: list[dict[str, Any]] = []
    for index, item in enumerate(nodes_raw, start=1):
        node_payload = item if isinstance(item, dict) else {}
        raw_node_id = str(node_payload.get("node_id", "") or "").strip()
        order_index = _coerce_positive_int(node_payload.get("order_index"))
        node_id = _normalize_flowchart_node_id(
            raw_value=raw_node_id,
            fallback_index=order_index or index,
        )
        if node_id is None:
            warnings.append(f"flowchart_graph node_id missing at node index {index}")
            node_id = f"N{index:03d}"
        if order_index is None:
            order_index = _extract_node_index(node_id)

        row_index = _coerce_positive_int(node_payload.get("row_index"))
        col_index = _coerce_positive_int(node_payload.get("col_index"))
        bbox_hint = _normalize_bbox_hint(node_payload.get("bbox_hint"))
        shape = _normalize_flowchart_shape(node_payload.get("shape"))
        text = str(node_payload.get("text", "") or "").strip()

        if raw_node_id:
            raw_id_map[raw_node_id] = node_id
        raw_id_map[node_id] = node_id
        if order_index is not None:
            raw_id_map[str(order_index)] = node_id

        normalized_nodes.append(
            {
                "node_id": node_id,
                "order_index": order_index,
                "row_index": row_index,
                "col_index": col_index,
                "bbox_hint": bbox_hint,
                "shape": shape,
                "text": text,
            }
        )

    normalized_edges: list[dict[str, Any]] = []
    for item in edges_raw:
        edge_payload = item if isinstance(item, dict) else {}
        source = _normalize_flowchart_edge_ref(edge_payload.get("source"), raw_id_map)
        target = _normalize_flowchart_edge_ref(edge_payload.get("target"), raw_id_map)
        if source is None or target is None:
            warnings.append("flowchart_graph edge with invalid source/target was dropped")
            continue
        normalized_edges.append(
            {
                "source": source,
                "target": target,
                "label": str(edge_payload.get("label", "") or "").strip(),
            }
        )

    graph_source = str(value.get("graph_source", "") or "").strip().lower()
    if graph_source != "mermaid_fallback":
        graph_source = "model"

    node_order_rule = str(
        value.get("node_order_rule", "top_to_bottom_left_to_right") or "top_to_bottom_left_to_right"
    ).strip()
    return {
        "node_order_rule": node_order_rule,
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "graph_source": graph_source,
        "weak_candidate": bool(value.get("weak_candidate", False) or graph_source == "mermaid_fallback"),
    }


def _derive_flowchart_graph_from_mermaid_if_needed(
    structured_kind: str,
    structured_content: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    if structured_kind != "mermaid" or not structured_content.strip():
        return None

    from src.graph_fusion import extract_weak_flowchart_graph_from_mermaid

    derived = extract_weak_flowchart_graph_from_mermaid(structured_content)
    if derived is not None:
        warnings.append("flowchart_graph missing; derived weak mermaid fallback graph")
    return derived


def _normalize_flowchart_node_id(raw_value: Any, fallback_index: int | None = None) -> str | None:
    if raw_value is None:
        raw_text = ""
    else:
        raw_text = str(raw_value).strip()

    digits = re.findall(r"\d+", raw_text)
    if digits:
        return f"N{int(digits[0]):03d}"
    if fallback_index is not None and fallback_index > 0:
        return f"N{fallback_index:03d}"
    return None


def _normalize_flowchart_edge_ref(
    raw_value: Any,
    raw_id_map: dict[str, str],
) -> str | None:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None
    if raw_text in raw_id_map:
        return raw_id_map[raw_text]
    return _normalize_flowchart_node_id(raw_text)


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_node_index(node_id: str) -> int | None:
    matches = re.findall(r"\d+", str(node_id or ""))
    if not matches:
        return None
    return int(matches[0])


def _normalize_bbox_hint(value: Any) -> list[float] | None:
    if value is None or not isinstance(value, list) or len(value) != 4:
        return None

    normalized: list[float] = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return None
        normalized.append(round(min(1.0, max(0.0, parsed)), 4))
    return normalized


def _normalize_flowchart_shape(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_FLOWCHART_SHAPES:
        return normalized
    return "unknown"


def _normalize_ocr_regions(value: Any, warnings: list[str]) -> list[OcrRegion]:
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append("ocr_regions normalized from non-list to empty list")
        return []

    normalized_regions: list[OcrRegion] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            warnings.append(f"ocr_regions item at index {index} dropped because it is not an object")
            continue

        raw_role = str(item.get("role", "other") or "").strip().lower()
        role = OCR_REGION_ROLE_SYNONYMS.get(raw_role, raw_role)
        if role not in VALID_OCR_REGION_ROLES:
            warnings.append(
                f"ocr_regions[{index}].role normalized from '{raw_role or 'missing'}' to 'other'"
            )
            role = "other"

        text = str(item.get("text", "") or "").strip()
        if not text:
            warnings.append(f"ocr_regions[{index}] dropped because text is empty")
            continue

        raw_confidence = str(item.get("confidence", "medium") or "").strip().lower()
        confidence = raw_confidence if raw_confidence in VALID_CONFIDENCE else "medium"
        if raw_confidence and confidence != raw_confidence:
            warnings.append(
                f"ocr_regions[{index}].confidence normalized from '{raw_confidence}' to 'medium'"
            )

        normalized_regions.append(
            OcrRegion(
                role=role,
                text=text,
                bbox_hint=_normalize_bbox_hint(item.get("bbox_hint")),
                confidence=confidence,
            )
        )
    return normalized_regions


def _normalize_caption_structured(
    value: Any,
    caption: str,
    visible_text: list[str],
    image_type: str,
    warnings: list[str],
) -> CaptionStructured:
    if value is None:
        payload: dict[str, Any] = {}
    elif isinstance(value, dict):
        payload = value
    else:
        payload = {}
        warnings.append("caption_structured normalized from non-object to default object")

    inferred_visible_title = _guess_visible_title(visible_text)
    brief = (
        str(payload.get("brief", "") or "").strip()
        or caption
        or inferred_visible_title
        or (visible_text[0] if visible_text else "")
    )

    raw_visual_type = str(payload.get("visual_type", "") or "").strip()
    visual_type = _normalize_visual_type(raw_visual_type=raw_visual_type, image_type=image_type)
    if raw_visual_type and visual_type != raw_visual_type:
        warnings.append(
            f"caption_structured.visual_type normalized from '{raw_visual_type}' to '{visual_type}'"
        )

    raw_key_visible_text = _normalize_string_list(payload.get("key_visible_text"))
    key_visible_text = _normalize_key_visible_text(
        raw_items=raw_key_visible_text,
        visible_text=visible_text,
    )
    if raw_key_visible_text and key_visible_text != raw_key_visible_text[:10]:
        warnings.append("caption_structured.key_visible_text aligned to visible_text")
    if len(key_visible_text) > 10:
        key_visible_text = key_visible_text[:10]
        warnings.append("caption_structured.key_visible_text truncated to 10 items")

    raw_caption_source = str(payload.get("caption_source", "generated") or "").strip().lower()
    caption_source = (
        raw_caption_source if raw_caption_source in VALID_CAPTION_SOURCES else "generated"
    )
    if raw_caption_source and caption_source != raw_caption_source:
        warnings.append(
            "caption_structured.caption_source normalized "
            f"from '{raw_caption_source}' to 'generated'"
        )

    raw_confidence = str(payload.get("confidence", "medium") or "").strip().lower()
    confidence = raw_confidence if raw_confidence in VALID_CONFIDENCE else "medium"
    if raw_confidence and confidence != raw_confidence:
        warnings.append(
            f"caption_structured.confidence normalized from '{raw_confidence}' to 'medium'"
        )

    return CaptionStructured(
        brief=brief,
        visual_type=visual_type,
        main_subject=str(payload.get("main_subject", "") or "").strip(),
        visible_title=_normalize_visible_title(
            raw_visible_title=str(payload.get("visible_title", "") or "").strip(),
            visible_text=visible_text,
            inferred_visible_title=inferred_visible_title,
        ),
        key_visible_text=key_visible_text,
        structure_summary=(
            str(payload.get("structure_summary", "") or "").strip()
            or _default_structure_summary(image_type=image_type, visible_text=visible_text)
        ),
        caption_source=caption_source,
        confidence=confidence,
    )


def _normalize_visual_type(raw_visual_type: str, image_type: str) -> str:
    normalized = raw_visual_type.strip()
    if not normalized:
        return image_type

    lowered = normalized.lower()
    if lowered in VALID_IMAGE_TYPES:
        return lowered
    if normalized in VISUAL_TYPE_SYNONYMS:
        return VISUAL_TYPE_SYNONYMS[normalized]
    if "flow" in lowered or "流程" in normalized:
        return "flowchart"
    if "chart" in lowered or "plot" in lowered or "graph" in lowered or "图表" in normalized:
        return "chart"
    if "table" in lowered or "表格" in normalized:
        return "table"
    if "screen" in lowered or "screenshot" in lowered or "截图" in normalized:
        return "screenshot"
    if "document" in lowered or "doc" in lowered or "文档" in normalized:
        return "document"
    if "diagram" in lowered or "示意" in normalized:
        return "diagram"
    if "natural" in lowered or "photo" in lowered or "scene" in lowered or "自然" in normalized:
        return "natural_image"
    if "mixed" in lowered or "混合" in normalized:
        return "mixed"
    return image_type


def _guess_visible_title(visible_text: list[str]) -> str:
    title_candidates = _extract_title_candidates(visible_text)
    if title_candidates:
        return title_candidates[0]
    return ""


def _normalize_visible_title(
    raw_visible_title: str,
    visible_text: list[str],
    inferred_visible_title: str,
) -> str:
    title_candidates = _extract_title_candidates(visible_text)
    preferred_candidates = _prefer_chinese_titles(title_candidates)

    if preferred_candidates:
        if raw_visible_title:
            aligned = _best_title_candidate(raw_visible_title, preferred_candidates)
            if aligned is not None:
                return aligned
            if _contains_cjk(raw_visible_title):
                return raw_visible_title
            return preferred_candidates[0]
        return preferred_candidates[0]

    if raw_visible_title:
        aligned = _best_title_candidate(raw_visible_title, visible_text)
        if aligned is not None:
            return aligned
        return raw_visible_title

    return inferred_visible_title


def _normalize_key_visible_text(
    raw_items: list[str],
    visible_text: list[str],
) -> list[str]:
    if not raw_items:
        return visible_text[:10]

    aligned_items: list[tuple[str, int | None]] = []
    for item in raw_items:
        aligned_items.append(_align_key_visible_text_item(item=item, visible_text=visible_text))

    seen: set[str] = set()
    deduped: list[tuple[str, int | None]] = []
    for text, index in aligned_items:
        normalized = _normalize_compare_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((text, index))

    if len(deduped) < 3:
        for index, candidate in enumerate(visible_text):
            normalized = _normalize_compare_text(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append((candidate, index))
            if len(deduped) >= 3:
                break

    ordered = _sort_by_visible_text_order(deduped)
    return [text for text, _ in ordered[:10]]


def _align_key_visible_text_item(
    item: str,
    visible_text: list[str],
) -> tuple[str, int | None]:
    normalized_item = item.strip()
    if not normalized_item or not visible_text:
        return normalized_item, None

    best_index: int | None = None
    best_candidate = normalized_item
    best_score = 0.0
    for index, candidate in enumerate(visible_text):
        score = _visible_text_alignment_score(normalized_item, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_index = index

    if best_index is not None and best_score >= 0.5:
        return best_candidate, best_index
    return normalized_item, None


def _sort_by_visible_text_order(
    items: list[tuple[str, int | None]],
) -> list[tuple[str, int | None]]:
    indexed = [(text, index) for text, index in items if index is not None]
    extras = [(text, index) for text, index in items if index is None]
    indexed.sort(key=lambda value: value[1])
    return indexed + extras


def _extract_title_candidates(visible_text: list[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in visible_text:
        text = item.strip()
        if not text:
            continue
        if re.search(r"\.(png|jpg|jpeg|webp|bmp)$", text, re.IGNORECASE):
            continue
        if not any(pattern.match(text) for pattern in TITLE_PATTERNS):
            continue
        normalized = _normalize_compare_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(text)
    return candidates


def _prefer_chinese_titles(title_candidates: list[str]) -> list[str]:
    chinese = [candidate for candidate in title_candidates if _contains_cjk(candidate)]
    return chinese or title_candidates


def _best_title_candidate(raw_visible_title: str, candidates: list[str]) -> str | None:
    normalized_raw = raw_visible_title.strip()
    if not normalized_raw:
        return None

    best_candidate: str | None = None
    best_score = 0.0
    for candidate in candidates:
        score = _title_alignment_score(normalized_raw, candidate)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate is not None and best_score >= 0.55:
        return best_candidate
    return None


def _title_alignment_score(left: str, right: str) -> float:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    stripped_left = _strip_title_prefix(left)
    stripped_right = _strip_title_prefix(right)
    if stripped_left and stripped_right and stripped_left == stripped_right:
        return 1.0

    compact_left = _normalize_compact_text(stripped_left or left)
    compact_right = _normalize_compact_text(stripped_right or right)
    if not compact_left or not compact_right:
        return 0.0
    if compact_left in compact_right or compact_right in compact_left:
        ratio = min(len(compact_left), len(compact_right)) / max(
            len(compact_left), len(compact_right)
        )
        return 0.75 + 0.25 * ratio

    return max(
        _keyword_overlap(stripped_left or left, stripped_right or right),
        _char_similarity(stripped_left or left, stripped_right or right),
    )


def _visible_text_alignment_score(left: str, right: str) -> float:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    compact_left = _normalize_compact_text(left)
    compact_right = _normalize_compact_text(right)
    if compact_left and compact_right and (
        compact_left in compact_right or compact_right in compact_left
    ):
        ratio = min(len(compact_left), len(compact_right)) / max(
            len(compact_left), len(compact_right)
        )
        return 0.45 + 0.5 * ratio

    return max(
        0.6 * _keyword_overlap(left, right) + 0.4 * _char_similarity(left, right),
        _char_similarity(left, right),
    )


def _strip_title_prefix(text: str) -> str:
    return TITLE_PREFIX_PATTERN.sub("", text.strip())


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _normalize_compare_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _normalize_compact_text(text: str) -> str:
    normalized = _normalize_compare_text(text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def _keyword_overlap(left: str, right: str) -> float:
    left_tokens = _keyword_tokens(left)
    right_tokens = _keyword_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _char_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    left_bigrams = _char_bigrams(normalized_left)
    right_bigrams = _char_bigrams(normalized_right)
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _keyword_tokens(text: str) -> set[str]:
    normalized = _normalize_compare_text(text)
    return {token for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", normalized) if token}


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _default_structure_summary(image_type: str, visible_text: list[str]) -> str:
    mapping = {
        "natural_image": "可见主体与场景内容。",
        "chart": "可见图表结构与数据标注。",
        "table": "可见表格结构与单元格内容。",
        "flowchart": "可见流程节点、连线与分支关系。",
        "document": "可见文档版面与文字块结构。",
        "screenshot": "可见界面区域与控件布局。",
        "diagram": "可见示意结构与关系标注。",
        "mixed": "可见图文混合内容与局部结构。",
        "unknown": "可见图像内容，但结构类型不稳定。",
    }
    summary = mapping.get(image_type, mapping["unknown"])
    if visible_text:
        return f"{summary} 可见文字 {min(len(visible_text), 10)} 项以内。"
    return summary


def _strip_code_fences(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> tuple[str | None, bool]:
    start = text.find("{")
    if start == -1:
        return None, False

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], False

    repaired = _recover_truncated_json_object(text[start:])
    if repaired is not None:
        return repaired, True
    return None, False


def _recover_truncated_json_object(text: str) -> str | None:
    cut_positions = _json_repair_cut_positions(text)
    for cut_position in sorted(cut_positions, reverse=True):
        prefix = text[:cut_position].rstrip()
        if not prefix or prefix == "{":
            continue
        candidate = _close_truncated_json(prefix)
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def _json_repair_cut_positions(text: str) -> set[int]:
    positions = {len(text)}
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == ",":
            positions.add(index)
        elif char in "}]":
            positions.add(index + 1)
    return positions


def _close_truncated_json(prefix: str) -> str | None:
    candidate = prefix.rstrip()
    if not candidate:
        return None

    for _ in range(5):
        stack, in_string, escape = _scan_json_state(candidate)
        if escape and candidate:
            candidate = candidate[:-1].rstrip()
            continue
        if in_string:
            candidate = f'{candidate}"'

        candidate = candidate.rstrip()
        while candidate and candidate[-1] in ",:":
            candidate = candidate[:-1].rstrip()

        stack, in_string, escape = _scan_json_state(candidate)
        if escape and candidate:
            candidate = candidate[:-1].rstrip()
            continue
        if in_string:
            candidate = f'{candidate}"'
        candidate = candidate.rstrip()
        while candidate and candidate[-1] in ",:":
            candidate = candidate[:-1].rstrip()

        if not candidate:
            return None

        stack, _, _ = _scan_json_state(candidate)
        closing_chars = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        return candidate + closing_chars
    return None


def _scan_json_state(text: str) -> tuple[list[str], bool, bool]:
    stack: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()
    return stack, in_string, escape


def _append_warning(value: Any, warning: str) -> list[str]:
    warnings = _normalize_string_list(value)
    warnings.append(warning)
    return warnings


def _merge_errors(existing: str | None, new_error: str) -> str:
    if existing:
        return f"{existing}; {new_error}"
    return new_error
