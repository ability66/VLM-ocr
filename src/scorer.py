from __future__ import annotations

import re
import unicodedata
from collections import Counter
from itertools import combinations

from src.schema import ModelOutput, ParsedLabel
from src.seal_utils import is_stamp_mode, primary_seal_signature

TITLE_PREFIX_PATTERN = re.compile(
    r"^(?:(图|表)\s*\d+|(?:figure|table|chart)\s*\d+)\s*[:：.．\-]?\s*",
    re.IGNORECASE,
)


def score_consensus(
    image_id: str, labels: list[ParsedLabel], model_outputs: list[ModelOutput]
) -> dict[str, float | list[str] | str]:
    del image_id
    reasons: list[str] = []

    if not labels:
        return {
            "type_agreement": 0.0,
            "caption_agreement": 0.0,
            "structure_agreement": 0.0,
            "overall_score": 0.0,
            "reasons": ["no parsable labels"],
        }

    type_agreement = _majority_ratio([label.image_type for label in labels])
    stamp_mode = is_stamp_mode(labels)
    seal_agreement = _seal_agreement(labels)
    has_seal_regions = bool(any(primary_seal_signature(label) for label in labels))

    if stamp_mode:
        caption_agreement = seal_agreement
        structure_agreement = 1.0
        overall_score = 0.3 * type_agreement + 0.7 * seal_agreement
    else:
        caption_agreement = _caption_agreement(labels)
        structure_agreement = _structure_agreement(labels)
        overall_score = (
            0.3 * type_agreement + 0.3 * caption_agreement + 0.4 * structure_agreement
        )

    if len(labels) == 1:
        reasons.append("only one parsable label")
    if type_agreement < 0.5 and len(labels) > 1:
        reasons.append("low image_type agreement")
    if caption_agreement < 0.5 and len(labels) > 1:
        reasons.append("low caption agreement")
    if not stamp_mode and structure_agreement < 0.5 and len(labels) > 1:
        reasons.append("low structured output agreement")
    if has_seal_regions and seal_agreement < 1.0 and len(labels) > 1:
        reasons.append("low seal agreement")
    if sum(1 for output in model_outputs if output.success) == 0:
        reasons.append("no models succeeded")

    return {
        "type_agreement": round(type_agreement, 4),
        "caption_agreement": round(caption_agreement, 4),
        "structure_agreement": round(structure_agreement, 4),
        "seal_agreement": round(seal_agreement, 4),
        "has_seal_regions": has_seal_regions,
        "stamp_mode": stamp_mode,
        "overall_score": round(overall_score, 4),
        "reasons": reasons,
    }


def _majority_ratio(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    majority_count = counts.most_common(1)[0][1]
    return majority_count / len(values)


def _caption_agreement(labels: list[ParsedLabel]) -> float:
    if not labels:
        return 0.0
    if len(labels) == 1:
        return 0.5
    if not _can_use_structured_caption(labels):
        captions = [label.caption for label in labels if label.caption.strip()]
        return _legacy_caption_agreement(captions)

    captions = [label.caption_structured for label in labels]
    brief_similarity = _average_pairwise(
        captions,
        lambda left, right: _field_text_similarity(left.brief, right.brief),
    )
    visual_type_match = _average_pairwise(
        captions,
        lambda left, right: _field_exact_match(left.visual_type, right.visual_type),
    )
    main_subject_similarity = _average_pairwise(
        captions,
        lambda left, right: _field_text_similarity(left.main_subject, right.main_subject),
    )
    key_visible_text_overlap = _average_pairwise(
        captions,
        lambda left, right: _list_overlap_similarity(
            left.key_visible_text, right.key_visible_text
        ),
    )
    visible_title_match = _average_pairwise(
        captions,
        lambda left, right: _title_similarity(left.visible_title, right.visible_title),
    )

    score = (
        0.2 * brief_similarity
        + 0.2 * visual_type_match
        + 0.2 * main_subject_similarity
        + 0.3 * key_visible_text_overlap
        + 0.1 * visible_title_match
    )
    return max(0.0, min(1.0, score))


def _legacy_caption_agreement(captions: list[str]) -> float:
    if not captions:
        return 0.0
    if len(captions) == 1:
        return 0.5
    return _average_pairwise(captions, _text_similarity)


def _structure_agreement(labels: list[ParsedLabel]) -> float:
    kinds = [label.structured_label.kind for label in labels]
    if not kinds:
        return 0.0

    counter = Counter(kinds)
    majority_kind, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(kinds)
    major_labels = [label for label in labels if label.structured_label.kind == majority_kind]
    contents = [label.structured_label.content.strip() for label in major_labels]

    if majority_kind == "none":
        if len(major_labels) == len(labels):
            return 0.95
        return min(1.0, 0.65 + 0.35 * majority_ratio)

    if majority_kind == "mermaid":
        prefix_score = sum("flowchart TD" in content for content in contents) / max(
            len(contents), 1
        )
        arrow_counts = [_count_mermaid_arrows(content) for content in contents if content]
        arrow_score = _range_similarity(arrow_counts)
        keyword_score = _average_pairwise(contents, _keyword_similarity)
        empty_penalty = 0.6 if any(not content for content in contents) else 1.0
        score = (
            0.4 * majority_ratio
            + 0.2 * prefix_score
            + 0.2 * arrow_score
            + 0.2 * keyword_score
        )
        return max(0.0, min(1.0, score * empty_penalty))

    if majority_kind == "table":
        shape_score = _table_shape_agreement(contents)
        keyword_score = _average_pairwise(contents, _keyword_similarity)
        empty_penalty = 0.6 if any(not content for content in contents) else 1.0
        score = 0.5 * majority_ratio + 0.25 * shape_score + 0.25 * keyword_score
        return max(0.0, min(1.0, score * empty_penalty))

    text_score = _average_pairwise(contents, _text_similarity)
    empty_penalty = 0.6 if any(not content for content in contents) else 1.0
    score = 0.6 * majority_ratio + 0.4 * text_score
    return max(0.0, min(1.0, score * empty_penalty))


def _can_use_structured_caption(labels: list[ParsedLabel]) -> bool:
    caption_structured_values = []
    for label in labels:
        value = getattr(label, "caption_structured", None)
        if value is None:
            return False
        caption_structured_values.append(value)
    return any(_has_meaningful_caption_structured(value) for value in caption_structured_values)


def _seal_agreement(labels: list[ParsedLabel]) -> float:
    seal_signatures = [primary_seal_signature(label) for label in labels]
    if not any(seal_signatures):
        return 1.0
    return _average_pairwise(seal_signatures, _signature_exact_match)


def _has_meaningful_caption_structured(value: object) -> bool:
    brief = str(getattr(value, "brief", "") or "").strip()
    visual_type = str(getattr(value, "visual_type", "") or "").strip()
    main_subject = str(getattr(value, "main_subject", "") or "").strip()
    visible_title = str(getattr(value, "visible_title", "") or "").strip()
    structure_summary = str(getattr(value, "structure_summary", "") or "").strip()
    key_visible_text = getattr(value, "key_visible_text", [])
    return bool(
        brief
        or visual_type
        or main_subject
        or visible_title
        or structure_summary
        or key_visible_text
    )


def _average_pairwise(values: list[object], similarity_fn) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.5
    scores = [similarity_fn(left, right) for left, right in combinations(values, 2)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _signature_exact_match(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 0.5
    if not left or not right:
        return 0.0
    return 1.0 if left == right else 0.0


def _text_similarity(left: str, right: str) -> float:
    left = _normalize_compare_text(left)
    right = _normalize_compare_text(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_bigrams = _char_bigrams(left)
    right_bigrams = _char_bigrams(right)
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _keyword_similarity(left: str, right: str) -> float:
    left_tokens = _keyword_tokens(left)
    right_tokens = _keyword_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _field_text_similarity(left: str, right: str) -> float:
    if not _normalize_compare_text(left) and not _normalize_compare_text(right):
        return 0.5
    if not _normalize_compare_text(left) or not _normalize_compare_text(right):
        return 0.0
    return _text_similarity(left, right)


def _field_exact_match(left: str, right: str) -> float:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left and not normalized_right:
        return 0.5
    if not normalized_left or not normalized_right:
        return 0.0
    return 1.0 if normalized_left == normalized_right else 0.0


def _title_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left and not normalized_right:
        return 0.5
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    stripped_left = _strip_title_prefix(left)
    stripped_right = _strip_title_prefix(right)
    normalized_stripped_left = _normalize_compare_text(stripped_left)
    normalized_stripped_right = _normalize_compare_text(stripped_right)
    if normalized_stripped_left and normalized_stripped_left == normalized_stripped_right:
        return 1.0

    return _phrase_similarity(stripped_left or left, stripped_right or right)


def _list_overlap_similarity(left: list[str], right: list[str]) -> float:
    left_items = _deduplicate_normalized_list(left)
    right_items = _deduplicate_normalized_list(right)
    if not left_items and not right_items:
        return 0.5
    if not left_items or not right_items:
        return 0.0

    left_scores = [_best_phrase_match(item, right_items) for item in left_items]
    right_scores = [_best_phrase_match(item, left_items) for item in right_items]
    return (sum(left_scores) / len(left_scores) + sum(right_scores) / len(right_scores)) / 2


def _best_phrase_match(item: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    return max(_phrase_similarity(item, candidate) for candidate in candidates)


def _phrase_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    compact_left = _normalize_compact_text(left)
    compact_right = _normalize_compact_text(right)
    substring_score = 0.0
    if compact_left and compact_right and (
        compact_left in compact_right or compact_right in compact_left
    ):
        ratio = min(len(compact_left), len(compact_right)) / max(
            len(compact_left), len(compact_right)
        )
        substring_score = 0.45 + 0.5 * ratio

    keyword_score = _keyword_similarity(left, right)
    char_score = _text_similarity(left, right)
    return max(substring_score, 0.6 * keyword_score + 0.4 * char_score, char_score)


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _keyword_tokens(text: str) -> set[str]:
    normalized = _normalize_compare_text(text)
    return {token.lower() for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", normalized)}


def _normalize_compare_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _normalize_compact_text(text: str) -> str:
    normalized = _normalize_compare_text(text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def _strip_title_prefix(text: str) -> str:
    return TITLE_PREFIX_PATTERN.sub("", text.strip())


def _deduplicate_normalized_list(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_compare_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


def _count_mermaid_arrows(content: str) -> int:
    patterns = ["-->", "-.->", "==>"]
    return sum(content.count(pattern) for pattern in patterns)


def _range_similarity(values: list[int]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.5
    maximum = max(values)
    minimum = min(values)
    if maximum == 0:
        return 1.0 if minimum == 0 else 0.0
    return 1.0 - ((maximum - minimum) / maximum)


def _table_shape_agreement(contents: list[str]) -> float:
    shapes = [_table_shape(content) for content in contents if content.strip()]
    if not shapes:
        return 0.0
    if len(shapes) == 1:
        return 0.5

    scores: list[float] = []
    for (rows_a, cols_a), (rows_b, cols_b) in combinations(shapes, 2):
        row_score = _ratio_similarity(rows_a, rows_b)
        col_score = _ratio_similarity(cols_a, cols_b)
        scores.append((row_score + col_score) / 2)
    return sum(scores) / len(scores)


def _table_shape(content: str) -> tuple[int, int]:
    rows = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue

        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        elif "," in stripped:
            cells = [cell.strip() for cell in stripped.split(",")]
        else:
            cells = [stripped]
        rows.append([cell for cell in cells if cell])

    if not rows:
        return (0, 0)
    return (len(rows), max(len(row) for row in rows))


def _ratio_similarity(left: int, right: int) -> float:
    if left == 0 and right == 0:
        return 1.0
    if left == 0 or right == 0:
        return 0.0
    return min(left, right) / max(left, right)
