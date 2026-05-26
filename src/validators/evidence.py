from __future__ import annotations

import re
import unicodedata

from src.schema import ParsedLabel
from src.validators.mermaid_validator import extract_mermaid_node_texts
from src.validators.table_validator import parse_markdown_table

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")


def collect_evidence_texts(labels: list[ParsedLabel]) -> list[str]:
    evidence_texts: list[str] = []
    for label in labels:
        evidence_texts.extend(label.visible_text)
        evidence_texts.extend(label.caption_structured.key_visible_text)
        if label.caption_structured.visible_title.strip():
            evidence_texts.append(label.caption_structured.visible_title.strip())
        if label.caption_structured.main_subject.strip():
            evidence_texts.append(label.caption_structured.main_subject.strip())
    return _deduplicate_meaningful_texts(evidence_texts)


def extract_candidate_texts_from_label(label: ParsedLabel) -> list[str]:
    candidate_texts: list[str] = []
    if label.caption_structured.visible_title.strip():
        candidate_texts.append(label.caption_structured.visible_title.strip())

    structured = label.structured_label
    if structured.kind == "mermaid":
        candidate_texts.extend(extract_mermaid_node_texts(structured.content))
    elif structured.kind == "table":
        candidate_texts.extend(parse_markdown_table(structured.content)["cell_texts"])
    elif structured.kind == "text":
        candidate_texts.extend(_extract_text_lines(structured.content))

    return _deduplicate_meaningful_texts(candidate_texts)


def calculate_text_evidence_score(
    candidate_texts: list[str], evidence_texts: list[str]
) -> tuple[float, float, dict]:
    candidates = _deduplicate_meaningful_texts(candidate_texts, min_length=2)
    evidence = _deduplicate_meaningful_texts(evidence_texts, min_length=1)

    if not candidates:
        return 0.5, 0.0, {
            "candidate_count": 0,
            "supported_count": 0,
            "unsupported_count": 0,
            "supported_matches": [],
            "unsupported_texts": [],
        }

    supported_count = 0
    unsupported_texts: list[str] = []
    supported_matches: list[dict[str, object]] = []

    for candidate in candidates:
        best_match, best_score = best_evidence_match(candidate, evidence)
        if best_score >= 0.65:
            supported_count += 1
            supported_matches.append(
                {
                    "candidate": candidate,
                    "matched_evidence": best_match,
                    "similarity": round(best_score, 4),
                }
            )
        else:
            unsupported_texts.append(candidate)

    unsupported_count = len(candidates) - supported_count
    evidence_score = supported_count / len(candidates)
    hallucination_risk = unsupported_count / len(candidates)
    details = {
        "candidate_count": len(candidates),
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "supported_matches": supported_matches,
        "unsupported_texts": unsupported_texts,
    }
    return round(evidence_score, 4), round(hallucination_risk, 4), details


def best_evidence_match(candidate_text: str, evidence_texts: list[str]) -> tuple[str | None, float]:
    best_text: str | None = None
    best_score = 0.0
    for evidence_text in evidence_texts:
        score = text_similarity(candidate_text, evidence_text)
        if score > best_score:
            best_text = evidence_text
            best_score = score
    return best_text, best_score


def text_similarity(left: str, right: str) -> float:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    compact_left = compact_text(left)
    compact_right = compact_text(right)
    if compact_left and compact_right and (
        compact_left in compact_right or compact_right in compact_left
    ):
        shorter_length = min(len(compact_left), len(compact_right))
        ratio = min(len(compact_left), len(compact_right)) / max(
            len(compact_left), len(compact_right)
        )
        if shorter_length <= 2:
            return 0.25 + 0.25 * ratio
        if shorter_length <= 4:
            return 0.45 + 0.2 * ratio
        return 0.65 + 0.25 * ratio

    if _contains_cjk(left) or _contains_cjk(right):
        bigram_score = _jaccard_similarity(_char_bigrams(normalized_left), _char_bigrams(normalized_right))
        char_score = _jaccard_similarity(set(normalized_left), set(normalized_right))
        token_score = _jaccard_similarity(tokenize(left), tokenize(right))
        return max(bigram_score, char_score, token_score)

    return max(
        _jaccard_similarity(tokenize(left), tokenize(right)),
        _jaccard_similarity(_char_bigrams(normalized_left), _char_bigrams(normalized_right)),
    )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", "", normalized).strip().lower()
    return normalized


def compact_text(text: str) -> str:
    normalized = normalize_text(text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def tokenize(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return {token for token in _TOKEN_PATTERN.findall(normalized) if token}


def is_meaningful_text(text: str, min_length: int = 1) -> bool:
    compact = compact_text(text)
    if not compact:
        return False
    if len(compact) < min_length:
        return False
    return True


def _extract_text_lines(content: str) -> list[str]:
    texts: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip(" -|:\t")
        if line:
            texts.append(line)
    return texts


def _deduplicate_meaningful_texts(values: list[str], min_length: int = 1) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not is_meaningful_text(value, min_length=min_length):
            continue
        normalized = normalize_text(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(str(value).strip())
    return ordered


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
