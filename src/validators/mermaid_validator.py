from __future__ import annotations

import re

_HEADER_PATTERNS = ("flowchart TD", "graph TD")
_NODE_PATTERN = re.compile(
    r"\b[\w-]+\s*(?:"
    r"\[\s*\"?([^\]\"\n]+?)\"?\s*\]"
    r"|\(\s*\"?([^\)\"\n]+?)\"?\s*\)"
    r"|\{\s*\"?([^\}\"\n]+?)\"?\s*\}"
    r")"
)


def extract_mermaid_node_texts(content: str) -> list[str]:
    if not content.strip():
        return []

    node_texts: list[str] = []
    seen: set[str] = set()
    for match in _NODE_PATTERN.finditer(content):
        text = next((group for group in match.groups() if group), "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        node_texts.append(text)
    return node_texts


def validate_mermaid(content: str, image_type: str | None = None) -> dict:
    stripped = content.strip()
    errors: list[str] = []
    warnings: list[str] = []

    if not stripped:
        errors.append("kind is mermaid but content is empty")
        if image_type == "flowchart":
            errors.append("image_type is flowchart but content is empty")
        return {
            "valid_basic": False,
            "node_texts": [],
            "arrow_count": 0,
            "errors": _deduplicate(errors),
            "warnings": warnings,
            "score": 0.0,
        }

    has_header = any(header in stripped for header in _HEADER_PATTERNS)
    if not has_header:
        errors.append("kind is mermaid but missing flowchart TD or graph TD header")

    node_texts = extract_mermaid_node_texts(stripped)
    arrow_count = _count_mermaid_arrows(stripped)
    line_count = len([line for line in stripped.splitlines() if line.strip()])

    if not node_texts:
        warnings.append("mermaid content does not contain recognizable node text")
    if line_count <= 2:
        warnings.append("mermaid content is unusually short")
    if arrow_count == 0 and (image_type == "flowchart" or len(node_texts) >= 2):
        warnings.append("mermaid content has no arrows for a likely flowchart")

    score = 1.0
    if not has_header:
        score -= 0.45
    if not node_texts:
        score -= 0.2
    if line_count <= 2:
        score -= 0.2
    if arrow_count == 0:
        score -= 0.15

    return {
        "valid_basic": bool(stripped) and has_header,
        "node_texts": node_texts,
        "arrow_count": arrow_count,
        "errors": _deduplicate(errors),
        "warnings": _deduplicate(warnings),
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def _count_mermaid_arrows(content: str) -> int:
    patterns = ["-->", "-.->", "==>"]
    return sum(content.count(pattern) for pattern in patterns)


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
