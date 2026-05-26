from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from src.schema import ModelOutput, ParsedLabel
from src.validators.evidence import best_evidence_match, text_similarity

_HEADER_RE = re.compile(r"^\s*(flowchart|graph)\s+TD\b", re.IGNORECASE)
_STYLE_PREFIXES = ("classdef", "class ", "style ", "linkstyle")
_NODE_TOKEN_RE = re.compile(
    r"(?P<id>[A-Za-z0-9_:-]+)\s*(?:"
    r"\[\s*\"?(?P<square>[^\]\n\"]+?)\"?\s*\]"
    r"|\(\s*\"?(?P<round>[^\)\n\"]+?)\"?\s*\)"
    r"|\{\s*\"?(?P<curly>[^\}\n\"]+?)\"?\s*\}"
    r")?"
)
_PIPE_ARROW_RE = re.compile(r"\s*-->\s*\|\s*(?P<label>[^|]+?)\s*\|")
_TEXT_ARROW_RE = re.compile(r"\s*--\s+(?P<label>.+?)\s*-->")
_DOTTED_ARROW_RE = re.compile(r"\s*-\.->")
_THICK_ARROW_RE = re.compile(r"\s*==>")
_PLAIN_ARROW_RE = re.compile(r"\s*-->")
_DUPLICATE_CLASS_RE = re.compile(r":::[A-Za-z0-9_-]+")
_COMMON_EDGE_LABELS = {
    "是",
    "否",
    "yes",
    "no",
    "y",
    "n",
    "true",
    "false",
    "ok",
}


@dataclass
class GraphNode:
    canonical_id: str
    text: str
    raw_ids: list[str] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)
    support_models: list[str] = field(default_factory=list)
    support_count: int = 0
    confidence: float = 0.0
    evidence_supported: bool = False


@dataclass
class GraphEdge:
    source_text: str
    target_text: str
    label: str = ""
    support_models: list[str] = field(default_factory=list)
    support_count: int = 0
    confidence: float = 0.0
    evidence_supported: bool = False


@dataclass
class ParsedMermaidGraph:
    model_name: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FusedGraphResult:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    mermaid: str = ""
    node_vote_details: list[dict] = field(default_factory=list)
    edge_vote_details: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    graph_confidence: float = 0.0


@dataclass
class _NodeOccurrence:
    model_name: str
    raw_id: str
    raw_text: str
    normalized_text: str


@dataclass
class _EdgeOccurrence:
    model_name: str
    source_id: str
    target_id: str
    source_text: str
    target_text: str
    label: str
    normalized_label: str


def parse_mermaid_to_graph(content: str, model_name: str) -> ParsedMermaidGraph:
    graph = ParsedMermaidGraph(model_name=model_name)
    if not content.strip():
        graph.parse_errors.append("empty mermaid content")
        return graph

    processed_content = _normalize_mermaid_content(content)
    id_to_text: dict[str, str] = {}
    node_order: list[str] = []
    raw_edges: list[tuple[str, str, str]] = []
    header_found = False

    for line_number, raw_line in enumerate(processed_content.splitlines(), start=1):
        line = _clean_mermaid_line(raw_line)
        if not line:
            continue

        header_match = _HEADER_RE.match(line)
        if header_match:
            header_found = True
            line = line[header_match.end() :].strip()
            if not line:
                continue

        for segment in _split_mermaid_segments(line):
            if not segment:
                continue
            parsed_edges = _parse_segment_edges(
                segment=segment,
                id_to_text=id_to_text,
                node_order=node_order,
                parse_errors=graph.parse_errors,
                line_number=line_number,
            )
            if parsed_edges:
                raw_edges.extend(parsed_edges)
                continue

            node_token = _parse_node_token(segment, 0)
            if node_token is not None and node_token[2] == len(segment):
                node_id, node_text, _ = node_token
                _register_node(id_to_text, node_order, node_id, node_text)
                continue

            graph.parse_errors.append(f"line {line_number}: unparsed mermaid segment: {segment}")

    if not header_found:
        graph.warnings.append("missing flowchart TD / graph TD header; parsing attempted anyway")

    graph.nodes = [
        GraphNode(
            canonical_id=node_id,
            text=id_to_text[node_id],
            raw_ids=[node_id],
            raw_texts=[id_to_text[node_id]],
            support_models=[model_name],
            support_count=1,
            confidence=1.0,
            evidence_supported=False,
        )
        for node_id in node_order
    ]
    graph.edges = [
        GraphEdge(
            source_text=id_to_text.get(source_id, source_id),
            target_text=id_to_text.get(target_id, target_id),
            label=label,
            support_models=[model_name],
            support_count=1,
            confidence=1.0,
            evidence_supported=False,
        )
        for source_id, target_id, label in raw_edges
    ]
    graph.parse_errors = _deduplicate(graph.parse_errors)
    graph.warnings = _deduplicate(graph.warnings)
    return graph


def normalize_node_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("<br/>", " ").replace("<br />", " ").replace("<br>", " ")
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = _strip_mermaid_wrappers(normalized)
    normalized = normalized.lower()
    normalized = normalized.replace("`", "").replace('"', "").replace("'", "")
    normalized = re.sub(r"[•·•]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,;|")
    return normalized


def node_similarity(a: str, b: str) -> float:
    left = normalize_node_text(a)
    right = normalize_node_text(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_bigrams = _char_bigrams(left)
    right_bigrams = _char_bigrams(right)
    bigram_jaccard = _jaccard(left_bigrams, right_bigrams)
    char_set_jaccard = _jaccard(set(left), set(right))
    return round(0.6 * bigram_jaccard + 0.4 * char_set_jaccard, 4)


def cluster_nodes(graphs: list[ParsedMermaidGraph], evidence_texts: list[str]) -> list[GraphNode]:
    unique_model_count = max(len(graphs), 1)
    clusters: list[list[_NodeOccurrence]] = []

    for graph in graphs:
        deduped_nodes = _dedupe_graph_nodes(graph)
        for node in deduped_nodes:
            occurrence = _NodeOccurrence(
                model_name=graph.model_name,
                raw_id=node.raw_ids[0] if node.raw_ids else node.canonical_id,
                raw_text=node.text,
                normalized_text=normalize_node_text(node.text),
            )
            if not occurrence.normalized_text:
                continue

            best_index = -1
            best_score = 0.0
            for index, cluster in enumerate(clusters):
                if any(item.model_name == graph.model_name for item in cluster):
                    continue
                score = max(
                    node_similarity(occurrence.normalized_text, item.normalized_text)
                    for item in cluster
                )
                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index >= 0 and best_score >= 0.72:
                clusters[best_index].append(occurrence)
            else:
                clusters.append([occurrence])

    fused_nodes: list[GraphNode] = []
    for index, cluster in enumerate(clusters, start=1):
        support_models = [item.model_name for item in cluster]
        representative_text = _select_representative_text(
            [(item.raw_text, item.model_name) for item in cluster]
        )
        evidence_supported = _is_supported_by_evidence(representative_text, evidence_texts, 0.65)
        fused_nodes.append(
            GraphNode(
                canonical_id=f"node_{index}",
                text=representative_text,
                raw_ids=[item.raw_id for item in cluster],
                raw_texts=[item.raw_text for item in cluster],
                support_models=support_models,
                support_count=len(set(support_models)),
                confidence=round(len(set(support_models)) / unique_model_count, 4),
                evidence_supported=evidence_supported,
            )
        )
    return fused_nodes


def cluster_edges(
    graphs: list[ParsedMermaidGraph],
    fused_nodes: list[GraphNode],
    evidence_texts: list[str],
) -> list[GraphEdge]:
    unique_model_count = max(len(graphs), 1)
    mapped_edges, _ = _map_edges_to_fused_nodes(graphs=graphs, fused_nodes=fused_nodes)
    if not mapped_edges:
        return []

    node_lookup = {node.canonical_id: node for node in fused_nodes}
    clusters: list[list[_EdgeOccurrence]] = []

    for edge in mapped_edges:
        best_index = -1
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            if any(item.model_name == edge.model_name for item in cluster):
                continue
            sample = cluster[0]
            if sample.source_id != edge.source_id or sample.target_id != edge.target_id:
                continue
            score = _label_similarity(sample.label, edge.label)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index >= 0 and (
            (_is_empty_text(edge.label) and _is_empty_text(clusters[best_index][0].label))
            or best_score >= 0.65
        ):
            clusters[best_index].append(edge)
        else:
            clusters.append([edge])

    fused_edges: list[GraphEdge] = []
    for cluster in clusters:
        support_models = sorted({item.model_name for item in cluster})
        representative_label = _select_representative_text(
            [(item.label, item.model_name) for item in cluster]
        )
        source_node = node_lookup[cluster[0].source_id]
        target_node = node_lookup[cluster[0].target_id]
        label_supported = (
            True
            if not representative_label.strip()
            else _is_common_edge_label(representative_label)
            or _is_supported_by_evidence(representative_label, evidence_texts, 0.65)
        )
        fused_edges.append(
            GraphEdge(
                source_text=source_node.text,
                target_text=target_node.text,
                label=representative_label,
                support_models=support_models,
                support_count=len(support_models),
                confidence=round(len(support_models) / unique_model_count, 4),
                evidence_supported=(
                    source_node.evidence_supported
                    and target_node.evidence_supported
                    and label_supported
                ),
            )
        )
    return fused_edges


def build_fused_mermaid(
    fused_nodes: list[GraphNode],
    fused_edges: list[GraphEdge],
    min_node_confidence: float,
    min_edge_confidence: float,
) -> str:
    selected_edges = [edge for edge in fused_edges if edge.confidence >= min_edge_confidence]
    selected_node_keys = {
        normalize_node_text(node.text)
        for node in fused_nodes
        if node.confidence >= min_node_confidence
    }
    for edge in selected_edges:
        selected_node_keys.add(normalize_node_text(edge.source_text))
        selected_node_keys.add(normalize_node_text(edge.target_text))

    selected_nodes = [
        node for node in fused_nodes if normalize_node_text(node.text) in selected_node_keys
    ]
    if not selected_nodes:
        return "flowchart TD"

    numbering = {
        node.canonical_id: f"N{index}"
        for index, node in enumerate(selected_nodes, start=1)
    }
    text_to_canonical = {
        normalize_node_text(node.text): node.canonical_id for node in selected_nodes
    }

    lines = ["flowchart TD"]
    for node in selected_nodes:
        node_id = numbering[node.canonical_id]
        lines.append(f'{node_id}["{_escape_mermaid_text(node.text)}"]')

    for edge in selected_edges:
        source_key = text_to_canonical.get(normalize_node_text(edge.source_text))
        target_key = text_to_canonical.get(normalize_node_text(edge.target_text))
        if source_key is None or target_key is None:
            continue
        source_id = numbering[source_key]
        target_id = numbering[target_key]
        if edge.label.strip():
            lines.append(
                f"{source_id} -->|{_escape_mermaid_label(edge.label)}| {target_id}"
            )
        else:
            lines.append(f"{source_id} --> {target_id}")
    return "\n".join(lines)


def fuse_mermaid_outputs(
    labels: list[ParsedLabel],
    model_outputs: list[ModelOutput],
    evidence_texts: list[str],
) -> FusedGraphResult | None:
    paired = [
        (label, output)
        for label, output in zip(labels, model_outputs)
        if label.structured_label.kind == "mermaid" and label.structured_label.content.strip()
    ]
    if len(paired) < 2:
        return None

    parsed_graphs = [
        parse_mermaid_to_graph(label.structured_label.content, output.model_name)
        for label, output in paired
    ]
    fused_nodes = cluster_nodes(parsed_graphs, evidence_texts)
    fused_edges = cluster_edges(parsed_graphs, fused_nodes, evidence_texts)

    num_models = len(parsed_graphs)
    min_node_confidence, min_edge_confidence = _default_thresholds(num_models)
    mermaid = build_fused_mermaid(
        fused_nodes=fused_nodes,
        fused_edges=fused_edges,
        min_node_confidence=min_node_confidence,
        min_edge_confidence=min_edge_confidence,
    )

    mapped_edges, mapping_warnings = _map_edges_to_fused_nodes(
        graphs=parsed_graphs,
        fused_nodes=fused_nodes,
    )
    warnings = _collect_graph_warnings(
        parsed_graphs=parsed_graphs,
        fused_nodes=fused_nodes,
        fused_edges=fused_edges,
        mapped_edges=mapped_edges,
        min_node_confidence=min_node_confidence,
        min_edge_confidence=min_edge_confidence,
        mapping_warnings=mapping_warnings,
    )
    critical_errors = _build_critical_errors(parsed_graphs=parsed_graphs, warnings=warnings)
    majority_type = _majority_value([label.image_type for label, _ in paired])
    if majority_type == "flowchart" and not fused_edges:
        critical_errors.append("empty_fused_edges")
    if fused_nodes and mermaid.strip() == "flowchart TD":
        critical_errors.append("empty_fused_mermaid")

    graph_confidence = _compute_graph_confidence(
        parsed_graphs=parsed_graphs,
        fused_nodes=fused_nodes,
        fused_edges=fused_edges,
        warnings=warnings,
        critical_errors=critical_errors,
    )

    return FusedGraphResult(
        nodes=fused_nodes,
        edges=fused_edges,
        mermaid=mermaid,
        node_vote_details=[asdict(node) for node in fused_nodes],
        edge_vote_details=[asdict(edge) for edge in fused_edges],
        warnings=_deduplicate(warnings),
        critical_errors=_deduplicate(critical_errors),
        graph_confidence=graph_confidence,
    )


def _clean_mermaid_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("%%"):
        return ""
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _STYLE_PREFIXES):
        return ""
    cleaned = _DUPLICATE_CLASS_RE.sub("", stripped).strip()
    return cleaned


def _split_mermaid_segments(line: str) -> list[str]:
    if ";" not in line:
        stripped = line.strip()
        return [stripped] if stripped else []

    segments: list[str] = []
    current: list[str] = []
    square_depth = 0
    round_depth = 0
    curly_depth = 0
    in_double_quote = False
    in_single_quote = False

    for char in line:
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote

        if not in_double_quote and not in_single_quote:
            if char == "[":
                square_depth += 1
            elif char == "]" and square_depth > 0:
                square_depth -= 1
            elif char == "(":
                round_depth += 1
            elif char == ")" and round_depth > 0:
                round_depth -= 1
            elif char == "{":
                curly_depth += 1
            elif char == "}" and curly_depth > 0:
                curly_depth -= 1
            elif (
                char == ";"
                and square_depth == 0
                and round_depth == 0
                and curly_depth == 0
            ):
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments


def _parse_segment_edges(
    segment: str,
    id_to_text: dict[str, str],
    node_order: list[str],
    parse_errors: list[str],
    line_number: int,
) -> list[tuple[str, str, str]]:
    first = _parse_node_token(segment, 0)
    if first is None:
        return []

    source_id, source_text, position = first
    _register_node(id_to_text, node_order, source_id, source_text)
    previous_id = source_id
    edges: list[tuple[str, str, str]] = []
    saw_arrow = False

    while True:
        arrow_match = _parse_arrow(segment, position)
        if arrow_match is None:
            break
        label, position = arrow_match
        next_node = _parse_node_token(segment, position)
        if next_node is None:
            parse_errors.append(
                f"line {line_number}: edge target missing after arrow in segment: {segment}"
            )
            return edges
        target_id, target_text, position = next_node
        _register_node(id_to_text, node_order, target_id, target_text)
        edges.append((previous_id, target_id, label))
        previous_id = target_id
        saw_arrow = True

    if saw_arrow:
        if segment[position:].strip():
            parse_errors.append(
                f"line {line_number}: trailing content after edge parse: {segment[position:].strip()}"
            )
        return edges
    return []


def _parse_node_token(segment: str, position: int) -> tuple[str, str, int] | None:
    while position < len(segment) and segment[position].isspace():
        position += 1
    match = _NODE_TOKEN_RE.match(segment, position)
    if match is None:
        return None

    node_id = match.group("id")
    raw_text = next(
        (
            group
            for group in (
                match.group("square"),
                match.group("round"),
                match.group("curly"),
            )
            if group is not None
        ),
        node_id,
    )
    text = _strip_mermaid_wrappers(raw_text.strip()) or node_id
    return node_id, text, match.end()


def _parse_arrow(segment: str, position: int) -> tuple[str, int] | None:
    for pattern in (_PIPE_ARROW_RE, _TEXT_ARROW_RE, _DOTTED_ARROW_RE, _THICK_ARROW_RE, _PLAIN_ARROW_RE):
        match = pattern.match(segment, position)
        if match is None:
            continue
        label = str(match.groupdict().get("label", "") or "").strip()
        return label, match.end()
    return None


def _register_node(
    id_to_text: dict[str, str],
    node_order: list[str],
    node_id: str,
    node_text: str,
) -> None:
    if node_id not in id_to_text:
        node_order.append(node_id)
        id_to_text[node_id] = node_text or node_id
        return

    current_text = id_to_text[node_id]
    candidate = node_text or node_id
    if _text_information_score(candidate) > _text_information_score(current_text):
        id_to_text[node_id] = candidate


def _normalize_mermaid_content(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(content or ""))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    chars: list[str] = []
    square_depth = 0
    round_depth = 0
    curly_depth = 0

    for char in normalized:
        if char == "\n" and (square_depth > 0 or round_depth > 0 or curly_depth > 0):
            chars.append(" ")
            continue

        chars.append(char)
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1
        elif char == "(":
            round_depth += 1
        elif char == ")" and round_depth > 0:
            round_depth -= 1
        elif char == "{":
            curly_depth += 1
        elif char == "}" and curly_depth > 0:
            curly_depth -= 1
    return "".join(chars)


def _strip_mermaid_wrappers(text: str) -> str:
    value = str(text or "").strip()
    changed = True
    while value and changed:
        changed = False
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1].strip()
            changed = True
        for left, right in (("[", "]"), ("(", ")"), ("{", "}")):
            if value.startswith(left) and value.endswith(right) and len(value) >= 2:
                value = value[1:-1].strip()
                changed = True
    return value


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _dedupe_graph_nodes(graph: ParsedMermaidGraph) -> list[GraphNode]:
    seen: dict[str, GraphNode] = {}
    ordered_keys: list[str] = []
    for node in graph.nodes:
        normalized = normalize_node_text(node.text)
        if not normalized:
            continue
        if normalized in seen:
            graph.warnings.append(f"duplicate node text: {node.text}")
            continue
        seen[normalized] = node
        ordered_keys.append(normalized)
    graph.warnings = _deduplicate(graph.warnings)
    return [seen[key] for key in ordered_keys]


def _select_representative_text(candidates: list[tuple[str, str]]) -> str:
    if not candidates:
        return ""

    by_normalized: dict[str, dict[str, object]] = {}
    for raw_text, model_name in candidates:
        normalized = normalize_node_text(raw_text)
        if normalized not in by_normalized:
            by_normalized[normalized] = {
                "best_text": raw_text.strip(),
                "models": {model_name},
            }
            continue
        by_normalized[normalized]["models"].add(model_name)
        best_text = str(by_normalized[normalized]["best_text"])
        if _text_information_score(raw_text) > _text_information_score(best_text):
            by_normalized[normalized]["best_text"] = raw_text.strip()

    ranked = sorted(
        by_normalized.values(),
        key=lambda item: (
            -len(item["models"]),
            -_text_information_score(str(item["best_text"])),
            -len(str(item["best_text"])),
            str(item["best_text"]),
        ),
    )
    return str(ranked[0]["best_text"]).strip()


def _text_information_score(text: str) -> int:
    normalized = normalize_node_text(text)
    meaningful_chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9±/%\-()]+", normalized)
    return sum(len(item) for item in meaningful_chars)


def _is_supported_by_evidence(text: str, evidence_texts: list[str], threshold: float) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    _, score = best_evidence_match(candidate, evidence_texts)
    return score >= threshold


def _map_edges_to_fused_nodes(
    graphs: list[ParsedMermaidGraph],
    fused_nodes: list[GraphNode],
) -> tuple[list[_EdgeOccurrence], list[str]]:
    model_node_maps: dict[str, list[tuple[str, str, GraphNode]]] = defaultdict(list)
    for node in fused_nodes:
        for raw_id, raw_text, model_name in zip(node.raw_ids, node.raw_texts, node.support_models):
            model_node_maps[model_name].append((raw_id, raw_text, node))

    mapped_edges: list[_EdgeOccurrence] = []
    warnings: list[str] = []

    for graph in graphs:
        seen_keys: set[tuple[str, str, str]] = set()
        for edge in graph.edges:
            source_node = _resolve_fused_node_for_model(
                model_name=graph.model_name,
                node_text=edge.source_text,
                fused_nodes=fused_nodes,
                model_node_maps=model_node_maps,
            )
            target_node = _resolve_fused_node_for_model(
                model_name=graph.model_name,
                node_text=edge.target_text,
                fused_nodes=fused_nodes,
                model_node_maps=model_node_maps,
            )
            if source_node is None or target_node is None:
                warnings.append(
                    f"unmapped_edge:{graph.model_name}:{edge.source_text}->{edge.target_text}"
                )
                continue

            key = (
                source_node.canonical_id,
                target_node.canonical_id,
                normalize_node_text(edge.label),
            )
            if key in seen_keys:
                warnings.append(
                    f"duplicate_edge:{graph.model_name}:{edge.source_text}->{edge.target_text}"
                )
                continue
            seen_keys.add(key)
            mapped_edges.append(
                _EdgeOccurrence(
                    model_name=graph.model_name,
                    source_id=source_node.canonical_id,
                    target_id=target_node.canonical_id,
                    source_text=source_node.text,
                    target_text=target_node.text,
                    label=edge.label.strip(),
                    normalized_label=normalize_node_text(edge.label),
                )
            )

    return mapped_edges, _deduplicate(warnings)


def _resolve_fused_node_for_model(
    model_name: str,
    node_text: str,
    fused_nodes: list[GraphNode],
    model_node_maps: dict[str, list[tuple[str, str, GraphNode]]],
) -> GraphNode | None:
    normalized = normalize_node_text(node_text)
    best_node: GraphNode | None = None
    best_score = 0.0

    for raw_id, raw_text, node in model_node_maps.get(model_name, []):
        score = max(
            node_similarity(normalized, raw_text),
            node_similarity(normalized, raw_id),
            node_similarity(normalized, node.text),
        )
        if score > best_score:
            best_score = score
            best_node = node
    if best_node is not None and best_score >= 0.72:
        return best_node

    for node in fused_nodes:
        candidate_scores = [node_similarity(normalized, node.text)]
        candidate_scores.extend(node_similarity(normalized, raw) for raw in node.raw_texts)
        score = max(candidate_scores)
        if score > best_score:
            best_score = score
            best_node = node
    if best_node is not None and best_score >= 0.72:
        return best_node
    return None


def _label_similarity(left: str, right: str) -> float:
    if _is_empty_text(left) and _is_empty_text(right):
        return 1.0
    return max(node_similarity(left, right), text_similarity(left, right))


def _is_empty_text(text: str) -> bool:
    return not normalize_node_text(text)


def _is_common_edge_label(label: str) -> bool:
    return normalize_node_text(label) in _COMMON_EDGE_LABELS


def _collect_graph_warnings(
    parsed_graphs: list[ParsedMermaidGraph],
    fused_nodes: list[GraphNode],
    fused_edges: list[GraphEdge],
    mapped_edges: list[_EdgeOccurrence],
    min_node_confidence: float,
    min_edge_confidence: float,
    mapping_warnings: list[str],
) -> list[str]:
    warnings: list[str] = []
    for graph in parsed_graphs:
        warnings.extend(graph.warnings)
        warnings.extend(f"parse_error:{error}" for error in graph.parse_errors)

    for node in fused_nodes:
        if node.support_count == 1:
            warnings.append(f"low_support_node:{node.text}")
        if not node.evidence_supported:
            warnings.append(f"unsupported_node:{node.text}")

    for edge in fused_edges:
        if edge.support_count == 1:
            warnings.append(
                f"low_support_edge:{edge.source_text}->{edge.label or '(empty)'}->{edge.target_text}"
            )
        source_supported = next(
            (node.evidence_supported for node in fused_nodes if node.text == edge.source_text),
            False,
        )
        target_supported = next(
            (node.evidence_supported for node in fused_nodes if node.text == edge.target_text),
            False,
        )
        if not source_supported or not target_supported:
            warnings.append(
                f"unsupported_edge_nodes:{edge.source_text}->{edge.label or '(empty)'}->{edge.target_text}"
            )

    node_confidence_lookup = {
        normalize_node_text(node.text): node.confidence for node in fused_nodes
    }
    for edge in fused_edges:
        if edge.confidence < min_edge_confidence:
            continue
        if (
            node_confidence_lookup.get(normalize_node_text(edge.source_text), 0.0)
            < min_node_confidence
            or node_confidence_lookup.get(normalize_node_text(edge.target_text), 0.0)
            < min_node_confidence
        ):
            warnings.append(
                f"edge_requires_low_conf_nodes:{edge.source_text}->{edge.label or '(empty)'}->{edge.target_text}"
            )

    warnings.extend(mapping_warnings)
    warnings.extend(_detect_edge_disagreement(mapped_edges, fused_nodes))
    return _deduplicate(warnings)


def _detect_edge_disagreement(
    mapped_edges: list[_EdgeOccurrence],
    fused_nodes: list[GraphNode],
) -> list[str]:
    text_lookup = {node.canonical_id: node.text for node in fused_nodes}
    pair_signatures: dict[tuple[str, str], dict[tuple[str, str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    support_lookup = {node.canonical_id: node.support_count for node in fused_nodes}
    for edge in mapped_edges:
        pair_key = tuple(sorted([edge.source_id, edge.target_id]))
        signature = (edge.source_id, edge.target_id, edge.normalized_label)
        pair_signatures[pair_key][signature].add(edge.model_name)

    warnings: list[str] = []
    for pair_key, signature_map in pair_signatures.items():
        total_models = len({model for models in signature_map.values() for model in models})
        if total_models < 2 or len(signature_map) < 2:
            continue
        if min(support_lookup.get(pair_key[0], 0), support_lookup.get(pair_key[1], 0)) < 2:
            continue
        warnings.append(
            "edge_disagreement:"
            + ",".join(
                f"{text_lookup.get(source, source)}->{text_lookup.get(target, target)}:{label or '(empty)'}"
                for source, target, label in sorted(signature_map.keys())
            )
        )
    return warnings


def _default_thresholds(num_models: int) -> tuple[float, float]:
    if num_models == 2:
        return 1.0, 1.0
    return 2.0 / 3.0, 2.0 / 3.0


def _build_critical_errors(
    parsed_graphs: list[ParsedMermaidGraph],
    warnings: list[str],
) -> list[str]:
    critical_errors: list[str] = []
    parse_error_count = sum(len(graph.parse_errors) for graph in parsed_graphs)
    if parse_error_count > 0:
        critical_errors.append("mermaid_parse_errors_present")
    if _count_warning_prefix(warnings, "edge_disagreement:") > 0:
        critical_errors.append("edge_disagreement")
    if _count_warning_prefix(warnings, "duplicate node text:") > 0:
        critical_errors.append("duplicate_node_texts_present")
    return _deduplicate(critical_errors)


def _compute_graph_confidence(
    parsed_graphs: list[ParsedMermaidGraph],
    fused_nodes: list[GraphNode],
    fused_edges: list[GraphEdge],
    warnings: list[str],
    critical_errors: list[str],
) -> float:
    average_node_confidence = _average([node.confidence for node in fused_nodes], default=0.0)
    average_edge_confidence = _average([edge.confidence for edge in fused_edges], default=0.0)
    supported_items = sum(1 for node in fused_nodes if node.evidence_supported) + sum(
        1 for edge in fused_edges if edge.evidence_supported
    )
    total_items = len(fused_nodes) + len(fused_edges)
    evidence_support_ratio = supported_items / total_items if total_items else 0.0
    base_confidence = (
        0.4 * average_node_confidence
        + 0.4 * average_edge_confidence
        + 0.2 * evidence_support_ratio
    )

    parse_error_count = sum(len(graph.parse_errors) for graph in parsed_graphs)
    edge_disagreement_count = _count_warning_prefix(warnings, "edge_disagreement:")
    low_support_edge_count = _count_warning_prefix(warnings, "low_support_edge:")
    unsupported_claim_count = _count_warning_prefix(warnings, "unsupported_node:") + _count_warning_prefix(
        warnings, "unsupported_edge_nodes:"
    )
    low_support_edge_ratio = low_support_edge_count / len(fused_edges) if fused_edges else 0.0
    unsupported_claim_ratio = unsupported_claim_count / total_items if total_items else 0.0

    penalty = min(0.22, 0.05 * parse_error_count)
    penalty += min(0.18, 0.12 * edge_disagreement_count)
    penalty += 0.10 * low_support_edge_ratio
    penalty += 0.08 * unsupported_claim_ratio
    if "duplicate_node_texts_present" in set(critical_errors):
        penalty += 0.08
    if critical_errors:
        penalty += 0.02

    return round(max(0.0, base_confidence - penalty), 4)


def _average(values: list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _count_warning_prefix(values: list[str], prefix: str) -> int:
    return sum(1 for value in values if value.startswith(prefix))


def _majority_value(values: list[str]) -> str:
    if not values:
        return "unknown"
    counter = Counter(values)
    highest = counter.most_common(1)[0][1]
    for value in values:
        if counter[value] == highest:
            return value
    return values[0]


def _escape_mermaid_text(text: str) -> str:
    return str(text or "").replace('"', "&quot;")


def _escape_mermaid_label(text: str) -> str:
    return _escape_mermaid_text(text).replace("|", "/")


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
