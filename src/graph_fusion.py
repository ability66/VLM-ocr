from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from src.schema import ModelOutput, ParsedLabel

HEADER_RE = re.compile(r"^\s*(flowchart|graph)\s+TD\b", re.IGNORECASE)
STYLE_PREFIXES = ("classdef", "class ", "style ", "linkstyle")
NODE_TOKEN_RE = re.compile(
    r"(?P<id>[A-Za-z0-9_:-]+)\s*(?:"
    r"\[\s*\"?(?P<square>[^\]\n\"]+?)\"?\s*\]"
    r"|\(\s*\"?(?P<round>[^\)\n\"]+?)\"?\s*\)"
    r"|\{\s*\"?(?P<curly>[^\}\n\"]+?)\"?\s*\}"
    r")?"
)
PIPE_ARROW_RE = re.compile(r"\s*-->\s*\|\s*(?P<label>[^|]+?)\s*\|")
TEXT_ARROW_RE = re.compile(r"\s*--\s+(?P<label>.+?)\s*-->")
DOTTED_ARROW_RE = re.compile(r"\s*-\.->")
THICK_ARROW_RE = re.compile(r"\s*==>")
PLAIN_ARROW_RE = re.compile(r"\s*-->")
DUPLICATE_CLASS_RE = re.compile(r":::[A-Za-z0-9_-]+")
VALID_SHAPES = {"rectangle", "diamond", "ellipse", "rounded", "unknown"}


@dataclass
class VisualNodeAnchor:
    model_name: str
    node_id: str
    order_index: int | None
    row_index: int | None
    col_index: int | None
    bbox_hint: list[float] | None
    shape: str
    text: str
    graph_source: str = "model"


@dataclass
class VisualEdgeClaim:
    model_name: str
    source: str
    target: str
    label: str
    graph_source: str = "model"


@dataclass
class ParsedVisualGraph:
    model_name: str
    graph_source: str
    node_order_rule: str = "top_to_bottom_left_to_right"
    nodes: list[VisualNodeAnchor] = field(default_factory=list)
    edges: list[VisualEdgeClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    weak_candidate: bool = False


@dataclass
class FusedVisualNode:
    fused_id: str
    order_index: int
    row_index_votes: list[int] = field(default_factory=list)
    col_index_votes: list[int] = field(default_factory=list)
    bbox_hints: list[list[float]] = field(default_factory=list)
    shape_votes: list[str] = field(default_factory=list)
    text_votes: list[str] = field(default_factory=list)
    support_models: list[str] = field(default_factory=list)
    support_count: int = 0
    confidence: float = 0.0
    text_consistency: float = 1.0
    representative_text: str = ""
    representative_shape: str = "unknown"


@dataclass
class FusedVisualEdge:
    source: str
    target: str
    label_votes: list[str] = field(default_factory=list)
    support_models: list[str] = field(default_factory=list)
    support_count: int = 0
    confidence: float = 0.0
    label_consistency: float = 1.0
    label: str = ""


@dataclass
class FusedGraphResult:
    nodes: list[FusedVisualNode] = field(default_factory=list)
    edges: list[FusedVisualEdge] = field(default_factory=list)
    mermaid: str = ""
    node_vote_details: list[dict[str, Any]] = field(default_factory=list)
    edge_vote_details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    graph_confidence: float = 0.0
    fusion_method: str = "none"
    fusion_status: str = "failed"
    node_alignment_errors: list[str] = field(default_factory=list)
    edge_alignment_errors: list[str] = field(default_factory=list)
    low_support_edges: list[dict[str, Any]] = field(default_factory=list)
    low_text_consistency_nodes: list[str] = field(default_factory=list)
    inconsistent_node_count: int = 0


def extract_weak_flowchart_graph_from_mermaid(content: str) -> dict[str, Any] | None:
    if not str(content or "").strip():
        return None

    processed_content = _normalize_mermaid_content(content)
    node_lookup: dict[str, dict[str, str]] = {}
    node_order: list[str] = []
    raw_edges: list[tuple[str, str, str]] = []

    for line_number, raw_line in enumerate(processed_content.splitlines(), start=1):
        line = _clean_mermaid_line(raw_line)
        if not line:
            continue

        header_match = HEADER_RE.match(line)
        if header_match:
            line = line[header_match.end() :].strip()
            if not line:
                continue

        for segment in _split_mermaid_segments(line):
            if not segment:
                continue
            parsed_edges = _parse_segment_edges(
                segment=segment,
                node_lookup=node_lookup,
                node_order=node_order,
            )
            if parsed_edges:
                raw_edges.extend(parsed_edges)
                continue

            node_token = _parse_node_token(segment, 0)
            if node_token is None or node_token[3] != len(segment):
                continue
            raw_id, raw_text, shape, _ = node_token
            _register_mermaid_node(
                node_lookup=node_lookup,
                node_order=node_order,
                raw_id=raw_id,
                text=raw_text,
                shape=shape,
            )

    if not node_order:
        return None

    raw_to_normalized = {
        raw_id: f"N{index:03d}" for index, raw_id in enumerate(node_order, start=1)
    }
    nodes = [
        {
            "node_id": raw_to_normalized[raw_id],
            "order_index": index,
            "row_index": None,
            "col_index": None,
            "bbox_hint": None,
            "shape": node_lookup[raw_id]["shape"],
            "text": node_lookup[raw_id]["text"],
        }
        for index, raw_id in enumerate(node_order, start=1)
    ]
    edges: list[dict[str, Any]] = []
    for raw_source, raw_target, label in raw_edges:
        source = raw_to_normalized.get(raw_source)
        target = raw_to_normalized.get(raw_target)
        if source is None or target is None:
            continue
        edges.append({"source": source, "target": target, "label": label})

    return {
        "node_order_rule": "mermaid_appearance_order",
        "nodes": nodes,
        "edges": edges,
        "graph_source": "mermaid_fallback",
        "weak_candidate": True,
    }


def bbox_center_distance(
    bbox_a: list[float] | None,
    bbox_b: list[float] | None,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 1.0
    center_a = ((bbox_a[0] + bbox_a[2]) / 2.0, (bbox_a[1] + bbox_a[3]) / 2.0)
    center_b = ((bbox_b[0] + bbox_b[2]) / 2.0, (bbox_b[1] + bbox_b[3]) / 2.0)
    return round(math.dist(center_a, center_b), 4)


def bbox_iou(
    bbox_a: list[float] | None,
    bbox_b: list[float] | None,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 0.0

    inter_left = max(bbox_a[0], bbox_b[0])
    inter_top = max(bbox_a[1], bbox_b[1])
    inter_right = min(bbox_a[2], bbox_b[2])
    inter_bottom = min(bbox_a[3], bbox_b[3])
    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, bbox_a[2] - bbox_a[0]) * max(0.0, bbox_a[3] - bbox_a[1])
    area_b = max(0.0, bbox_b[2] - bbox_b[0]) * max(0.0, bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return round(inter_area / union, 4)


def compute_text_consistency(text_votes: list[str]) -> float:
    normalized_votes = [normalize_vote_text(text) for text in text_votes if normalize_vote_text(text)]
    if len(normalized_votes) <= 1:
        return 1.0

    scores: list[float] = []
    for index, left in enumerate(normalized_votes):
        for right in normalized_votes[index + 1 :]:
            scores.append(_bigram_jaccard(left, right))
    if not scores:
        return 1.0
    return round(sum(scores) / len(scores), 4)


def build_fused_mermaid_from_visual_graph(
    nodes: list[FusedVisualNode],
    edges: list[FusedVisualEdge],
) -> str:
    if not nodes:
        return ""

    ordered_nodes = sorted(nodes, key=lambda item: (item.order_index, item.fused_id))
    node_order_lookup = {node.fused_id: node.order_index for node in ordered_nodes}
    lines = ["flowchart TD"]

    for node in ordered_nodes:
        text = node.representative_text.strip() or node.fused_id
        lines.append(_format_mermaid_node(node_id=node.fused_id, text=text, shape=node.representative_shape))

    ordered_edges = sorted(
        edges,
        key=lambda item: (
            node_order_lookup.get(item.source, 10**9),
            node_order_lookup.get(item.target, 10**9),
            item.source,
            item.target,
        ),
    )
    for edge in ordered_edges:
        if edge.label.strip():
            lines.append(
                f"{edge.source} -->|{_escape_mermaid_label(edge.label)}| {edge.target}"
            )
        else:
            lines.append(f"{edge.source} --> {edge.target}")
    return "\n".join(lines)


def _format_mermaid_node(node_id: str, text: str, shape: str) -> str:
    escaped_text = _escape_mermaid_text(text)
    if shape == "diamond":
        return f'{node_id}{{"{escaped_text}"}}'
    if shape == "rounded":
        return f'{node_id}("{escaped_text}")'
    if shape == "ellipse":
        return f'{node_id}(("{escaped_text}"))'
    return f'{node_id}["{escaped_text}"]'


def fuse_mermaid_outputs(
    labels: list[ParsedLabel],
    model_outputs: list[ModelOutput],
    evidence_texts: list[str],
) -> FusedGraphResult | None:
    del evidence_texts

    parsed_graphs: list[ParsedVisualGraph] = []
    for label, output in zip(labels, model_outputs):
        graph = _build_visual_graph_for_label(label=label, model_name=output.model_name)
        if graph is not None:
            parsed_graphs.append(graph)

    if len(parsed_graphs) < 2:
        return None

    num_models = len(parsed_graphs)
    fusion_method = (
        "visual_order"
        if all(not graph.weak_candidate and graph.graph_source == "model" for graph in parsed_graphs)
        else "mermaid_fallback"
    )

    warnings: list[str] = []
    critical_errors: list[str] = []
    node_alignment_errors: list[str] = []
    edge_alignment_errors: list[str] = []

    for graph in parsed_graphs:
        warnings.extend(graph.warnings)
        if graph.parse_errors:
            warnings.extend(f"{graph.model_name}:{error}" for error in graph.parse_errors)

    if any(graph.parse_errors for graph in parsed_graphs):
        critical_errors.append("visual_graph_parse_errors")

    if any(not graph.nodes for graph in parsed_graphs):
        critical_errors.append("empty_visual_nodes")

    selected_node_ids, status_hint, selection_errors, inconsistent_node_count = _select_fused_node_ids(
        parsed_graphs=parsed_graphs
    )
    node_alignment_errors.extend(selection_errors)

    if not selected_node_ids:
        critical_errors.append("node_id_alignment_failed")
        result = FusedGraphResult(
            warnings=_deduplicate(warnings),
            critical_errors=_deduplicate(critical_errors),
            fusion_method=fusion_method,
            fusion_status="failed",
            node_alignment_errors=_deduplicate(node_alignment_errors),
            edge_alignment_errors=[],
            inconsistent_node_count=inconsistent_node_count,
        )
        result.graph_confidence = _compute_graph_confidence(result, num_models=num_models)
        return result

    fused_nodes, node_errors, low_text_nodes, node_warnings = _fuse_nodes(
        parsed_graphs=parsed_graphs,
        selected_node_ids=selected_node_ids,
        num_models=num_models,
    )
    node_alignment_errors.extend(node_errors)
    warnings.extend(node_warnings)

    fused_edges, low_support_edges, edge_errors, edge_warnings = _fuse_edges(
        parsed_graphs=parsed_graphs,
        fused_nodes=fused_nodes,
        num_models=num_models,
    )
    edge_alignment_errors.extend(edge_errors)
    warnings.extend(edge_warnings)

    (
        fused_nodes,
        fused_edges,
        low_support_edges,
        low_text_nodes,
        node_alignment_errors,
        edge_alignment_errors,
        warnings,
    ) = _prune_orphan_low_consistency_nodes(
        fused_nodes=fused_nodes,
        fused_edges=fused_edges,
        low_support_edges=low_support_edges,
        low_text_nodes=low_text_nodes,
        node_alignment_errors=node_alignment_errors,
        edge_alignment_errors=edge_alignment_errors,
        warnings=warnings,
        num_models=num_models,
    )

    if num_models == 2 and any(error.startswith("node_position_conflict:") for error in node_alignment_errors):
        status_hint = "ambiguous"

    if fusion_method == "mermaid_fallback" and status_hint != "failed":
        status_hint = "ambiguous"

    if status_hint == "fused" and not fused_edges:
        status_hint = "partial"
    if status_hint == "fused" and edge_alignment_errors:
        status_hint = "partial"
    mermaid = ""
    if fused_nodes:
        mermaid = build_fused_mermaid_from_visual_graph(fused_nodes, fused_edges)
    if not mermaid and fused_nodes:
        critical_errors.append("empty_fused_mermaid")

    result = FusedGraphResult(
        nodes=fused_nodes,
        edges=fused_edges,
        mermaid=mermaid,
        node_vote_details=[asdict(node) for node in fused_nodes],
        edge_vote_details=[asdict(edge) for edge in fused_edges],
        warnings=_deduplicate(warnings),
        critical_errors=_deduplicate(critical_errors),
        fusion_method=fusion_method,
        fusion_status=status_hint,
        node_alignment_errors=_deduplicate(node_alignment_errors),
        edge_alignment_errors=_deduplicate(edge_alignment_errors),
        low_support_edges=low_support_edges,
        low_text_consistency_nodes=_deduplicate(low_text_nodes),
        inconsistent_node_count=inconsistent_node_count,
    )
    result.graph_confidence = _compute_graph_confidence(result, num_models=num_models)

    previous_status = result.fusion_status
    if result.fusion_status == "fused":
        if result.fusion_method != "visual_order":
            result.fusion_status = "ambiguous"
        elif result.graph_confidence < 0.70:
            result.fusion_status = "partial"
        elif result.critical_errors:
            result.fusion_status = "failed"
    if result.fusion_status != previous_status:
        result.graph_confidence = _compute_graph_confidence(result, num_models=num_models)

    return result


def _build_visual_graph_for_label(
    label: ParsedLabel,
    model_name: str,
) -> ParsedVisualGraph | None:
    payload = label.flowchart_graph if isinstance(label.flowchart_graph, dict) else None
    if payload is None and label.structured_label.kind == "mermaid":
        payload = extract_weak_flowchart_graph_from_mermaid(label.structured_label.content)
    if payload is None:
        return None
    return _graph_from_payload(payload=payload, model_name=model_name)


def _graph_from_payload(
    payload: dict[str, Any],
    model_name: str,
) -> ParsedVisualGraph:
    graph_source = str(payload.get("graph_source", "") or "").strip().lower()
    if graph_source != "mermaid_fallback":
        graph_source = "model"

    graph = ParsedVisualGraph(
        model_name=model_name,
        graph_source=graph_source,
        node_order_rule=str(
            payload.get("node_order_rule", "top_to_bottom_left_to_right") or "top_to_bottom_left_to_right"
        ).strip(),
        weak_candidate=bool(payload.get("weak_candidate", False) or graph_source == "mermaid_fallback"),
    )

    node_items = payload.get("nodes")
    edge_items = payload.get("edges")
    raw_nodes = node_items if isinstance(node_items, list) else []
    raw_edges = edge_items if isinstance(edge_items, list) else []

    raw_id_map: dict[str, str] = {}
    seen_node_ids: set[str] = set()
    for index, item in enumerate(raw_nodes, start=1):
        node_payload = item if isinstance(item, dict) else {}
        raw_node_id = str(node_payload.get("node_id", "") or "").strip()
        order_index = _coerce_positive_int(node_payload.get("order_index"))
        node_id = _normalize_node_id(raw_node_id, fallback_index=order_index or index)
        if node_id is None:
            graph.parse_errors.append(f"invalid_node_id_at_index:{index}")
            continue
        if order_index is None:
            order_index = _extract_node_index(node_id)
        anchor = VisualNodeAnchor(
            model_name=model_name,
            node_id=node_id,
            order_index=order_index,
            row_index=_coerce_positive_int(node_payload.get("row_index")),
            col_index=_coerce_positive_int(node_payload.get("col_index")),
            bbox_hint=_normalize_bbox_hint(node_payload.get("bbox_hint")),
            shape=_normalize_shape(node_payload.get("shape")),
            text=str(node_payload.get("text", "") or "").strip(),
            graph_source=graph.graph_source,
        )
        if node_id in seen_node_ids:
            graph.warnings.append(f"duplicate_node_id:{model_name}:{node_id}")
            continue
        seen_node_ids.add(node_id)
        graph.nodes.append(anchor)
        if raw_node_id:
            raw_id_map[raw_node_id] = node_id
        raw_id_map[node_id] = node_id
        if order_index is not None:
            raw_id_map[str(order_index)] = node_id

    known_node_ids = {node.node_id for node in graph.nodes}
    for item in raw_edges:
        edge_payload = item if isinstance(item, dict) else {}
        source = _normalize_edge_ref(edge_payload.get("source"), raw_id_map)
        target = _normalize_edge_ref(edge_payload.get("target"), raw_id_map)
        if source is None or target is None:
            graph.parse_errors.append("edge_with_invalid_source_or_target")
            continue
        if source not in known_node_ids or target not in known_node_ids:
            graph.parse_errors.append(f"edge_references_unknown_node:{source}->{target}")
            continue
        graph.edges.append(
            VisualEdgeClaim(
                model_name=model_name,
                source=source,
                target=target,
                label=str(edge_payload.get("label", "") or "").strip(),
                graph_source=graph.graph_source,
            )
        )

    graph.warnings = _deduplicate(graph.warnings)
    graph.parse_errors = _deduplicate(graph.parse_errors)
    return _reindex_graph_by_visual_order(graph)


def _reindex_graph_by_visual_order(graph: ParsedVisualGraph) -> ParsedVisualGraph:
    if not graph.nodes:
        return graph

    depth_lookup = _compute_node_depths(graph)
    rows = _group_nodes_by_visual_rows(graph.nodes, depth_lookup)
    original_to_reindexed: dict[str, str] = {}
    reindexed_nodes: list[VisualNodeAnchor] = []
    original_ids = [node.node_id for node in graph.nodes]
    reordered_ids: list[str] = []

    for row_number, row_nodes in enumerate(rows, start=1):
        ordered_row_nodes = sorted(
            row_nodes,
            key=lambda node: _node_x_sort_key(node=node, depth_lookup=depth_lookup),
        )
        for col_number, node in enumerate(ordered_row_nodes, start=1):
            next_index = len(reindexed_nodes) + 1
            new_id = f"N{next_index:03d}"
            original_to_reindexed[node.node_id] = new_id
            reordered_ids.append(node.node_id)
            reindexed_nodes.append(
                VisualNodeAnchor(
                    model_name=node.model_name,
                    node_id=new_id,
                    order_index=next_index,
                    row_index=row_number,
                    col_index=col_number,
                    bbox_hint=node.bbox_hint,
                    shape=node.shape,
                    text=node.text,
                    graph_source=node.graph_source,
                )
            )

    reindexed_edges: list[VisualEdgeClaim] = []
    warnings = list(graph.warnings)
    parse_errors = list(graph.parse_errors)
    for edge in graph.edges:
        source = original_to_reindexed.get(edge.source)
        target = original_to_reindexed.get(edge.target)
        if source is None or target is None:
            parse_errors.append(
                f"edge_missing_after_visual_reindex:{edge.source}->{edge.target}"
            )
            continue
        reindexed_edges.append(
            VisualEdgeClaim(
                model_name=edge.model_name,
                source=source,
                target=target,
                label=edge.label,
                graph_source=edge.graph_source,
            )
        )

    if original_ids != reordered_ids:
        warnings.append(f"visual_reindex_applied:{graph.model_name}")

    return ParsedVisualGraph(
        model_name=graph.model_name,
        graph_source=graph.graph_source,
        node_order_rule=graph.node_order_rule,
        nodes=reindexed_nodes,
        edges=reindexed_edges,
        warnings=_deduplicate(warnings),
        parse_errors=_deduplicate(parse_errors),
        weak_candidate=graph.weak_candidate,
    )


def _group_nodes_by_visual_rows(
    nodes: list[VisualNodeAnchor],
    depth_lookup: dict[str, int],
) -> list[list[VisualNodeAnchor]]:
    ordered = sorted(
        nodes,
        key=lambda node: _node_visual_sort_key(node=node, depth_lookup=depth_lookup),
    )
    if not ordered:
        return []

    row_tolerance = _row_grouping_tolerance(ordered)
    rows: list[list[VisualNodeAnchor]] = []
    current_row: list[VisualNodeAnchor] = []
    current_row_y: float | None = None
    current_row_bounds: tuple[float, float] | None = None

    for node in ordered:
        raw_y_center, _, height = _node_visual_anchor(node)
        y_center = _node_effective_y(node=node, depth_lookup=depth_lookup)
        tolerance = max(row_tolerance, height * 0.9 if height > 0 else 0.0)
        if (
            current_row
            and current_row_y is not None
            and not _belongs_to_current_row(
                node=node,
                node_effective_y=y_center,
                node_raw_y=raw_y_center,
                current_row_y=current_row_y,
                current_row_bounds=current_row_bounds,
                tolerance=tolerance,
            )
        ):
            rows.append(current_row)
            current_row = [node]
            current_row_y = y_center
            current_row_bounds = _node_vertical_bounds(node)
            continue

        current_row.append(node)
        if current_row_y is None:
            current_row_y = y_center
        else:
            current_row_y = ((current_row_y * (len(current_row) - 1)) + y_center) / len(current_row)
        current_row_bounds = _merge_vertical_bounds(current_row_bounds, _node_vertical_bounds(node))

    if current_row:
        rows.append(current_row)
    return rows


def _row_grouping_tolerance(nodes: list[VisualNodeAnchor]) -> float:
    heights = sorted(
        bbox[3] - bbox[1]
        for bbox in (node.bbox_hint for node in nodes)
        if bbox is not None and len(bbox) == 4
    )
    median_height = heights[len(heights) // 2] if heights else 0.0
    return max(0.045, median_height * 1.1)


def _node_visual_sort_key(
    node: VisualNodeAnchor,
    depth_lookup: dict[str, int],
) -> tuple[float, float, int, str]:
    y_center = _node_effective_y(node=node, depth_lookup=depth_lookup)
    _, x_center, _ = _node_visual_anchor(node)
    return (
        round(y_center, 4),
        round(x_center, 4),
        node.order_index or _extract_node_index(node.node_id) or 10**9,
        node.node_id,
    )


def _node_x_sort_key(
    node: VisualNodeAnchor,
    depth_lookup: dict[str, int],
) -> tuple[float, float, int, str]:
    y_center = _node_effective_y(node=node, depth_lookup=depth_lookup)
    _, x_center, _ = _node_visual_anchor(node)
    return (
        round(x_center, 4),
        round(y_center, 4),
        node.order_index or _extract_node_index(node.node_id) or 10**9,
        node.node_id,
    )


def _node_visual_anchor(node: VisualNodeAnchor) -> tuple[float, float, float]:
    if node.bbox_hint is not None:
        bbox = node.bbox_hint
        return (
            (bbox[1] + bbox[3]) / 2.0,
            (bbox[0] + bbox[2]) / 2.0,
            max(0.0, bbox[3] - bbox[1]),
        )

    row_value = float(node.row_index or node.order_index or _extract_node_index(node.node_id) or 10**6)
    col_value = float(node.col_index or 1)
    return row_value * 0.1, col_value * 0.1, 0.0


def _node_effective_y(
    node: VisualNodeAnchor,
    depth_lookup: dict[str, int],
) -> float:
    raw_y, _, _ = _node_visual_anchor(node)
    depth = depth_lookup.get(node.node_id, 1)
    return raw_y + 0.12 * max(0, depth - 1)


def _compute_node_depths(graph: ParsedVisualGraph) -> dict[str, int]:
    known_node_ids = {node.node_id for node in graph.nodes}
    children: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {node_id: 0 for node_id in known_node_ids}
    for edge in graph.edges:
        if edge.source not in known_node_ids or edge.target not in known_node_ids:
            continue
        if edge.target in children[edge.source]:
            continue
        children[edge.source].add(edge.target)
        indegree[edge.target] += 1

    roots = sorted([node_id for node_id, degree in indegree.items() if degree == 0], key=_node_sort_key)
    if not roots:
        return {node.node_id: node.order_index or index for index, node in enumerate(graph.nodes, start=1)}

    depth_lookup: dict[str, int] = {node_id: 1 for node_id in roots}
    queue = list(roots)
    while queue:
        current = queue.pop(0)
        current_depth = depth_lookup.get(current, 1)
        for child in sorted(children.get(current, set()), key=_node_sort_key):
            next_depth = current_depth + 1
            if child not in depth_lookup:
                depth_lookup[child] = next_depth
                queue.append(child)

    for node in graph.nodes:
        depth_lookup.setdefault(node.node_id, max(1, node.order_index or 1))
    return depth_lookup


def _belongs_to_current_row(
    node: VisualNodeAnchor,
    node_effective_y: float,
    node_raw_y: float,
    current_row_y: float,
    current_row_bounds: tuple[float, float] | None,
    tolerance: float,
) -> bool:
    if abs(node_effective_y - current_row_y) <= tolerance:
        return True
    node_bounds = _node_vertical_bounds(node)
    if current_row_bounds is None or node_bounds is None:
        return False
    current_top, current_bottom = current_row_bounds
    node_top, node_bottom = node_bounds
    overlap = min(current_bottom, node_bottom) - max(current_top, node_top)
    if overlap <= 0:
        return False
    min_height = min(current_bottom - current_top, node_bottom - node_top)
    if min_height <= 0:
        return False
    return (
        overlap / min_height >= 0.35
        and abs(node_effective_y - current_row_y) <= max(tolerance, 0.08)
        and abs(node_raw_y - ((current_top + current_bottom) / 2.0)) <= max(tolerance, 0.08)
    )


def _node_vertical_bounds(node: VisualNodeAnchor) -> tuple[float, float] | None:
    if node.bbox_hint is None:
        return None
    return node.bbox_hint[1], node.bbox_hint[3]


def _merge_vertical_bounds(
    current_bounds: tuple[float, float] | None,
    node_bounds: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if current_bounds is None:
        return node_bounds
    if node_bounds is None:
        return current_bounds
    return min(current_bounds[0], node_bounds[0]), max(current_bounds[1], node_bounds[1])


def _select_fused_node_ids(
    parsed_graphs: list[ParsedVisualGraph],
) -> tuple[list[str], str, list[str], int]:
    num_models = len(parsed_graphs)
    errors: list[str] = []
    support_counter: Counter[str] = Counter()
    node_counts = [len(graph.nodes) for graph in parsed_graphs]
    node_id_sets = [{node.node_id for node in graph.nodes} for graph in parsed_graphs]
    inconsistent_node_count = 0

    for graph, node_id_set in zip(parsed_graphs, node_id_sets):
        if not _has_continuous_node_ids(node_id_set, expected_count=len(graph.nodes)):
            errors.append(f"non_continuous_node_ids:{graph.model_name}")
        for node_id in node_id_set:
            support_counter[node_id] += 1

    if any(count == 0 for count in node_counts):
        return [], "failed", errors, max(node_counts, default=0)

    if num_models == 2:
        first_count, second_count = node_counts
        if first_count != second_count:
            errors.append(
                f"inconsistent_node_count:{parsed_graphs[0].model_name}={first_count},{parsed_graphs[1].model_name}={second_count}"
            )
            inconsistent_node_count = abs(first_count - second_count)
            selected = sorted(node_id_sets[0] & node_id_sets[1], key=_node_sort_key)
            return selected, "ambiguous", errors, inconsistent_node_count
        if node_id_sets[0] != node_id_sets[1]:
            diff = node_id_sets[0] ^ node_id_sets[1]
            errors.append("node_id_set_mismatch")
            inconsistent_node_count = len(diff)
            selected = sorted(node_id_sets[0] & node_id_sets[1], key=_node_sort_key)
            return selected, "ambiguous", errors, inconsistent_node_count
        selected = sorted(node_id_sets[0], key=_node_sort_key)
        return selected, "fused", errors, 0

    min_support_count = _minimum_majority_support(num_models)
    selected = sorted(
        [node_id for node_id, count in support_counter.items() if count >= min_support_count],
        key=_node_sort_key,
    )
    inconsistent_node_count = sum(
        1 for node_id, count in support_counter.items() if count != num_models and node_id in set(selected)
    )
    if max(node_counts) - min(node_counts) >= 2:
        errors.append(
            "inconsistent_node_count_range:"
            + ",".join(f"{graph.model_name}={len(graph.nodes)}" for graph in parsed_graphs)
        )
        return selected, "ambiguous", errors, inconsistent_node_count
    if not selected:
        return [], "failed", errors, inconsistent_node_count
    if not _has_continuous_node_ids(set(selected), expected_count=len(selected)):
        errors.append("non_continuous_selected_node_ids")
        return selected, "ambiguous", errors, inconsistent_node_count
    status = "fused" if inconsistent_node_count == 0 else "partial"
    return selected, status, errors, inconsistent_node_count


def _fuse_nodes(
    parsed_graphs: list[ParsedVisualGraph],
    selected_node_ids: list[str],
    num_models: int,
) -> tuple[list[FusedVisualNode], list[str], list[str], list[str]]:
    node_claims: dict[str, list[VisualNodeAnchor]] = defaultdict(list)
    warnings: list[str] = []

    for graph in parsed_graphs:
        graph_ids = {node.node_id for node in graph.nodes}
        for node in graph.nodes:
            if node.node_id in selected_node_ids:
                node_claims[node.node_id].append(node)
        for node_id in selected_node_ids:
            if node_id not in graph_ids:
                warnings.append(f"missing_node_claim:{graph.model_name}:{node_id}")

    fused_nodes: list[FusedVisualNode] = []
    alignment_errors: list[str] = []
    low_text_nodes: list[str] = []

    for node_id in selected_node_ids:
        anchors = node_claims.get(node_id, [])
        if not anchors:
            continue

        support_models = sorted({anchor.model_name for anchor in anchors})
        text_votes = [anchor.text.strip() for anchor in anchors if anchor.text.strip()]
        shape_votes = [anchor.shape for anchor in anchors if anchor.shape]
        row_votes = [anchor.row_index for anchor in anchors if anchor.row_index is not None]
        col_votes = [anchor.col_index for anchor in anchors if anchor.col_index is not None]
        bbox_hints = [anchor.bbox_hint for anchor in anchors if anchor.bbox_hint is not None]
        order_index = _majority_positive_int(
            [anchor.order_index for anchor in anchors if anchor.order_index is not None]
        ) or _extract_node_index(node_id) or len(fused_nodes) + 1

        text_consistency = compute_text_consistency(text_votes)
        if text_votes and text_consistency < 0.55:
            low_text_nodes.append(node_id)
            warnings.append(f"low_text_consistency_for_same_visual_node:{node_id}")

        node_errors, node_warnings = _inspect_node_alignment(node_id=node_id, anchors=anchors)
        alignment_errors.extend(node_errors)
        warnings.extend(node_warnings)
        fused_nodes.append(
            FusedVisualNode(
                fused_id=node_id,
                order_index=order_index,
                row_index_votes=row_votes,
                col_index_votes=col_votes,
                bbox_hints=bbox_hints,
                shape_votes=shape_votes,
                text_votes=text_votes,
                support_models=support_models,
                support_count=len(support_models),
                confidence=round(len(support_models) / max(1, num_models), 4),
                text_consistency=text_consistency,
                representative_text=_select_representative_text(text_votes),
                representative_shape=_select_representative_shape(shape_votes),
            )
        )

    fused_nodes.sort(key=lambda item: (item.order_index, item.fused_id))
    return (
        fused_nodes,
        _deduplicate(alignment_errors),
        _deduplicate(low_text_nodes),
        _deduplicate(warnings),
    )


def _fuse_edges(
    parsed_graphs: list[ParsedVisualGraph],
    fused_nodes: list[FusedVisualNode],
    num_models: int,
) -> tuple[list[FusedVisualEdge], list[dict[str, Any]], list[str], list[str]]:
    fused_node_ids = {node.fused_id for node in fused_nodes}
    edge_claims: dict[tuple[str, str], list[VisualEdgeClaim]] = defaultdict(list)
    pair_directions: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    edge_alignment_errors: list[str] = []
    warnings: list[str] = []

    for graph in parsed_graphs:
        graph_node_ids = {node.node_id for node in graph.nodes}
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in graph.edges:
            if edge.source not in graph_node_ids or edge.target not in graph_node_ids:
                edge_alignment_errors.append(
                    f"edge_unknown_node:{graph.model_name}:{edge.source}->{edge.target}"
                )
                continue
            if edge.source not in fused_node_ids or edge.target not in fused_node_ids:
                edge_alignment_errors.append(
                    f"edge_references_unfused_node:{graph.model_name}:{edge.source}->{edge.target}"
                )
                continue

            seen_key = (edge.source, edge.target, edge.label.strip())
            if seen_key in seen_edges:
                warnings.append(
                    f"duplicate_edge_claim:{graph.model_name}:{edge.source}->{edge.target}"
                )
                continue
            seen_edges.add(seen_key)
            edge_claims[(edge.source, edge.target)].append(edge)
            pair_directions[tuple(sorted((edge.source, edge.target)))].add(
                (edge.source, edge.target)
            )

    for source_target_pair, directions in pair_directions.items():
        if len(directions) > 1:
            edge_alignment_errors.append(
                f"edge_direction_conflict:{source_target_pair[0]}<->{source_target_pair[1]}"
            )

    min_support_count = _minimum_majority_support(num_models)
    fused_edges: list[FusedVisualEdge] = []
    low_support_edges: list[dict[str, Any]] = []

    for (source, target), claims in sorted(edge_claims.items(), key=lambda item: (_node_sort_key(item[0][0]), _node_sort_key(item[0][1]))):
        support_models = sorted({claim.model_name for claim in claims})
        support_count = len(support_models)
        label_votes = [claim.label.strip() for claim in claims]
        label_consistency = compute_text_consistency(label_votes)
        if label_votes and label_consistency < 0.55:
            warnings.append(f"low_label_consistency_for_visual_edge:{source}->{target}")

        fused_edge = FusedVisualEdge(
            source=source,
            target=target,
            label_votes=label_votes,
            support_models=support_models,
            support_count=support_count,
            confidence=round(support_count / max(1, num_models), 4),
            label_consistency=label_consistency,
            label=_select_representative_text(label_votes, allow_empty=True),
        )
        if support_count >= min_support_count:
            fused_edges.append(fused_edge)
        else:
            low_support_edges.append(
                {
                    "source": source,
                    "target": target,
                    "label_votes": label_votes,
                    "support_models": support_models,
                    "support_count": support_count,
                }
            )

    return (
        fused_edges,
        low_support_edges,
        _deduplicate(edge_alignment_errors),
        _deduplicate(warnings),
    )


def _prune_orphan_low_consistency_nodes(
    fused_nodes: list[FusedVisualNode],
    fused_edges: list[FusedVisualEdge],
    low_support_edges: list[dict[str, Any]],
    low_text_nodes: list[str],
    node_alignment_errors: list[str],
    edge_alignment_errors: list[str],
    warnings: list[str],
    num_models: int,
) -> tuple[
    list[FusedVisualNode],
    list[FusedVisualEdge],
    list[dict[str, Any]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    remaining_nodes = list(fused_nodes)
    remaining_fused_edges = list(fused_edges)
    remaining_low_support_edges = list(low_support_edges)
    remaining_low_text_nodes = list(low_text_nodes)
    remaining_node_alignment_errors = list(node_alignment_errors)
    remaining_edge_alignment_errors = list(edge_alignment_errors)
    remaining_warnings = list(warnings)

    while True:
        fused_edge_degree: Counter[str] = Counter()
        for edge in remaining_fused_edges:
            fused_edge_degree[edge.source] += 1
            fused_edge_degree[edge.target] += 1

        low_support_degree: Counter[str] = Counter()
        for edge in remaining_low_support_edges:
            source = str(edge.get("source", "") or "").strip()
            target = str(edge.get("target", "") or "").strip()
            if source:
                low_support_degree[source] += 1
            if target:
                low_support_degree[target] += 1

        candidate_ids = sorted(
            [
                node.fused_id
                for node in remaining_nodes
                if node.support_count < num_models
                and node.text_consistency < 0.35
                and fused_edge_degree.get(node.fused_id, 0) == 0
                and low_support_degree.get(node.fused_id, 0) > 0
            ],
            key=_node_sort_key,
        )
        if not candidate_ids:
            break

        candidate_set = set(candidate_ids)
        remaining_nodes = [
            node for node in remaining_nodes if node.fused_id not in candidate_set
        ]
        remaining_fused_edges = [
            edge
            for edge in remaining_fused_edges
            if edge.source not in candidate_set and edge.target not in candidate_set
        ]
        remaining_low_support_edges = [
            edge
            for edge in remaining_low_support_edges
            if str(edge.get("source", "") or "").strip() not in candidate_set
            and str(edge.get("target", "") or "").strip() not in candidate_set
        ]
        remaining_low_text_nodes = [
            node_id for node_id in remaining_low_text_nodes if node_id not in candidate_set
        ]
        remaining_node_alignment_errors = [
            entry
            for entry in remaining_node_alignment_errors
            if not _entry_mentions_any_node(entry, candidate_set)
        ]
        remaining_edge_alignment_errors = [
            entry
            for entry in remaining_edge_alignment_errors
            if not _entry_mentions_any_node(entry, candidate_set)
        ]
        remaining_warnings = [
            entry
            for entry in remaining_warnings
            if not _entry_mentions_any_node(entry, candidate_set)
        ]
        remaining_warnings.extend(
            f"pruned_orphan_low_consistency_node:{node_id}" for node_id in candidate_ids
        )

    remaining_nodes.sort(key=lambda item: (item.order_index, item.fused_id))
    return (
        remaining_nodes,
        remaining_fused_edges,
        remaining_low_support_edges,
        _deduplicate(remaining_low_text_nodes),
        _deduplicate(remaining_node_alignment_errors),
        _deduplicate(remaining_edge_alignment_errors),
        _deduplicate(remaining_warnings),
    )


def _inspect_node_alignment(
    node_id: str,
    anchors: list[VisualNodeAnchor],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for index, left in enumerate(anchors):
        for right in anchors[index + 1 :]:
            if left.row_index is not None and right.row_index is not None and abs(left.row_index - right.row_index) > 1:
                warnings.append(
                    f"row_index_conflict:{node_id}:{left.model_name}:{right.model_name}"
                )
            if left.col_index is not None and right.col_index is not None and abs(left.col_index - right.col_index) > 1:
                warnings.append(
                    f"col_index_conflict:{node_id}:{left.model_name}:{right.model_name}"
                )
            if (
                left.shape != "unknown"
                and right.shape != "unknown"
                and left.shape != right.shape
            ):
                warnings.append(
                    f"shape_conflict:{node_id}:{left.model_name}:{right.model_name}"
                )
            if left.bbox_hint is not None and right.bbox_hint is not None:
                center_distance = bbox_center_distance(left.bbox_hint, right.bbox_hint)
                iou = bbox_iou(left.bbox_hint, right.bbox_hint)
                if center_distance > 0.20:
                    warnings.append(
                        f"bbox_center_distance_high:{node_id}:{left.model_name}:{right.model_name}:{center_distance:.3f}"
                    )
                row_gap = (
                    abs(left.row_index - right.row_index)
                    if left.row_index is not None and right.row_index is not None
                    else 0
                )
                col_gap = (
                    abs(left.col_index - right.col_index)
                    if left.col_index is not None and right.col_index is not None
                    else 0
                )
                if center_distance > 0.28 and iou < 0.05 and (row_gap > 1 or col_gap > 1):
                    errors.append(
                        f"node_position_conflict:{node_id}:{left.model_name}:{right.model_name}"
                    )
    return _deduplicate(errors), _deduplicate(warnings)


def _select_representative_text(
    text_votes: list[str],
    allow_empty: bool = False,
) -> str:
    cleaned_votes = [str(text).strip() for text in text_votes]
    if not allow_empty:
        cleaned_votes = [text for text in cleaned_votes if text]
    if not cleaned_votes:
        return ""

    normalized_to_values: dict[str, list[str]] = defaultdict(list)
    for text in cleaned_votes:
        normalized = normalize_vote_text(text)
        normalized_to_values[normalized].append(text)

    ranked = sorted(
        normalized_to_values.items(),
        key=lambda item: (
            -len(item[1]),
            -max(_visible_text_score(value) for value in item[1]),
            -max(len(value) for value in item[1]),
            item[0],
        ),
    )
    best_values = ranked[0][1]
    return max(best_values, key=lambda value: (_visible_text_score(value), len(value), value))


def _select_representative_shape(shape_votes: list[str]) -> str:
    cleaned_votes = [shape for shape in shape_votes if shape in VALID_SHAPES]
    if not cleaned_votes:
        return "unknown"

    counter = Counter(cleaned_votes)
    preferred = sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            item[0] == "unknown",
            item[0],
        ),
    )
    return preferred[0][0]


def _compute_graph_confidence(
    result: FusedGraphResult,
    num_models: int,
) -> float:
    if not result.nodes:
        return 0.0

    node_support = _average([node.confidence for node in result.nodes], default=0.0)
    edge_support = _average([edge.confidence for edge in result.edges], default=0.0)
    text_consistency = _average([node.text_consistency for node in result.nodes], default=1.0)
    label_consistency = _average([edge.label_consistency for edge in result.edges], default=1.0)

    base = (
        0.45 * node_support
        + 0.25 * edge_support
        + 0.20 * text_consistency
        + 0.10 * label_consistency
    )

    penalty = 0.0
    if result.fusion_method == "mermaid_fallback":
        penalty += 0.18
    if result.fusion_status == "partial":
        penalty += 0.08
    elif result.fusion_status == "ambiguous":
        penalty += 0.18
    elif result.fusion_status == "failed":
        penalty += 0.35

    penalty += min(0.18, 0.03 * len(result.node_alignment_errors) + 0.03 * len(result.edge_alignment_errors))
    penalty += min(0.12, 0.04 * len(result.low_support_edges))
    penalty += min(0.10, 0.03 * len(result.low_text_consistency_nodes))
    penalty += min(0.15, 0.05 * len(result.critical_errors))
    if not result.edges:
        penalty += 0.08
    if num_models >= 3 and result.inconsistent_node_count > 0:
        penalty += min(0.10, 0.03 * result.inconsistent_node_count)

    return round(max(0.0, base - penalty), 4)


def _minimum_majority_support(num_models: int) -> int:
    if num_models <= 2:
        return num_models
    return math.ceil((2 * num_models) / 3)


def normalize_vote_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("<br/>", " ").replace("<br />", " ").replace("<br>", " ")
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _bigram_jaccard(left: str, right: str) -> float:
    left_bigrams = _char_bigrams(left)
    right_bigrams = _char_bigrams(right)
    if not left_bigrams and not right_bigrams:
        return 1.0
    if not left_bigrams or not right_bigrams:
        return 0.0
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _visible_text_score(text: str) -> int:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9±/%\-()]+", normalize_vote_text(text))
    return sum(len(token) for token in tokens)


def _majority_positive_int(values: list[int]) -> int | None:
    if not values:
        return None
    counter = Counter(values)
    highest = counter.most_common(1)[0][1]
    for value in values:
        if counter[value] == highest:
            return value
    return values[0]


def _has_continuous_node_ids(node_ids: set[str], expected_count: int) -> bool:
    if len(node_ids) != expected_count or expected_count <= 0:
        return False
    numeric_values = [_extract_node_index(node_id) for node_id in node_ids]
    if any(value is None for value in numeric_values):
        return False
    numeric_values = sorted(value for value in numeric_values if value is not None)
    return numeric_values == list(range(1, expected_count + 1))


def _normalize_node_id(raw_value: Any, fallback_index: int | None = None) -> str | None:
    raw_text = str(raw_value or "").strip()
    matches = re.findall(r"\d+", raw_text)
    if matches:
        return f"N{int(matches[0]):03d}"
    if fallback_index is not None and fallback_index > 0:
        return f"N{fallback_index:03d}"
    return None


def _normalize_edge_ref(raw_value: Any, raw_id_map: dict[str, str]) -> str | None:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return None
    if raw_text in raw_id_map:
        return raw_id_map[raw_text]
    return _normalize_node_id(raw_text)


def _extract_node_index(node_id: str) -> int | None:
    matches = re.findall(r"\d+", str(node_id or ""))
    if not matches:
        return None
    return int(matches[0])


def _normalize_shape(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_SHAPES:
        return normalized
    return "unknown"


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


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _node_sort_key(node_id: str) -> tuple[int, str]:
    return (_extract_node_index(node_id) or 10**9, node_id)


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


def _clean_mermaid_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("%%"):
        return ""
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in STYLE_PREFIXES):
        return ""
    return DUPLICATE_CLASS_RE.sub("", stripped).strip()


def _split_mermaid_segments(line: str) -> list[str]:
    if ";" not in line:
        return [line.strip()] if line.strip() else []

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
            elif char == ";" and square_depth == 0 and round_depth == 0 and curly_depth == 0:
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
    node_lookup: dict[str, dict[str, str]],
    node_order: list[str],
) -> list[tuple[str, str, str]]:
    first = _parse_node_token(segment, 0)
    if first is None:
        return []

    source_id, source_text, source_shape, position = first
    _register_mermaid_node(
        node_lookup=node_lookup,
        node_order=node_order,
        raw_id=source_id,
        text=source_text,
        shape=source_shape,
    )
    previous_id = source_id
    edges: list[tuple[str, str, str]] = []

    while True:
        arrow_match = _parse_arrow(segment, position)
        if arrow_match is None:
            break
        label, position = arrow_match
        next_node = _parse_node_token(segment, position)
        if next_node is None:
            return edges
        target_id, target_text, target_shape, position = next_node
        _register_mermaid_node(
            node_lookup=node_lookup,
            node_order=node_order,
            raw_id=target_id,
            text=target_text,
            shape=target_shape,
        )
        edges.append((previous_id, target_id, label))
        previous_id = target_id
    return edges


def _parse_node_token(
    segment: str,
    position: int,
) -> tuple[str, str, str, int] | None:
    while position < len(segment) and segment[position].isspace():
        position += 1
    match = NODE_TOKEN_RE.match(segment, position)
    if match is None:
        return None

    raw_id = match.group("id")
    if match.group("square") is not None:
        raw_text = match.group("square")
        shape = "rectangle"
    elif match.group("round") is not None:
        raw_text = match.group("round")
        shape = "ellipse"
    elif match.group("curly") is not None:
        raw_text = match.group("curly")
        shape = "diamond"
    else:
        raw_text = raw_id
        shape = "unknown"

    text = _strip_mermaid_wrappers(str(raw_text or "").strip()) or raw_id
    return raw_id, text, shape, match.end()


def _parse_arrow(segment: str, position: int) -> tuple[str, int] | None:
    for pattern in (PIPE_ARROW_RE, TEXT_ARROW_RE, DOTTED_ARROW_RE, THICK_ARROW_RE, PLAIN_ARROW_RE):
        match = pattern.match(segment, position)
        if match is None:
            continue
        label = str(match.groupdict().get("label", "") or "").strip()
        return label, match.end()
    return None


def _register_mermaid_node(
    node_lookup: dict[str, dict[str, str]],
    node_order: list[str],
    raw_id: str,
    text: str,
    shape: str,
) -> None:
    if raw_id not in node_lookup:
        node_lookup[raw_id] = {
            "text": text or raw_id,
            "shape": shape or "unknown",
        }
        node_order.append(raw_id)
        return

    existing = node_lookup[raw_id]
    if _visible_text_score(text) > _visible_text_score(existing["text"]):
        existing["text"] = text or raw_id
    if existing["shape"] == "unknown" and shape != "unknown":
        existing["shape"] = shape


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


def _average(values: list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _escape_mermaid_text(text: str) -> str:
    return str(text or "").replace('"', "&quot;")


def _escape_mermaid_label(text: str) -> str:
    return _escape_mermaid_text(text).replace("|", "/")


def _entry_mentions_any_node(entry: str, node_ids: set[str]) -> bool:
    for node_id in node_ids:
        if (
            f":{node_id}:" in entry
            or entry.endswith(f":{node_id}")
            or f":{node_id}->" in entry
            or entry.endswith(f"->{node_id}")
            or f"<->{node_id}" in entry
        ):
            return True
    return False


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
