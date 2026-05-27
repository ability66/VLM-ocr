from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.graph_fusion import FusedGraphResult
from src.schema import (
    CaptionStructured,
    ConsensusResult,
    ImageTask,
    ModelOutput,
    ParsedLabel,
    StructuredLabel,
)
from src.validators.validation import ValidationResult


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    per_image_dir = output_dir / "per_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, per_image_dir


def clear_previous_outputs(output_dir: Path) -> None:
    _, per_image_dir = ensure_output_dirs(output_dir)
    for file_path in per_image_dir.glob("*.json"):
        file_path.unlink()
    summary_path = output_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()


def initialize_summary_file(output_dir: Path) -> Path:
    summary_path = output_dir / "summary.jsonl"
    summary_path.write_text("", encoding="utf-8")
    return summary_path


def write_image_result(
    output_dir: Path,
    image_task: ImageTask,
    model_outputs: list[ModelOutput],
    normalized_results: list[ParsedLabel | None],
    consensus: ConsensusResult,
    validation_result: ValidationResult,
    graph_fusion_result: FusedGraphResult | None = None,
) -> dict[str, Any]:
    _, per_image_dir = ensure_output_dirs(output_dir)
    final_label = build_final_label(
        normalized_results=normalized_results,
        model_outputs=model_outputs,
        graph_fusion_result=graph_fusion_result,
    )
    final_label_status = _final_label_status(consensus.decision)
    normalized_labels = [label.model_dump() for label in normalized_results if label is not None]
    graph_fusion = build_graph_fusion_payload(
        normalized_results=normalized_results,
        model_outputs=model_outputs,
        graph_fusion_result=graph_fusion_result,
    )

    record = {
        "image": image_task.model_dump(),
        "model_outputs": [output.model_dump() for output in model_outputs],
        "normalized_labels": normalized_labels,
        "validation": validation_result.to_dict(),
        "consensus": consensus.model_dump(),
        "final_label": final_label,
        "final_label_status": final_label_status,
        "graph_fusion": graph_fusion,
    }

    output_path = per_image_dir / f"{image_task.image_id}.json"
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def append_summary_record(summary_path: Path, summary_record: dict[str, Any]) -> None:
    with summary_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary_record, ensure_ascii=False))
        file.write("\n")


def build_summary_record(
    image_task: ImageTask,
    model_outputs: list[ModelOutput],
    consensus: ConsensusResult,
    final_label: dict[str, Any],
    final_label_status: str,
    graph_fusion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    num_success = sum(1 for output in model_outputs if output.success)
    vendor_values = [output.vendor for output in model_outputs]
    source_type_values = [output.source_type for output in model_outputs]
    vendors = _summarize_optional_values(vendor_values)
    source_types = _summarize_optional_values(source_type_values)
    graph_fusion = graph_fusion or {"enabled": False}
    node_vote_details = list(graph_fusion.get("node_vote_details", []))
    edge_vote_details = list(graph_fusion.get("edge_vote_details", []))
    low_support_edges = list(graph_fusion.get("low_support_edges", []))
    low_text_nodes = list(graph_fusion.get("low_text_consistency_nodes", []))
    return {
        "image_id": image_task.image_id,
        "file_name": image_task.file_name,
        "decision": consensus.decision,
        "overall_score": consensus.overall_score,
        "evidence_score": consensus.evidence_score,
        "validator_score": consensus.validator_score,
        "hallucination_risk": consensus.hallucination_risk,
        "accept_score": consensus.accept_score,
        "image_type": final_label.get("image_type", "unknown"),
        "structure_kind": final_label.get("structured_label", {}).get("kind", "none"),
        "final_label_status": final_label_status,
        "num_models": len(model_outputs),
        "num_success": num_success,
        "vendor": vendors,
        "source_type": source_types,
        "model_names": [output.model_name for output in model_outputs],
        "reasons": consensus.reasons,
        "validation_errors": consensus.validation_errors,
        "validation_warnings": consensus.validation_warnings,
        "escalation_reasons": consensus.escalation_reasons,
        "graph_fusion_enabled": bool(graph_fusion.get("enabled", False)),
        "graph_fusion_method": graph_fusion.get("fusion_method", "none"),
        "graph_fusion_status": graph_fusion.get("fusion_status", "failed"),
        "graph_confidence": float(graph_fusion.get("graph_confidence", 0.0) or 0.0),
        "fused_node_count": int(graph_fusion.get("fused_node_count", len(node_vote_details)) or 0),
        "fused_edge_count": int(graph_fusion.get("fused_edge_count", len(edge_vote_details)) or 0),
        "inconsistent_node_count": int(graph_fusion.get("inconsistent_node_count", 0) or 0),
        "low_support_node_count": sum(
            1 for node in node_vote_details if int(node.get("support_count", 0) or 0) == 1
        ),
        "low_support_edge_count": len(low_support_edges),
        "low_text_consistency_node_count": len(low_text_nodes),
        "graph_fusion_warnings": list(graph_fusion.get("warnings", [])),
        "graph_fusion_errors": list(graph_fusion.get("critical_errors", [])),
    }


def build_final_label(
    normalized_results: list[ParsedLabel | None],
    model_outputs: list[ModelOutput],
    graph_fusion_result: FusedGraphResult | None = None,
) -> dict[str, Any]:
    paired = [
        (output, label)
        for output, label in zip(model_outputs, normalized_results)
        if label is not None
    ]
    if not paired:
        structured = StructuredLabel(
            kind="text",
            content="",
            format="plain_text",
            source="none",
        )
        caption_structured = CaptionStructured(visual_type="unknown")
        return {
            "image_type": "unknown",
            "caption": "",
            "caption_structured": caption_structured.model_dump(),
            "structured_label": structured.model_dump(),
        }

    labels = [label for _, label in paired]
    majority_type = _majority_choice([label.image_type for label in labels], default="unknown")
    caption = _select_caption(paired=paired, majority_type=majority_type)
    caption_structured = _select_caption_structured(
        paired=paired,
        majority_type=majority_type,
    )

    if (
        majority_type == "flowchart"
        and graph_fusion_result is not None
        and graph_fusion_result.mermaid.strip()
    ):
        structured = StructuredLabel(
            kind="mermaid",
            content=graph_fusion_result.mermaid,
            format="mermaid",
            source="fused_graph",
            graph_confidence=graph_fusion_result.graph_confidence,
        )
    else:
        majority_kind = _majority_choice(
            [label.structured_label.kind for label in labels], default="none"
        )
        structured = _select_structured_label(
            paired=paired,
            majority_type=majority_type,
            majority_kind=majority_kind,
        )

    return {
        "image_type": majority_type,
        "caption": caption,
        "caption_structured": caption_structured.model_dump(),
        "structured_label": structured.model_dump(),
    }


def _select_caption(
    paired: list[tuple[ModelOutput, ParsedLabel]], majority_type: str
) -> str:
    for output, label in paired:
        if output.success and label.image_type == majority_type and label.caption.strip():
            return label.caption.strip()
        if (
            output.success
            and label.image_type == majority_type
            and label.caption_structured.brief.strip()
        ):
            return label.caption_structured.brief.strip()
    for _, label in paired:
        if label.caption.strip():
            return label.caption.strip()
        if label.caption_structured.brief.strip():
            return label.caption_structured.brief.strip()
    return ""


def _select_caption_structured(
    paired: list[tuple[ModelOutput, ParsedLabel]],
    majority_type: str,
) -> CaptionStructured:
    primary_candidates = [
        label.caption_structured
        for output, label in paired
        if output.success and label.image_type == majority_type
    ]
    fallback_candidates = [label.caption_structured for _, label in paired]

    for candidates in (primary_candidates, fallback_candidates):
        selected = _best_caption_structured(candidates)
        if selected is not None:
            return selected

    return CaptionStructured(visual_type=majority_type)


def _select_structured_label(
    paired: list[tuple[ModelOutput, ParsedLabel]],
    majority_type: str,
    majority_kind: str,
) -> StructuredLabel:
    for _, label in paired:
        struct = label.structured_label
        if struct.kind == majority_kind and struct.content.strip():
            return struct

    if majority_type == "natural_image":
        return StructuredLabel(kind="none", content="", format="none", source="none")

    fallback_content = ""
    for _, label in paired:
        content = label.structured_label.content.strip()
        if content:
            fallback_content = content
            break

    return StructuredLabel(kind="text", content=fallback_content, format="plain_text")


def build_graph_fusion_payload(
    normalized_results: list[ParsedLabel | None],
    model_outputs: list[ModelOutput],
    graph_fusion_result: FusedGraphResult | None = None,
) -> dict[str, Any]:
    paired_labels = [
        label
        for label in normalized_results
        if label is not None
    ]
    majority_type = _majority_choice(
        [label.image_type for label in paired_labels], default="unknown"
    )
    mermaid_count = sum(
        1
        for label in paired_labels
        if (
            (isinstance(label.flowchart_graph, dict) and label.flowchart_graph.get("nodes"))
            or (label.structured_label.kind == "mermaid" and label.structured_label.content.strip())
        )
    )
    del model_outputs

    base_payload = {
        "enabled": False,
        "fusion_method": "none",
        "fusion_status": "failed",
        "graph_confidence": 0.0,
        "fused_node_count": 0,
        "fused_edge_count": 0,
        "inconsistent_node_count": 0,
        "node_alignment_errors": [],
        "edge_alignment_errors": [],
        "low_support_edges": [],
        "low_text_consistency_nodes": [],
        "warnings": [],
        "critical_errors": [],
        "node_vote_details": [],
        "edge_vote_details": [],
    }

    if majority_type != "flowchart":
        base_payload["reason"] = "not flowchart"
        return base_payload
    if mermaid_count < 2 or graph_fusion_result is None:
        base_payload["reason"] = "not enough flowchart graph outputs"
        return base_payload

    return {
        "enabled": True,
        "fusion_method": graph_fusion_result.fusion_method,
        "fusion_status": graph_fusion_result.fusion_status,
        "graph_confidence": graph_fusion_result.graph_confidence,
        "fused_node_count": len(graph_fusion_result.nodes),
        "fused_edge_count": len(graph_fusion_result.edges),
        "inconsistent_node_count": graph_fusion_result.inconsistent_node_count,
        "node_alignment_errors": list(graph_fusion_result.node_alignment_errors),
        "edge_alignment_errors": list(graph_fusion_result.edge_alignment_errors),
        "low_support_edges": list(graph_fusion_result.low_support_edges),
        "low_text_consistency_nodes": list(graph_fusion_result.low_text_consistency_nodes),
        "node_vote_details": list(graph_fusion_result.node_vote_details),
        "edge_vote_details": list(graph_fusion_result.edge_vote_details),
        "warnings": list(graph_fusion_result.warnings),
        "critical_errors": list(graph_fusion_result.critical_errors),
    }


def _best_caption_structured(candidates: list[CaptionStructured]) -> CaptionStructured | None:
    best: CaptionStructured | None = None
    best_score = -1
    for candidate in candidates:
        score = _caption_structured_score(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    if best is None or best_score <= 0:
        return None
    return best


def _caption_structured_score(value: CaptionStructured) -> int:
    return (
        int(bool(value.brief.strip()))
        + int(bool(value.visual_type.strip()))
        + int(bool(value.main_subject.strip()))
        + int(bool(value.visible_title.strip()))
        + min(len(value.key_visible_text), 3)
        + int(bool(value.structure_summary.strip()))
    )


def _majority_choice(values: list[str], default: str) -> str:
    if not values:
        return default
    counter = Counter(values)
    highest = counter.most_common(1)[0][1]
    for value in values:
        if counter[value] == highest:
            return value
    return values[0]


def _summarize_optional_values(values: list[str | None]) -> str | list[str | None] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _final_label_status(decision: str) -> str:
    if decision == "accepted":
        return "accepted"
    if decision == "review":
        return "candidate"
    return "failed"
