from __future__ import annotations

from collections import Counter

from src.graph_fusion import FusedGraphResult
from src.schema import ConsensusResult, ModelOutput, ParsedLabel
from src.validators.validation import ValidationResult


def decide_consensus(
    image_id: str,
    labels: list[ParsedLabel],
    model_outputs: list[ModelOutput],
    score_result: dict[str, float | list[str] | str],
    validation_result: ValidationResult,
    graph_fusion_result: FusedGraphResult | None = None,
) -> ConsensusResult:
    reasons = list(score_result.get("reasons", []))
    validation_errors = list(validation_result.critical_errors)
    validation_warnings = list(validation_result.warnings)
    escalation_reasons: list[str] = []
    success_count = sum(1 for output in model_outputs if output.success)
    total_models = len(model_outputs)
    parsed_count = len(labels)

    type_agreement = float(score_result.get("type_agreement", 0.0))
    caption_agreement = float(score_result.get("caption_agreement", 0.0))
    structure_agreement = float(score_result.get("structure_agreement", 0.0))
    overall_score = float(score_result.get("overall_score", 0.0))
    evidence_score = float(validation_result.evidence_score)
    validator_score = float(validation_result.validator_score)
    hallucination_risk = float(validation_result.hallucination_risk)

    accept_score = round(
        0.35 * overall_score
        + 0.30 * evidence_score
        + 0.25 * validator_score
        + 0.10 * (1 - hallucination_risk),
        4,
    )

    if success_count == 0:
        decision = "failed"
        reasons.insert(0, "no models succeeded")
    elif not labels:
        decision = "failed"
        reasons.insert(0, "no parsable labels")
    elif parsed_count == 1:
        decision = "review"
        reasons.append("single model result cannot be auto-accepted")
        escalation_reasons.append("single_model_result")
    else:
        decision = "accepted" if _passes_acceptance_gate(
            total_models=total_models,
            overall_score=overall_score,
            type_agreement=type_agreement,
            structure_agreement=structure_agreement,
            evidence_score=evidence_score,
            validator_score=validator_score,
            hallucination_risk=hallucination_risk,
            validation_errors=validation_errors,
        ) else "review"

    majority_type = _majority_value([label.image_type for label in labels])
    majority_kind = _majority_value([label.structured_label.kind for label in labels])

    if total_models > 0 and success_count < (total_models / 2):
        reasons.append("less than half models succeeded")
        escalation_reasons.append("less_than_half_models_succeeded")

    if majority_type == "flowchart" and decision != "failed":
        if graph_fusion_result is None:
            decision = "review"
            reasons.append("flowchart result cannot be auto-accepted without graph fusion")
            escalation_reasons.append("flowchart_without_graph_fusion")
        else:
            if graph_fusion_result.fusion_method == "mermaid_fallback":
                decision = "review"
                reasons.append("graph fusion used mermaid fallback instead of visual flowchart_graph")
                escalation_reasons.append("flowchart_graph_missing_used_mermaid_fallback")
            elif graph_fusion_result.fusion_method != "visual_order":
                decision = "review"
                reasons.append("graph fusion method is not visual_order")
                escalation_reasons.append("unsupported_graph_fusion_method")
            if graph_fusion_result.fusion_status == "ambiguous":
                decision = "review"
                reasons.append("visual graph alignment is ambiguous")
                escalation_reasons.append("ambiguous_visual_graph_alignment")
            elif graph_fusion_result.fusion_status == "partial":
                decision = "review"
                reasons.append("visual graph fusion is only partial")
                escalation_reasons.append("partial_visual_graph_alignment")
            elif graph_fusion_result.fusion_status == "failed":
                decision = "review"
                reasons.append("visual graph fusion failed")
                escalation_reasons.append("graph_fusion_failed")
            if graph_fusion_result.graph_confidence < 0.70:
                decision = "review"
                reasons.append("graph fusion confidence below acceptance threshold")
                escalation_reasons.append("low_graph_confidence")
            if graph_fusion_result.inconsistent_node_count > 0 or _has_inconsistent_node_count(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion has inconsistent node count across models")
                escalation_reasons.append("inconsistent_node_count")
            if _has_hard_node_alignment_errors(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains node alignment errors")
                escalation_reasons.append("node_alignment_errors")
            if graph_fusion_result.edge_alignment_errors:
                decision = "review"
                reasons.append("graph fusion contains edge alignment errors")
                escalation_reasons.append("edge_alignment_errors")
            if not graph_fusion_result.edges:
                decision = "review"
                reasons.append("fused graph has no edges")
                escalation_reasons.append("empty_fused_edges")
            if _has_many_low_support_edges(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains low-support edges")
                escalation_reasons.append("low_support_edges")
            if graph_fusion_result.critical_errors:
                decision = "review"
                reasons.extend(graph_fusion_result.critical_errors)
                escalation_reasons.append("graph_fusion_critical_errors")

    if decision == "review":
        thresholds = _thresholds_for_model_count(total_models=total_models)
        if overall_score < thresholds["overall_score"]:
            reasons.append("overall consensus score below acceptance threshold")
            escalation_reasons.append("low_consensus_score")
        if type_agreement < thresholds["type_agreement"]:
            reasons.append("image type agreement below acceptance threshold")
            escalation_reasons.append("low_type_agreement")
        if structure_agreement < thresholds["structure_agreement"]:
            reasons.append("structure agreement below acceptance threshold")
            escalation_reasons.append("low_structure_agreement")
        if evidence_score < thresholds["evidence_score"]:
            reasons.append("evidence score below acceptance threshold")
            escalation_reasons.append("low_evidence_score")
        if validator_score < thresholds["validator_score"]:
            reasons.append("validator score below acceptance threshold")
            escalation_reasons.append("low_validator_score")
        if hallucination_risk > thresholds["hallucination_risk"]:
            reasons.append("hallucination risk above acceptance threshold")
            escalation_reasons.append("high_hallucination_risk")
        if validation_errors:
            reasons.extend(validation_errors)
            escalation_reasons.append("critical_validation_errors")
        if total_models == 2 and (
            type_agreement < 1.0
            or structure_agreement < 0.75
            or overall_score < 0.85
        ):
            escalation_reasons.append("two_model_disagreement")

    return ConsensusResult(
        image_id=image_id,
        type_agreement=type_agreement,
        caption_agreement=caption_agreement,
        structure_agreement=structure_agreement,
        overall_score=overall_score,
        evidence_score=evidence_score,
        validator_score=validator_score,
        hallucination_risk=hallucination_risk,
        accept_score=accept_score,
        decision=decision,
        reasons=_deduplicate(reasons),
        validation_errors=_deduplicate(validation_errors),
        validation_warnings=_deduplicate(validation_warnings),
        escalation_reasons=_deduplicate(escalation_reasons),
    )


def _passes_acceptance_gate(
    total_models: int,
    overall_score: float,
    type_agreement: float,
    structure_agreement: float,
    evidence_score: float,
    validator_score: float,
    hallucination_risk: float,
    validation_errors: list[str],
) -> bool:
    if total_models <= 1:
        return False

    thresholds = _thresholds_for_model_count(total_models=total_models)
    return bool(
        overall_score >= thresholds["overall_score"]
        and type_agreement >= thresholds["type_agreement"]
        and structure_agreement >= thresholds["structure_agreement"]
        and evidence_score >= thresholds["evidence_score"]
        and validator_score >= thresholds["validator_score"]
        and hallucination_risk <= thresholds["hallucination_risk"]
        and not validation_errors
    )


def _thresholds_for_model_count(total_models: int) -> dict[str, float]:
    if total_models == 2:
        return {
            "overall_score": 0.85,
            "type_agreement": 1.0,
            "structure_agreement": 0.75,
            "evidence_score": 0.65,
            "validator_score": 0.75,
            "hallucination_risk": 0.30,
        }
    return {
        "overall_score": 0.80,
        "type_agreement": 2.0 / 3.0,
        "structure_agreement": 0.70,
        "evidence_score": 0.60,
        "validator_score": 0.70,
        "hallucination_risk": 0.35,
    }


def _majority_value(values: list[str]) -> str:
    if not values:
        return "unknown"
    counter = Counter(values)
    majority_count = counter.most_common(1)[0][1]
    for value in values:
        if counter[value] == majority_count:
            return value
    return values[0]


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _has_many_low_support_edges(graph_fusion_result: FusedGraphResult) -> bool:
    low_support_count = len(graph_fusion_result.low_support_edges)
    if low_support_count == 0:
        return False

    fused_edge_count = len(graph_fusion_result.edges)
    if fused_edge_count == 0:
        return True

    total_edge_claims = fused_edge_count + low_support_count
    if low_support_count >= fused_edge_count:
        return True
    if low_support_count >= 3 and (low_support_count / total_edge_claims) >= 0.35:
        return True
    return False


def _has_inconsistent_node_count(graph_fusion_result: FusedGraphResult) -> bool:
    return any(
        error.startswith("inconsistent_node_count:")
        or error.startswith("inconsistent_node_count_range:")
        for error in graph_fusion_result.node_alignment_errors
    )


def _has_hard_node_alignment_errors(graph_fusion_result: FusedGraphResult) -> bool:
    return any(
        error.startswith("node_position_conflict:")
        or error in {"node_id_set_mismatch", "non_continuous_selected_node_ids"}
        or error.startswith("non_continuous_node_ids:")
        for error in graph_fusion_result.node_alignment_errors
    )
