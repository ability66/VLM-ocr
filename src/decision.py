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
            if graph_fusion_result.graph_confidence < 0.75:
                decision = "review"
                reasons.append("graph fusion confidence below acceptance threshold")
                escalation_reasons.append("low_graph_confidence")
            if _has_graph_parse_errors(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains mermaid parse errors")
                escalation_reasons.append("graph_parse_errors")
            if _has_edge_disagreement(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains edge disagreement")
                escalation_reasons.append("edge_disagreement")
            if _has_duplicate_node_texts(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains duplicate node texts that are unsafe to auto-merge")
                escalation_reasons.append("duplicate_node_texts")
            if not graph_fusion_result.edges:
                decision = "review"
                reasons.append("fused graph has no edges")
                escalation_reasons.append("empty_fused_edges")
            if _has_many_low_support_edges(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains many low-support edges")
                escalation_reasons.append("low_support_edges")
            if _has_many_unsupported_graph_claims(graph_fusion_result):
                decision = "review"
                reasons.append("graph fusion contains too many unsupported graph claims")
                escalation_reasons.append("unsupported_graph_claims")
            if graph_fusion_result.critical_errors:
                remaining_errors = [
                    error
                    for error in graph_fusion_result.critical_errors
                    if error
                    not in {
                        "mermaid_parse_errors_present",
                        "edge_disagreement",
                        "duplicate_node_texts_present",
                    }
                ]
                if remaining_errors:
                    decision = "review"
                    reasons.extend(remaining_errors)
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
    low_support_count = sum(1 for edge in graph_fusion_result.edges if edge.support_count == 1)
    if low_support_count == 0:
        return False
    edge_count = len(graph_fusion_result.edges)
    low_support_ratio = low_support_count / edge_count if edge_count else 0.0
    return bool(
        (low_support_count >= 5 and low_support_ratio >= 0.25)
        or low_support_ratio >= 0.45
    )


def _has_many_unsupported_graph_claims(graph_fusion_result: FusedGraphResult) -> bool:
    warnings = list(graph_fusion_result.warnings)
    unsupported_node_count = sum(
        1 for warning in warnings if warning.startswith("unsupported_node:")
    )
    unsupported_edge_node_count = sum(
        1 for warning in warnings if warning.startswith("unsupported_edge_nodes:")
    )
    total_unsupported = unsupported_node_count + unsupported_edge_node_count
    return total_unsupported >= 4 or unsupported_edge_node_count >= 3


def _has_graph_parse_errors(graph_fusion_result: FusedGraphResult) -> bool:
    return "mermaid_parse_errors_present" in set(graph_fusion_result.critical_errors)


def _has_edge_disagreement(graph_fusion_result: FusedGraphResult) -> bool:
    return "edge_disagreement" in set(graph_fusion_result.critical_errors)


def _has_duplicate_node_texts(graph_fusion_result: FusedGraphResult) -> bool:
    return "duplicate_node_texts_present" in set(graph_fusion_result.critical_errors)
