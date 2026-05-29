from __future__ import annotations

from collections import Counter

from src.schema import ParsedLabel
from src.validators.caption_validator import validate_caption_structured
from src.validators.evidence import (
    calculate_text_evidence_score,
    collect_evidence_texts,
    collect_ocr_region_role_pools,
    extract_candidate_texts_from_label,
    extract_special_ocr_region_texts,
    normalize_text,
)
from src.validators.mermaid_validator import validate_mermaid
from src.validators.table_validator import validate_table
from src.validators.validation import ValidationResult
from src.seal_utils import is_stamp_mode, primary_seal_signature, primary_seal_text


def validate_labels(image_id: str, labels: list[ParsedLabel]) -> ValidationResult:
    if not labels:
        return ValidationResult(
            image_id=image_id,
            evidence_score=0.0,
            validator_score=0.0,
            hallucination_risk=0.0,
            critical_errors=[],
            warnings=[],
            details={"label_count": 0, "evidence_text_pool": [], "per_label": []},
        )

    evidence_texts = collect_evidence_texts(labels)
    ocr_region_role_pools = collect_ocr_region_role_pools(labels)
    critical_errors: list[str] = []
    warnings: list[str] = []
    evidence_scores: list[float] = []
    validator_scores: list[float] = []
    hallucination_risks: list[float] = []
    per_label_details: list[dict] = []

    majority_type = _majority_value([label.image_type for label in labels], default="unknown")
    majority_kind = _majority_value(
        [label.structured_label.kind for label in labels], default="none"
    )
    stamp_mode = is_stamp_mode(labels)

    if majority_type == "flowchart" and majority_kind != "mermaid":
        critical_errors.append(
            "majority image_type is flowchart but structured output is not mermaid"
        )
    if majority_type in {"chart", "table"} and majority_kind != "table":
        critical_errors.append(
            "majority image_type is chart/table but structured output is not table"
        )

    ocr_errors, ocr_warnings, ocr_details = _validate_ocr_region_consensus(labels)
    critical_errors.extend(ocr_errors)
    warnings.extend(ocr_warnings)

    for index, label in enumerate(labels):
        label_detail = {
            "index": index,
            "image_type": label.image_type,
            "structured_kind": label.structured_label.kind,
            "structured_format": label.structured_label.format,
            "model_warnings": list(label.warnings),
        }

        label_errors: list[str] = []
        label_warnings: list[str] = list(label.warnings)

        try:
            caption_result = validate_caption_structured(label, evidence_texts)
        except Exception as exc:
            caption_result = {
                "errors": [],
                "warnings": [f"caption validator error: {type(exc).__name__}: {exc}"],
                "score": 0.0,
                "unsupported_fields": [],
            }

        label_errors.extend(caption_result["errors"])
        label_warnings.extend(caption_result["warnings"])

        candidate_texts: list[str] = []
        try:
            candidate_texts = extract_candidate_texts_from_label(label)
            evidence_score, hallucination_risk, evidence_detail = calculate_text_evidence_score(
                candidate_texts=candidate_texts,
                evidence_texts=evidence_texts,
            )
        except Exception as exc:
            evidence_score = 0.0
            hallucination_risk = 1.0
            evidence_detail = {
                "candidate_count": 0,
                "supported_count": 0,
                "unsupported_count": 0,
                "supported_matches": [],
                "unsupported_texts": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            label_warnings.append(f"evidence validator error: {type(exc).__name__}: {exc}")

        try:
            validator_result = _validate_structured_label(label, stamp_mode=stamp_mode)
        except Exception as exc:
            validator_result = {
                "valid_basic": False,
                "errors": [],
                "warnings": [f"structured validator error: {type(exc).__name__}: {exc}"],
                "score": 0.0,
            }
        label_errors.extend(validator_result["errors"])
        label_warnings.extend(validator_result["warnings"])

        if label.structured_label.kind in {"mermaid", "table"} and evidence_score <= 0.3:
            label_errors.append(
                f"low evidence score for structured {label.structured_label.kind} output"
            )
        if hallucination_risk >= 0.6:
            label_errors.append("hallucination risk is too high")

        label_validator_score = round(
            (float(caption_result["score"]) + float(validator_result["score"])) / 2, 4
        )
        validator_scores.append(label_validator_score)
        evidence_scores.append(evidence_score)
        hallucination_risks.append(hallucination_risk)
        critical_errors.extend(label_errors)
        warnings.extend(label_warnings)

        label_detail["candidate_texts"] = candidate_texts
        label_detail["caption_validation"] = caption_result
        label_detail["structured_validation"] = validator_result
        label_detail["evidence_score"] = evidence_score
        label_detail["hallucination_risk"] = hallucination_risk
        label_detail["evidence_details"] = evidence_detail
        label_detail["validator_score"] = label_validator_score
        per_label_details.append(label_detail)

    evidence_score = round(_average(evidence_scores, default=0.5), 4)
    validator_score = round(_average(validator_scores, default=0.0), 4)
    hallucination_risk = round(max(hallucination_risks) if hallucination_risks else 0.0, 4)

    return ValidationResult(
        image_id=image_id,
        evidence_score=evidence_score,
        validator_score=validator_score,
        hallucination_risk=hallucination_risk,
        critical_errors=_deduplicate(critical_errors),
        warnings=_deduplicate(warnings),
        details={
            "label_count": len(labels),
            "majority_image_type": majority_type,
            "majority_structured_kind": majority_kind,
            "stamp_mode": stamp_mode,
            "image_type_counts": dict(Counter(label.image_type for label in labels)),
            "structured_kind_counts": dict(
                Counter(label.structured_label.kind for label in labels)
            ),
            "evidence_text_pool": evidence_texts,
            "ocr_region_role_pools": ocr_region_role_pools,
            "ocr_region_consensus": ocr_details,
            "per_label": per_label_details,
        },
    )


def _validate_structured_label(label: ParsedLabel, stamp_mode: bool = False) -> dict:
    kind = label.structured_label.kind
    content = label.structured_label.content

    if stamp_mode:
        return {
            "valid_basic": True,
            "errors": [],
            "warnings": [],
            "score": 1.0,
        }

    if kind == "mermaid":
        return validate_mermaid(content=content, image_type=label.image_type)
    if kind == "table":
        return validate_table(content=content, image_type=label.image_type)
    if kind == "text":
        warnings: list[str] = []
        score = 0.8
        if not content.strip() and label.image_type not in {"natural_image", "unknown"}:
            warnings.append("kind is text but content is empty")
            score = 0.4
        return {
            "valid_basic": bool(content.strip()) or label.image_type in {"natural_image", "unknown"},
            "errors": [],
            "warnings": warnings,
            "score": score,
        }
    if kind == "none":
        warnings: list[str] = []
        score = 0.85
        if label.image_type in {"flowchart", "chart", "table"}:
            warnings.append(
                "structured output is none for an image type that often expects structure"
            )
            score = 0.35
        return {
            "valid_basic": label.image_type not in {"flowchart", "chart", "table"},
            "errors": [],
            "warnings": warnings,
            "score": score,
        }

    return {
        "valid_basic": False,
        "errors": [f"unsupported structured kind: {kind}"],
        "warnings": [],
        "score": 0.0,
    }


def _majority_value(values: list[str], default: str) -> str:
    if not values:
        return default
    counter = Counter(values)
    highest = counter.most_common(1)[0][1]
    for value in values:
        if counter[value] == highest:
            return value
    return values[0]


def _average(values: list[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _validate_ocr_region_consensus(
    labels: list[ParsedLabel],
) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    seal_signatures = [primary_seal_signature(label) for label in labels]
    has_any_seal = any(signature for signature in seal_signatures)

    if has_any_seal:
        if any(not signature for signature in seal_signatures):
            errors.append("primary seal text is missing in some model outputs")

        unique_signatures = {
            signature for signature in seal_signatures if signature
        }
        if len(unique_signatures) > 1:
            errors.append("primary seal texts are inconsistent across model outputs")

    text_role_map: dict[str, set[str]] = {}
    for label in labels:
        special_regions = extract_special_ocr_region_texts(label)
        for role, texts in special_regions.items():
            for text in texts:
                normalized = normalize_text(text)
                if not normalized:
                    continue
                text_role_map.setdefault(normalized, set()).add(role)

    role_conflicts = sorted(
        normalized_text
        for normalized_text, roles in text_role_map.items()
        if "seal" in roles and len(roles) > 1
    )
    if role_conflicts:
        errors.append("seal texts conflict with watermark/footer roles across model outputs")

    for label in labels:
        special_regions = extract_special_ocr_region_texts(label)
        if special_regions["seal"] and special_regions["watermark"]:
            overlap = {
                normalize_text(text)
                for text in special_regions["seal"]
            } & {
                normalize_text(text)
                for text in special_regions["watermark"]
            }
            if overlap:
                warnings.append("same text appears in both seal and watermark regions in one model output")
        if special_regions["seal"] and special_regions["footer"]:
            overlap = {
                normalize_text(text)
                for text in special_regions["seal"]
            } & {
                normalize_text(text)
                for text in special_regions["footer"]
            }
            if overlap:
                warnings.append("same text appears in both seal and footer regions in one model output")

    return (
        _deduplicate(errors),
        _deduplicate(warnings),
        {
            "has_any_seal": has_any_seal,
            "seal_signatures": [[signature] if signature else [] for signature in seal_signatures],
            "primary_seal_texts": [primary_seal_text(label) for label in labels],
            "role_conflicts": role_conflicts,
        },
    )


__all__ = ["ValidationResult", "validate_labels"]
