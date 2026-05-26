from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    image_id: str
    evidence_score: float = 0.0
    validator_score: float = 0.0
    hallucination_risk: float = 0.0
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "evidence_score": self.evidence_score,
            "validator_score": self.validator_score,
            "hallucination_risk": self.hallucination_risk,
            "critical_errors": list(self.critical_errors),
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }
