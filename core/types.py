"""Core types used across all GENESIS modules.

These types define the standard interfaces for inputs, outputs, and results
that all level modules must use.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(float):
    """Confidence score between 0.0 and 1.0.

    Values:
        0.0 - No confidence (should refuse)
        0.5 - Uncertain (may refuse depending on threshold)
        0.85 - Default threshold for acceptance
        1.0 - Full confidence (verified)
    """

    def __new__(cls, value: float) -> "Confidence":
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {value}")
        return super().__new__(cls, value)


class UncertaintyReason(Enum):
    """Standard reasons for uncertainty."""

    AMBIGUOUS_INPUT = "Input has multiple valid interpretations"
    INSUFFICIENT_DATA = "Not enough information to determine answer"
    OUT_OF_SCOPE = "Input is outside this module's domain"
    LOW_CONFIDENCE = "Model confidence below threshold"
    VERIFICATION_FAILED = "Output failed deterministic verification"
    CONTRADICTION = "Input contains contradictory information"
    MALFORMED_INPUT = "Input is malformed or invalid"
    UNKNOWN = "Unknown reason for uncertainty"


@dataclass
class VerificationResult:
    """Result of deterministic verification."""

    passed: bool
    method: str  # e.g., "compiler", "static_analysis", "test_execution"
    details: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class ModuleOutput:
    """Standard output format for all level modules.

    Every module must return this format. The orchestrator uses these
    fields to decide whether to accept, retry, or refuse.
    """

    result: Any
    """The actual output data. Type depends on the module."""

    confidence: Confidence
    """How confident the module is in this result."""

    is_uncertain: bool
    """If True, this output should be treated as a refusal."""

    uncertainty_reason: UncertaintyReason | None = None
    """Why the module is uncertain (if is_uncertain is True)."""

    verification: VerificationResult | None = None
    """Result of deterministic verification (if performed)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional information about the processing."""

    def __post_init__(self) -> None:
        """Validate output consistency."""
        if self.is_uncertain and self.uncertainty_reason is None:
            raise ValueError("uncertainty_reason required when is_uncertain is True")
        if self.is_uncertain and self.confidence > 0.5:
            raise ValueError("Confidence should be <= 0.5 when uncertain")

    @classmethod
    def uncertain(
        cls,
        reason: UncertaintyReason,
        details: str = "",
    ) -> "ModuleOutput":
        """Create an uncertain/refusal output."""
        return cls(
            result=None,
            confidence=Confidence(0.0),
            is_uncertain=True,
            uncertainty_reason=reason,
            metadata={"details": details} if details else {},
        )

    @classmethod
    def success(
        cls,
        result: Any,
        confidence: float = 0.95,
        verification: VerificationResult | None = None,
    ) -> "ModuleOutput":
        """Create a successful output."""
        return cls(
            result=result,
            confidence=Confidence(confidence),
            is_uncertain=False,
            verification=verification,
        )


# Threshold constants
CONFIDENCE_THRESHOLD = Confidence(0.85)
"""Default threshold below which outputs should be refused."""
