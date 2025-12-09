"""Tests for core types.

These tests verify the fundamental types used across all GENESIS modules.
"""

import pytest

from core.types import (
    CONFIDENCE_THRESHOLD,
    Confidence,
    ModuleOutput,
    UncertaintyReason,
    VerificationResult,
)


class TestConfidence:
    """Tests for Confidence type."""

    def test_confidence_valid_values(self) -> None:
        """Valid confidence values should be accepted."""
        assert Confidence(0.0) == 0.0
        assert Confidence(0.5) == 0.5
        assert Confidence(1.0) == 1.0
        assert Confidence(0.85) == 0.85

    def test_confidence_below_zero_raises(self) -> None:
        """Confidence below 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Confidence(-0.1)

    def test_confidence_above_one_raises(self) -> None:
        """Confidence above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Confidence(1.1)

    def test_confidence_threshold_is_085(self) -> None:
        """Default confidence threshold should be 0.85."""
        assert CONFIDENCE_THRESHOLD == 0.85


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_verification_result_passed(self) -> None:
        """Passed verification result."""
        result = VerificationResult(
            passed=True,
            method="compiler",
            details="Compiled successfully",
        )
        assert result.passed is True
        assert result.method == "compiler"
        assert result.errors == []

    def test_verification_result_failed(self) -> None:
        """Failed verification result with errors."""
        result = VerificationResult(
            passed=False,
            method="static_analysis",
            errors=["Undefined variable 'x'", "Type mismatch"],
        )
        assert result.passed is False
        assert len(result.errors) == 2


class TestModuleOutput:
    """Tests for ModuleOutput."""

    def test_successful_output(self) -> None:
        """Create a successful output."""
        output = ModuleOutput.success(
            result={"parsed": "data"},
            confidence=0.95,
        )
        assert output.result == {"parsed": "data"}
        assert output.confidence == 0.95
        assert output.is_uncertain is False
        assert output.uncertainty_reason is None

    def test_uncertain_output(self) -> None:
        """Create an uncertain/refusal output."""
        output = ModuleOutput.uncertain(
            reason=UncertaintyReason.AMBIGUOUS_INPUT,
            details="Multiple valid interpretations",
        )
        assert output.result is None
        assert output.confidence == 0.0
        assert output.is_uncertain is True
        assert output.uncertainty_reason == UncertaintyReason.AMBIGUOUS_INPUT

    def test_uncertain_requires_reason(self) -> None:
        """Uncertain output must have a reason."""
        with pytest.raises(ValueError, match="uncertainty_reason required"):
            ModuleOutput(
                result=None,
                confidence=Confidence(0.0),
                is_uncertain=True,
                uncertainty_reason=None,  # Missing!
            )

    def test_uncertain_requires_low_confidence(self) -> None:
        """Uncertain output must have low confidence."""
        with pytest.raises(ValueError, match="Confidence should be <= 0.5"):
            ModuleOutput(
                result=None,
                confidence=Confidence(0.9),  # Too high!
                is_uncertain=True,
                uncertainty_reason=UncertaintyReason.AMBIGUOUS_INPUT,
            )

    def test_output_with_verification(self) -> None:
        """Output with verification result."""
        verification = VerificationResult(
            passed=True,
            method="compiler",
        )
        output = ModuleOutput.success(
            result="compiled_code",
            confidence=0.99,
            verification=verification,
        )
        assert output.verification is not None
        assert output.verification.passed is True


class TestUncertaintyReason:
    """Tests for UncertaintyReason enum."""

    def test_all_reasons_have_descriptions(self) -> None:
        """All uncertainty reasons should have meaningful values."""
        for reason in UncertaintyReason:
            assert len(reason.value) > 0
            assert isinstance(reason.value, str)

    def test_common_reasons_exist(self) -> None:
        """Common uncertainty reasons should be defined."""
        assert UncertaintyReason.AMBIGUOUS_INPUT
        assert UncertaintyReason.INSUFFICIENT_DATA
        assert UncertaintyReason.OUT_OF_SCOPE
        assert UncertaintyReason.LOW_CONFIDENCE
        assert UncertaintyReason.VERIFICATION_FAILED
