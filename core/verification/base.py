"""Base verification interfaces.

All verifiers must implement the Verifier protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from core.types import VerificationResult


class VerificationError(Exception):
    """Raised when verification encounters an error (not a failure)."""

    pass


T = TypeVar("T")


class Verifier(ABC, Generic[T]):
    """Abstract base class for all verifiers.

    Verifiers provide deterministic ground truth for outputs.
    They answer: "Is this output valid/correct?"

    Subclasses must implement:
    - verify(): Check if input is valid
    - name: Human-readable name for this verifier
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this verifier."""
        ...

    @abstractmethod
    def verify(self, data: T) -> VerificationResult:
        """Verify the given data.

        Args:
            data: The data to verify

        Returns:
            VerificationResult with passed=True/False and details

        Raises:
            VerificationError: If verification itself fails (not the data)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


@dataclass
class CompositeVerifier(Verifier[T]):
    """Combines multiple verifiers - all must pass."""

    verifiers: list[Verifier[T]]

    @property
    def name(self) -> str:
        names = ", ".join(v.name for v in self.verifiers)
        return f"Composite({names})"

    def verify(self, data: T) -> VerificationResult:
        """Run all verifiers. Fails if any fail."""
        all_errors: list[str] = []
        all_details: list[str] = []

        for verifier in self.verifiers:
            result = verifier.verify(data)
            if not result.passed:
                all_errors.extend(result.errors)
            if result.details:
                all_details.append(f"[{verifier.name}] {result.details}")

        return VerificationResult(
            passed=len(all_errors) == 0,
            method=self.name,
            details="; ".join(all_details),
            errors=all_errors,
        )
