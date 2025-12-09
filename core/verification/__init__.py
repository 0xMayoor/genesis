"""GENESIS Verification Framework.

Deterministic verification tools that provide ground truth.
These tools are the foundation of zero-hallucination.

Principle: Verification output > Model confidence
"""

from core.verification.base import VerificationError, Verifier
from core.verification.syntax import SyntaxVerifier

__all__ = ["Verifier", "VerificationError", "SyntaxVerifier"]
