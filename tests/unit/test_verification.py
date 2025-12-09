"""Tests for verification framework.

These tests verify that our verification tools work correctly.
"""

import pytest

from core.types import VerificationResult
from core.verification.base import CompositeVerifier
from core.verification.syntax import CSyntaxVerifier, PythonSyntaxVerifier


class TestPythonSyntaxVerifier:
    """Tests for Python syntax verification."""

    def test_valid_python_code(self) -> None:
        """Valid Python code should pass."""
        verifier = PythonSyntaxVerifier()
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

x = hello("world")
"""
        result = verifier.verify(code)
        assert result.passed is True
        assert result.method == "python_syntax"

    def test_invalid_python_syntax(self) -> None:
        """Invalid Python syntax should fail."""
        verifier = PythonSyntaxVerifier()
        code = """
def broken(
    return "missing closing paren"
"""
        result = verifier.verify(code)
        assert result.passed is False
        assert len(result.errors) > 0
        assert "SyntaxError" in result.errors[0]

    def test_empty_code_is_valid(self) -> None:
        """Empty string is valid Python."""
        verifier = PythonSyntaxVerifier()
        result = verifier.verify("")
        assert result.passed is True

    def test_single_expression(self) -> None:
        """Single expression is valid."""
        verifier = PythonSyntaxVerifier()
        result = verifier.verify("1 + 2")
        assert result.passed is True

    def test_indentation_error(self) -> None:
        """Indentation errors should fail."""
        verifier = PythonSyntaxVerifier()
        code = """
def foo():
return 1
"""
        result = verifier.verify(code)
        assert result.passed is False


class TestCSyntaxVerifier:
    """Tests for C syntax verification."""

    @pytest.fixture
    def verifier(self) -> CSyntaxVerifier:
        return CSyntaxVerifier()

    def test_valid_c_code(self, verifier: CSyntaxVerifier) -> None:
        """Valid C code should pass."""
        code = """
#include <stdio.h>

int main(void) {
    printf("Hello, world!\\n");
    return 0;
}
"""
        result = verifier.verify(code)
        assert result.passed is True
        assert result.method == "c_syntax"

    def test_invalid_c_syntax(self, verifier: CSyntaxVerifier) -> None:
        """Invalid C syntax should fail."""
        code = """
int main(void) {
    int x =
    return 0;
}
"""
        result = verifier.verify(code)
        assert result.passed is False
        assert len(result.errors) > 0

    def test_missing_semicolon(self, verifier: CSyntaxVerifier) -> None:
        """Missing semicolon should fail."""
        code = """
int main(void) {
    int x = 5
    return 0;
}
"""
        result = verifier.verify(code)
        assert result.passed is False


class TestCompositeVerifier:
    """Tests for composite verifier."""

    def test_all_pass(self) -> None:
        """All verifiers pass -> composite passes."""
        v1 = PythonSyntaxVerifier()
        composite = CompositeVerifier([v1])

        result = composite.verify("x = 1")
        assert result.passed is True

    def test_one_fails(self) -> None:
        """One verifier fails -> composite fails."""
        v1 = PythonSyntaxVerifier()
        composite = CompositeVerifier([v1])

        result = composite.verify("def broken(")
        assert result.passed is False
        assert len(result.errors) > 0

    def test_composite_name(self) -> None:
        """Composite name includes all verifier names."""
        v1 = PythonSyntaxVerifier()
        composite = CompositeVerifier([v1])

        assert "python_syntax" in composite.name


class TestVerifierProtocol:
    """Tests for verifier protocol compliance."""

    def test_verifier_has_name(self) -> None:
        """All verifiers must have a name."""
        verifier = PythonSyntaxVerifier()
        assert isinstance(verifier.name, str)
        assert len(verifier.name) > 0

    def test_verifier_returns_verification_result(self) -> None:
        """Verify returns VerificationResult."""
        verifier = PythonSyntaxVerifier()
        result = verifier.verify("x = 1")
        assert isinstance(result, VerificationResult)

    def test_verifier_repr(self) -> None:
        """Verifier has useful repr."""
        verifier = PythonSyntaxVerifier()
        repr_str = repr(verifier)
        assert "PythonSyntaxVerifier" in repr_str
        assert "python_syntax" in repr_str
