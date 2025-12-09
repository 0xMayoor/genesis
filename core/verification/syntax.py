"""Syntax verification using compilers and parsers.

These verifiers check that code is syntactically valid.
"""

import ast
import subprocess
import tempfile
from pathlib import Path

from core.types import VerificationResult
from core.verification.base import VerificationError, Verifier


class PythonSyntaxVerifier(Verifier[str]):
    """Verify Python code syntax using ast.parse."""

    @property
    def name(self) -> str:
        return "python_syntax"

    def verify(self, code: str) -> VerificationResult:
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return VerificationResult(
                passed=True,
                method=self.name,
                details="Valid Python syntax",
            )
        except SyntaxError as e:
            return VerificationResult(
                passed=False,
                method=self.name,
                details=f"Line {e.lineno}: {e.msg}",
                errors=[f"SyntaxError at line {e.lineno}: {e.msg}"],
            )


class CSyntaxVerifier(Verifier[str]):
    """Verify C code syntax using gcc -fsyntax-only."""

    def __init__(self, compiler: str = "gcc") -> None:
        self._compiler = compiler

    @property
    def name(self) -> str:
        return "c_syntax"

    def verify(self, code: str) -> VerificationResult:
        """Check if code is valid C syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                [self._compiler, "-fsyntax-only", "-x", "c", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return VerificationResult(
                    passed=True,
                    method=self.name,
                    details="Valid C syntax",
                )
            else:
                errors = result.stderr.strip().split("\n")
                return VerificationResult(
                    passed=False,
                    method=self.name,
                    details=result.stderr[:200],
                    errors=errors[:5],  # Limit errors
                )
        except FileNotFoundError as e:
            raise VerificationError(f"Compiler not found: {self._compiler}") from e
        except subprocess.TimeoutExpired as e:
            raise VerificationError("Compilation timed out") from e
        finally:
            temp_path.unlink(missing_ok=True)


class AssemblySyntaxVerifier(Verifier[str]):
    """Verify assembly syntax using assembler."""

    def __init__(self, assembler: str = "as", arch: str = "x86_64") -> None:
        self._assembler = assembler
        self._arch = arch

    @property
    def name(self) -> str:
        return f"asm_syntax_{self._arch}"

    def verify(self, code: str) -> VerificationResult:
        """Check if code is valid assembly syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".s", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            # Just check syntax, don't produce output
            result = subprocess.run(
                [self._assembler, "-o", "/dev/null", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return VerificationResult(
                    passed=True,
                    method=self.name,
                    details="Valid assembly syntax",
                )
            else:
                errors = result.stderr.strip().split("\n")
                return VerificationResult(
                    passed=False,
                    method=self.name,
                    details=result.stderr[:200],
                    errors=errors[:5],
                )
        except FileNotFoundError as e:
            raise VerificationError(f"Assembler not found: {self._assembler}") from e
        except subprocess.TimeoutExpired as e:
            raise VerificationError("Assembly timed out") from e
        finally:
            temp_path.unlink(missing_ok=True)


# Convenience alias
SyntaxVerifier = PythonSyntaxVerifier
