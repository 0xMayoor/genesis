"""Type definitions for Level 0: Machine Code Patterns.

These types define the input/output contract for Level 0.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.types import Confidence, UncertaintyReason, VerificationResult


class Architecture(Enum):
    """Supported CPU architectures."""

    X86_64 = "x86_64"
    X86_32 = "x86_32"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


class InstructionCategory(Enum):
    """Categories of machine instructions."""

    DATA_TRANSFER = "data_transfer"  # mov, push, pop, lea
    ARITHMETIC = "arithmetic"  # add, sub, mul, div, inc, dec
    LOGIC = "logic"  # and, or, xor, not, shl, shr
    CONTROL_FLOW = "control_flow"  # jmp, call, ret, je, jne
    COMPARISON = "comparison"  # cmp, test
    STACK = "stack"  # push, pop, enter, leave
    SYSTEM = "system"  # syscall, int, nop
    STRING = "string"  # movs, cmps, scas
    SIMD = "simd"  # SSE, AVX instructions
    UNKNOWN = "unknown"  # Unrecognized


@dataclass(frozen=True)
class Instruction:
    """A single decoded machine instruction."""

    offset: int
    """Byte offset from start of input."""

    raw_bytes: bytes
    """Raw instruction bytes."""

    mnemonic: str
    """Instruction mnemonic (e.g., 'mov', 'jmp')."""

    operands: tuple[str, ...]
    """Operands as strings (e.g., ('eax', 'ebx'))."""

    size: int
    """Instruction size in bytes."""

    category: InstructionCategory
    """Instruction category."""

    @property
    def assembly(self) -> str:
        """Full assembly representation."""
        if self.operands:
            return f"{self.mnemonic} {', '.join(self.operands)}"
        return self.mnemonic

    def __str__(self) -> str:
        hex_bytes = self.raw_bytes.hex()
        return f"0x{self.offset:04x}: {hex_bytes:<16} {self.assembly}"


@dataclass
class Level0Input:
    """Input to Level 0 module."""

    data: bytes
    """Raw binary data to analyze."""

    architecture: Architecture | None = None
    """Target architecture. None for auto-detect."""

    base_address: int = 0
    """Base address for offset calculations."""

    max_instructions: int | None = None
    """Maximum instructions to decode. None for all."""

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError(f"data must be bytes, got {type(self.data)}")
        if len(self.data) == 0:
            raise ValueError("data cannot be empty")


@dataclass
class Level0Output:
    """Output from Level 0 module."""

    instructions: tuple[Instruction, ...]
    """Decoded instructions."""

    architecture: Architecture
    """Detected or confirmed architecture."""

    bytes_processed: int
    """Number of bytes successfully processed."""

    confidence: Confidence
    """Overall confidence in the output."""

    is_uncertain: bool
    """If True, this output should be treated as a refusal."""

    uncertainty_reason: UncertaintyReason | None = None
    """Why uncertain (if is_uncertain is True)."""

    verification: VerificationResult | None = None
    """Verification result if performed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional information."""

    def __post_init__(self) -> None:
        if self.is_uncertain and self.uncertainty_reason is None:
            raise ValueError("uncertainty_reason required when is_uncertain")

    @classmethod
    def uncertain(
        cls,
        reason: UncertaintyReason,
        architecture: Architecture = Architecture.UNKNOWN,
        details: str = "",
    ) -> "Level0Output":
        """Create an uncertain/refusal output."""
        return cls(
            instructions=(),
            architecture=architecture,
            bytes_processed=0,
            confidence=Confidence(0.0),
            is_uncertain=True,
            uncertainty_reason=reason,
            metadata={"details": details} if details else {},
        )

    @classmethod
    def success(
        cls,
        instructions: tuple[Instruction, ...],
        architecture: Architecture,
        bytes_processed: int,
        confidence: float = 0.95,
        verification: VerificationResult | None = None,
    ) -> "Level0Output":
        """Create a successful output."""
        return cls(
            instructions=instructions,
            architecture=architecture,
            bytes_processed=bytes_processed,
            confidence=Confidence(confidence),
            is_uncertain=False,
            verification=verification,
        )

    @property
    def instruction_count(self) -> int:
        """Number of decoded instructions."""
        return len(self.instructions)
