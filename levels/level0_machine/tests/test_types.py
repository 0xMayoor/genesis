"""Tests for Level 0 types."""

import pytest

from core.types import Confidence, UncertaintyReason
from levels.level0_machine.types import (
    Architecture,
    Instruction,
    InstructionCategory,
    Level0Input,
    Level0Output,
)


class TestInstruction:
    """Tests for Instruction dataclass."""

    def test_instruction_creation(self) -> None:
        """Create a valid instruction."""
        insn = Instruction(
            offset=0,
            raw_bytes=b"\x89\xd8",
            mnemonic="mov",
            operands=("eax", "ebx"),
            size=2,
            category=InstructionCategory.DATA_TRANSFER,
        )
        assert insn.mnemonic == "mov"
        assert insn.size == 2
        assert insn.category == InstructionCategory.DATA_TRANSFER

    def test_instruction_assembly_property(self) -> None:
        """Assembly property formats correctly."""
        insn = Instruction(
            offset=0,
            raw_bytes=b"\x89\xd8",
            mnemonic="mov",
            operands=("eax", "ebx"),
            size=2,
            category=InstructionCategory.DATA_TRANSFER,
        )
        assert insn.assembly == "mov eax, ebx"

    def test_instruction_no_operands(self) -> None:
        """Instruction with no operands."""
        insn = Instruction(
            offset=0,
            raw_bytes=b"\xc3",
            mnemonic="ret",
            operands=(),
            size=1,
            category=InstructionCategory.CONTROL_FLOW,
        )
        assert insn.assembly == "ret"

    def test_instruction_str(self) -> None:
        """String representation includes all info."""
        insn = Instruction(
            offset=0x10,
            raw_bytes=b"\x89\xd8",
            mnemonic="mov",
            operands=("eax", "ebx"),
            size=2,
            category=InstructionCategory.DATA_TRANSFER,
        )
        s = str(insn)
        assert "0x0010" in s
        assert "89d8" in s
        assert "mov" in s

    def test_instruction_is_frozen(self) -> None:
        """Instructions are immutable."""
        insn = Instruction(
            offset=0,
            raw_bytes=b"\xc3",
            mnemonic="ret",
            operands=(),
            size=1,
            category=InstructionCategory.CONTROL_FLOW,
        )
        with pytest.raises(AttributeError):
            insn.mnemonic = "call"  # type: ignore


class TestLevel0Input:
    """Tests for Level0Input."""

    def test_valid_input(self) -> None:
        """Create valid input."""
        inp = Level0Input(data=b"\x90\x90\x90")
        assert inp.data == b"\x90\x90\x90"
        assert inp.architecture is None
        assert inp.base_address == 0

    def test_input_with_architecture(self) -> None:
        """Input with specified architecture."""
        inp = Level0Input(
            data=b"\x90",
            architecture=Architecture.X86_64,
        )
        assert inp.architecture == Architecture.X86_64

    def test_empty_data_raises(self) -> None:
        """Empty data should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Level0Input(data=b"")

    def test_non_bytes_raises(self) -> None:
        """Non-bytes data should raise TypeError."""
        with pytest.raises(TypeError, match="must be bytes"):
            Level0Input(data="not bytes")  # type: ignore


class TestLevel0Output:
    """Tests for Level0Output."""

    def test_successful_output(self) -> None:
        """Create successful output."""
        insn = Instruction(
            offset=0,
            raw_bytes=b"\x90",
            mnemonic="nop",
            operands=(),
            size=1,
            category=InstructionCategory.SYSTEM,
        )
        output = Level0Output.success(
            instructions=(insn,),
            architecture=Architecture.X86_64,
            bytes_processed=1,
            confidence=0.95,
        )
        assert output.is_uncertain is False
        assert output.instruction_count == 1
        assert output.confidence == 0.95

    def test_uncertain_output(self) -> None:
        """Create uncertain output."""
        output = Level0Output.uncertain(
            reason=UncertaintyReason.MALFORMED_INPUT,
            details="Invalid bytes",
        )
        assert output.is_uncertain is True
        assert output.uncertainty_reason == UncertaintyReason.MALFORMED_INPUT
        assert output.instruction_count == 0
        assert output.confidence == 0.0

    def test_uncertain_requires_reason(self) -> None:
        """Uncertain output must have reason."""
        with pytest.raises(ValueError, match="uncertainty_reason required"):
            Level0Output(
                instructions=(),
                architecture=Architecture.X86_64,
                bytes_processed=0,
                confidence=Confidence(0.0),
                is_uncertain=True,
                uncertainty_reason=None,
            )


class TestArchitecture:
    """Tests for Architecture enum."""

    def test_all_architectures_defined(self) -> None:
        """All expected architectures exist."""
        assert Architecture.X86_64
        assert Architecture.X86_32
        assert Architecture.ARM64
        assert Architecture.ARM32
        assert Architecture.UNKNOWN


class TestInstructionCategory:
    """Tests for InstructionCategory enum."""

    def test_all_categories_defined(self) -> None:
        """All expected categories exist."""
        assert InstructionCategory.DATA_TRANSFER
        assert InstructionCategory.ARITHMETIC
        assert InstructionCategory.LOGIC
        assert InstructionCategory.CONTROL_FLOW
        assert InstructionCategory.COMPARISON
        assert InstructionCategory.STACK
        assert InstructionCategory.SYSTEM
        assert InstructionCategory.UNKNOWN
