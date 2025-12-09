"""Property-based tests for Level 0.

These tests use Hypothesis to generate random inputs and verify
that certain properties ALWAYS hold, regardless of input.

Key properties:
1. Module never crashes on any input
2. Output types are always correct
3. Uncertain outputs always have reasons
4. Byte reconstruction matches original (when successful)
5. Confidence is always in valid range
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from core.types import UncertaintyReason
from levels.level0_machine import (
    Architecture,
    InstructionCategory,
    Level0Input,
    Level0Module,
    Level0Output,
)

# Strategies for generating test data
architectures = st.sampled_from(
    [
        Architecture.X86_64,
        Architecture.X86_32,
        Architecture.ARM64,
        Architecture.ARM32,
        Architecture.UNKNOWN,
    ]
)

# Valid x86 instruction bytes (common patterns)
valid_x86_bytes = st.sampled_from(
    [
        b"\x90",  # nop
        b"\xc3",  # ret
        b"\x55",  # push rbp
        b"\x5d",  # pop rbp
        b"\x89\xd8",  # mov eax, ebx
        b"\x31\xc0",  # xor eax, eax
        b"\x48\x89\xe5",  # mov rbp, rsp
        b"\xe8\x00\x00\x00\x00",  # call +0
        b"\xeb\x00",  # jmp +0
        b"\x83\xc0\x01",  # add eax, 1
    ]
)

# Random bytes (may or may not be valid)
random_bytes = st.binary(min_size=1, max_size=256)

# Sequences of valid instructions
valid_instruction_sequence = st.lists(valid_x86_bytes, min_size=1, max_size=20).map(
    lambda parts: b"".join(parts)
)


class TestLevel0NeverCrashes:
    """Property: Module NEVER crashes, regardless of input."""

    @given(data=random_bytes)
    @settings(max_examples=500)
    def test_random_bytes_never_crash(self, data: bytes) -> None:
        """Random bytes should never cause a crash."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)

        # This should NEVER raise an exception
        output = module.process(inp)

        # Output should always be valid
        assert isinstance(output, Level0Output)

    @given(data=random_bytes, arch=architectures)
    @settings(max_examples=200)
    def test_any_architecture_never_crashes(self, data: bytes, arch: Architecture) -> None:
        """Any architecture with any data should not crash."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=arch)

        output = module.process(inp)
        assert isinstance(output, Level0Output)

    @given(data=st.binary(min_size=1, max_size=1))
    def test_single_byte_never_crashes(self, data: bytes) -> None:
        """Single bytes should never crash."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)

        output = module.process(inp)
        assert isinstance(output, Level0Output)


class TestLevel0OutputInvariants:
    """Property: Output always satisfies type invariants."""

    @given(data=random_bytes)
    @settings(max_examples=300)
    def test_confidence_always_valid(self, data: bytes) -> None:
        """Confidence is always between 0 and 1."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        assert 0.0 <= float(output.confidence) <= 1.0

    @given(data=random_bytes)
    @settings(max_examples=300)
    def test_uncertain_always_has_reason(self, data: bytes) -> None:
        """If uncertain, must have a reason."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        if output.is_uncertain:
            assert output.uncertainty_reason is not None
            assert isinstance(output.uncertainty_reason, UncertaintyReason)

    @given(data=random_bytes)
    @settings(max_examples=300)
    def test_architecture_always_set(self, data: bytes) -> None:
        """Architecture is always set in output."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        assert isinstance(output.architecture, Architecture)

    @given(data=valid_instruction_sequence)
    @settings(max_examples=200)
    def test_instructions_have_valid_categories(self, data: bytes) -> None:
        """All decoded instructions have valid categories."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        for insn in output.instructions:
            assert isinstance(insn.category, InstructionCategory)


class TestLevel0ByteReconstruction:
    """Property: Decoded bytes can be reconstructed."""

    @given(data=valid_instruction_sequence)
    @settings(max_examples=200)
    def test_byte_reconstruction_matches(self, data: bytes) -> None:
        """Reconstructed bytes match original for successful decodes."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        if not output.is_uncertain and output.instruction_count > 0:
            # Reconstruct bytes from instructions
            reconstructed = b"".join(i.raw_bytes for i in output.instructions)

            # Should match the processed portion
            assert reconstructed == data[: output.bytes_processed]

    @given(data=valid_instruction_sequence)
    @settings(max_examples=200)
    def test_instruction_sizes_sum_to_bytes_processed(self, data: bytes) -> None:
        """Sum of instruction sizes equals bytes_processed."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        if not output.is_uncertain:
            total_size = sum(i.size for i in output.instructions)
            assert total_size == output.bytes_processed


class TestLevel0Determinism:
    """Property: Same input always produces same output."""

    @given(data=random_bytes)
    @settings(max_examples=100)
    def test_deterministic_output(self, data: bytes) -> None:
        """Same input produces identical output."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)

        output1 = module.process(inp)
        output2 = module.process(inp)

        assert output1.is_uncertain == output2.is_uncertain
        assert output1.instruction_count == output2.instruction_count
        assert float(output1.confidence) == float(output2.confidence)

        if not output1.is_uncertain:
            for i1, i2 in zip(output1.instructions, output2.instructions, strict=False):
                assert i1.mnemonic == i2.mnemonic
                assert i1.raw_bytes == i2.raw_bytes


class TestLevel0InstructionProperties:
    """Property: Individual instructions satisfy invariants."""

    @given(data=valid_instruction_sequence)
    @settings(max_examples=200)
    def test_instruction_offset_monotonic(self, data: bytes) -> None:
        """Instruction offsets are monotonically increasing."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        if output.instruction_count > 1:
            for i in range(1, len(output.instructions)):
                prev = output.instructions[i - 1]
                curr = output.instructions[i]
                assert curr.offset > prev.offset

    @given(data=valid_instruction_sequence)
    @settings(max_examples=200)
    def test_instruction_size_positive(self, data: bytes) -> None:
        """All instructions have positive size."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        for insn in output.instructions:
            assert insn.size > 0

    @given(data=valid_instruction_sequence)
    @settings(max_examples=200)
    def test_instruction_mnemonic_not_empty(self, data: bytes) -> None:
        """All instructions have non-empty mnemonics."""
        module = Level0Module()
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        for insn in output.instructions:
            assert len(insn.mnemonic) > 0
