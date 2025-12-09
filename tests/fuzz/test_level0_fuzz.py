"""Fuzzing tests for Level 0.

These tests generate adversarial inputs designed to break the module:
- Malformed bytes
- Boundary conditions
- Pathological patterns
- Resource exhaustion attempts

The goal is to find inputs that cause:
- Crashes
- Hangs
- Memory issues
- Incorrect uncertainty handling
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from levels.level0_machine import (
    Architecture,
    Level0Input,
    Level0Module,
    Level0Output,
)


class TestLevel0FuzzMalformed:
    """Fuzz with malformed/adversarial byte patterns."""

    @pytest.fixture
    def module(self) -> Level0Module:
        return Level0Module()

    def test_all_zeros(self, module: Level0Module) -> None:
        """All zero bytes."""
        data = b"\x00" * 1000
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_all_ones(self, module: Level0Module) -> None:
        """All 0xFF bytes."""
        data = b"\xff" * 1000
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_alternating_pattern(self, module: Level0Module) -> None:
        """Alternating byte pattern."""
        data = b"\xaa\x55" * 500
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_prefix_flood(self, module: Level0Module) -> None:
        """Many instruction prefixes (can confuse decoders)."""
        # 0x66 is operand size prefix, 0x67 is address size prefix
        data = b"\x66\x67" * 100 + b"\x90"  # prefixes + nop
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_rex_prefix_flood(self, module: Level0Module) -> None:
        """Many REX prefixes (x86_64 specific)."""
        data = b"\x48\x48\x48\x48\x89\xd8"  # REX.W prefixes + mov
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_incomplete_multibyte(self, module: Level0Module) -> None:
        """Incomplete multi-byte instructions."""
        incomplete_patterns = [
            b"\x0f",  # Two-byte opcode prefix alone
            b"\x0f\x38",  # Three-byte opcode prefix incomplete
            b"\x0f\x3a",  # Three-byte opcode prefix incomplete
            b"\xc4",  # VEX prefix alone
            b"\xc5",  # VEX prefix alone
            b"\x62",  # EVEX prefix alone
        ]
        for data in incomplete_patterns:
            inp = Level0Input(data=data, architecture=Architecture.X86_64)
            output = module.process(inp)
            assert isinstance(output, Level0Output)

    def test_invalid_modrm(self, module: Level0Module) -> None:
        """Instructions with potentially invalid ModR/M bytes."""
        # Various ModR/M combinations
        for modrm in range(256):
            data = bytes([0x89, modrm])  # mov with various ModR/M
            inp = Level0Input(data=data, architecture=Architecture.X86_64)
            output = module.process(inp)
            assert isinstance(output, Level0Output)


class TestLevel0FuzzBoundary:
    """Fuzz boundary conditions."""

    @pytest.fixture
    def module(self) -> Level0Module:
        return Level0Module()

    def test_single_byte_all_values(self, module: Level0Module) -> None:
        """Test every possible single byte value."""
        for byte_val in range(256):
            data = bytes([byte_val])
            inp = Level0Input(data=data, architecture=Architecture.X86_64)
            output = module.process(inp)
            assert isinstance(output, Level0Output)

    def test_two_bytes_sample(self, module: Level0Module) -> None:
        """Sample of two-byte combinations."""
        import random

        random.seed(42)  # Reproducible

        for _ in range(1000):
            b1, b2 = random.randint(0, 255), random.randint(0, 255)
            data = bytes([b1, b2])
            inp = Level0Input(data=data, architecture=Architecture.X86_64)
            output = module.process(inp)
            assert isinstance(output, Level0Output)

    def test_max_instruction_length(self, module: Level0Module) -> None:
        """x86 max instruction length is 15 bytes."""
        # Create a 15-byte instruction (with prefixes)
        data = b"\x66" * 14 + b"\x90"  # 14 prefixes + nop
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_beyond_max_instruction_length(self, module: Level0Module) -> None:
        """Beyond 15 bytes should be handled."""
        data = b"\x66" * 20 + b"\x90"  # Too many prefixes
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)


class TestLevel0FuzzResourceExhaustion:
    """Test resource exhaustion scenarios."""

    @pytest.fixture
    def module(self) -> Level0Module:
        return Level0Module()

    def test_large_input(self, module: Level0Module) -> None:
        """Large input shouldn't cause memory issues."""
        data = b"\x90" * 100000  # 100KB of NOPs
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_many_small_instructions(self, module: Level0Module) -> None:
        """Many small instructions."""
        data = b"\x90" * 10000  # 10K NOPs
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        output = module.process(inp)

        if not output.is_uncertain:
            assert output.instruction_count == 10000

    def test_repeated_processing(self, module: Level0Module) -> None:
        """Repeated processing shouldn't leak resources."""
        data = b"\x55\x48\x89\xe5\xc3"
        inp = Level0Input(data=data, architecture=Architecture.X86_64)

        for _ in range(1000):
            output = module.process(inp)
            assert isinstance(output, Level0Output)


class TestLevel0FuzzArchitectureMismatch:
    """Test architecture mismatch scenarios."""

    @pytest.fixture
    def module(self) -> Level0Module:
        return Level0Module()

    def test_x86_code_as_arm(self, module: Level0Module) -> None:
        """x86 code interpreted as ARM."""
        x86_code = b"\x55\x48\x89\xe5\xc3"  # x86_64 function prologue
        inp = Level0Input(data=x86_code, architecture=Architecture.ARM64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_arm_code_as_x86(self, module: Level0Module) -> None:
        """ARM code interpreted as x86."""
        # ARM64: mov x0, #0; ret
        arm_code = b"\x00\x00\x80\xd2\xc0\x03\x5f\xd6"
        inp = Level0Input(data=arm_code, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    def test_32bit_code_as_64bit(self, module: Level0Module) -> None:
        """32-bit code interpreted as 64-bit."""
        x86_32_code = b"\x55\x89\xe5\xc3"  # 32-bit prologue
        inp = Level0Input(data=x86_32_code, architecture=Architecture.X86_64)
        output = module.process(inp)
        assert isinstance(output, Level0Output)


class TestLevel0FuzzHypothesis:
    """Hypothesis-driven fuzzing for deeper exploration."""

    @given(
        data=st.binary(min_size=1, max_size=100),
        base_addr=st.integers(min_value=0, max_value=0xFFFFFFFF),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_random_data_with_base_address(self, data: bytes, base_addr: int) -> None:
        """Random data with various base addresses."""
        module = Level0Module()
        inp = Level0Input(
            data=data,
            architecture=Architecture.X86_64,
            base_address=base_addr,
        )
        output = module.process(inp)
        assert isinstance(output, Level0Output)

    @given(
        data=st.binary(min_size=1, max_size=50),
        max_insn=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_random_data_with_max_instructions(self, data: bytes, max_insn: int) -> None:
        """Random data with instruction limits."""
        module = Level0Module()
        inp = Level0Input(
            data=data,
            architecture=Architecture.X86_64,
            max_instructions=max_insn,
        )
        output = module.process(inp)
        assert isinstance(output, Level0Output)

        # If successful, should respect limit
        if not output.is_uncertain:
            assert output.instruction_count <= max_insn
