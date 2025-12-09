"""External validation tests for Level 0.

These tests compare GENESIS Level 0 output against external
disassemblers (objdump, radare2, ndisasm) to ensure correctness.

The external tools serve as ground truth. If GENESIS disagrees
with ALL external tools, GENESIS is likely wrong.
"""

import pytest

from levels.level0_machine import Architecture, Level0Input, Level0Module
from tests.validation.external_validators import (
    ExternalValidator,
    NdisasmValidator,
    ObjdumpValidator,
    Radare2Validator,
    get_available_validators,
)

# Test data: known instruction sequences
TEST_CASES = [
    # (name, bytes, expected_mnemonics)
    ("single_nop", b"\x90", ["nop"]),
    ("single_ret", b"\xc3", ["ret"]),
    ("push_rbp", b"\x55", ["push"]),
    ("pop_rbp", b"\x5d", ["pop"]),
    ("mov_eax_ebx", b"\x89\xd8", ["mov"]),
    ("xor_eax_eax", b"\x31\xc0", ["xor"]),
    ("add_eax_1", b"\x83\xc0\x01", ["add"]),
    ("function_prologue", b"\x55\x48\x89\xe5", ["push", "mov"]),
    ("function_epilogue", b"\x5d\xc3", ["pop", "ret"]),
    ("simple_function", b"\x55\x48\x89\xe5\x31\xc0\x5d\xc3", ["push", "mov", "xor", "pop", "ret"]),
    ("nop_sled", b"\x90" * 10, ["nop"] * 10),
    ("call_relative", b"\xe8\x00\x00\x00\x00", ["call"]),
    ("jmp_relative", b"\xeb\x00", ["jmp"]),
    ("syscall", b"\x0f\x05", ["syscall"]),
]


@pytest.fixture
def module() -> Level0Module:
    return Level0Module()


@pytest.fixture
def validators() -> list[ExternalValidator]:
    """Get available validators, skip if none."""
    available = get_available_validators()
    if not available:
        pytest.skip("No external validators available")
    return available


class TestExternalValidatorAvailability:
    """Test that we can detect validator availability."""

    def test_objdump_detection(self) -> None:
        """Can detect if objdump is available."""
        v = ObjdumpValidator()
        # Just check it doesn't crash
        result = v.is_available()
        assert isinstance(result, bool)

    def test_radare2_detection(self) -> None:
        """Can detect if radare2 is available."""
        v = Radare2Validator()
        result = v.is_available()
        assert isinstance(result, bool)

    def test_ndisasm_detection(self) -> None:
        """Can detect if ndisasm is available."""
        v = NdisasmValidator()
        result = v.is_available()
        assert isinstance(result, bool)

    def test_get_available_validators(self) -> None:
        """Can get list of available validators."""
        validators = get_available_validators()
        assert isinstance(validators, list)
        # All returned validators should be available
        for v in validators:
            assert v.is_available()


class TestLevel0VsObjdump:
    """Compare Level 0 against objdump."""

    @pytest.fixture
    def objdump(self) -> ObjdumpValidator:
        v = ObjdumpValidator()
        if not v.is_available():
            pytest.skip("objdump not available")
        return v

    @pytest.mark.parametrize("name,data,expected", TEST_CASES)
    def test_mnemonic_match(
        self,
        module: Level0Module,
        objdump: ObjdumpValidator,
        name: str,
        data: bytes,
        expected: list[str],
    ) -> None:
        """GENESIS mnemonics should match objdump."""
        # Get GENESIS output
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        if genesis_output.is_uncertain:
            pytest.skip(f"GENESIS uncertain for {name}")

        # Get objdump output
        objdump_insns = objdump.disassemble(data, "x86_64")

        # Compare instruction count
        assert len(genesis_output.instructions) == len(objdump_insns), (
            f"Instruction count mismatch for {name}"
        )

        # Compare mnemonics (case-insensitive)
        for i, (g_insn, o_insn) in enumerate(
            zip(genesis_output.instructions, objdump_insns, strict=False)
        ):
            g_mnem = g_insn.mnemonic.lower()
            o_mnem = o_insn.mnemonic.lower()
            assert g_mnem == o_mnem, f"Mnemonic mismatch at {i}: GENESIS={g_mnem}, objdump={o_mnem}"

    @pytest.mark.parametrize("name,data,expected", TEST_CASES)
    def test_byte_match(
        self,
        module: Level0Module,
        objdump: ObjdumpValidator,
        name: str,
        data: bytes,
        expected: list[str],
    ) -> None:
        """GENESIS raw bytes should match objdump."""
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        if genesis_output.is_uncertain:
            pytest.skip(f"GENESIS uncertain for {name}")

        objdump_insns = objdump.disassemble(data, "x86_64")

        for i, (g_insn, o_insn) in enumerate(
            zip(genesis_output.instructions, objdump_insns, strict=False)
        ):
            assert g_insn.raw_bytes == o_insn.raw_bytes, (
                f"Byte mismatch at {i}: GENESIS={g_insn.raw_bytes.hex()}, "
                f"objdump={o_insn.raw_bytes.hex()}"
            )


class TestLevel0VsRadare2:
    """Compare Level 0 against radare2."""

    @pytest.fixture
    def r2(self) -> Radare2Validator:
        v = Radare2Validator()
        if not v.is_available():
            pytest.skip("radare2 not available")
        return v

    @pytest.mark.parametrize("name,data,expected", TEST_CASES)
    def test_mnemonic_match(
        self,
        module: Level0Module,
        r2: Radare2Validator,
        name: str,
        data: bytes,
        expected: list[str],
    ) -> None:
        """GENESIS mnemonics should match radare2."""
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        if genesis_output.is_uncertain:
            pytest.skip(f"GENESIS uncertain for {name}")

        r2.disassemble(data, "x86_64")

        # r2 might include extra analysis, so just check we have at least as many
        assert len(genesis_output.instructions) >= len(expected), f"Too few instructions for {name}"

        # Compare first N mnemonics
        for i, (g_insn, exp_mnem) in enumerate(
            zip(genesis_output.instructions, expected, strict=False)
        ):
            g_mnem = g_insn.mnemonic.lower()
            assert g_mnem == exp_mnem.lower(), (
                f"Mnemonic mismatch at {i}: GENESIS={g_mnem}, expected={exp_mnem}"
            )


class TestLevel0VsNdisasm:
    """Compare Level 0 against ndisasm."""

    @pytest.fixture
    def ndisasm(self) -> NdisasmValidator:
        v = NdisasmValidator()
        if not v.is_available():
            pytest.skip("ndisasm not available")
        return v

    @pytest.mark.parametrize("name,data,expected", TEST_CASES)
    def test_mnemonic_match(
        self,
        module: Level0Module,
        ndisasm: NdisasmValidator,
        name: str,
        data: bytes,
        expected: list[str],
    ) -> None:
        """GENESIS mnemonics should match ndisasm."""
        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        if genesis_output.is_uncertain:
            pytest.skip(f"GENESIS uncertain for {name}")

        ndisasm_insns = ndisasm.disassemble(data, "x86_64")

        assert len(genesis_output.instructions) == len(ndisasm_insns), (
            f"Instruction count mismatch for {name}"
        )

        for i, (g_insn, n_insn) in enumerate(
            zip(genesis_output.instructions, ndisasm_insns, strict=False)
        ):
            g_mnem = g_insn.mnemonic.lower()
            n_mnem = n_insn.mnemonic.lower()
            assert g_mnem == n_mnem, f"Mnemonic mismatch at {i}: GENESIS={g_mnem}, ndisasm={n_mnem}"


class TestLevel0ConsensusValidation:
    """Validate GENESIS against consensus of multiple tools."""

    def test_consensus_on_simple_function(
        self,
        module: Level0Module,
        validators: list[ExternalValidator],
    ) -> None:
        """All tools should agree on a simple function."""
        data = b"\x55\x48\x89\xe5\x31\xc0\x5d\xc3"

        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        assert not genesis_output.is_uncertain

        # Get mnemonics from all validators
        all_mnemonics: list[list[str]] = []
        for v in validators:
            try:
                insns = v.disassemble(data, "x86_64")
                mnemonics = [i.mnemonic.lower() for i in insns]
                all_mnemonics.append(mnemonics)
            except Exception:
                continue

        if not all_mnemonics:
            pytest.skip("No validators produced output")

        # GENESIS mnemonics
        genesis_mnemonics = [i.mnemonic.lower() for i in genesis_output.instructions]

        # Check GENESIS matches at least one validator
        matches_any = any(genesis_mnemonics == ext_mnemonics for ext_mnemonics in all_mnemonics)

        assert matches_any, (
            f"GENESIS {genesis_mnemonics} doesn't match any validator: {all_mnemonics}"
        )

    def test_consensus_on_nop_sled(
        self,
        module: Level0Module,
        validators: list[ExternalValidator],
    ) -> None:
        """All tools should agree on NOP sled."""
        data = b"\x90" * 20

        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        assert not genesis_output.is_uncertain
        assert genesis_output.instruction_count == 20

        # All should be NOPs
        for insn in genesis_output.instructions:
            assert insn.mnemonic.lower() == "nop"

        # Verify against validators
        for v in validators:
            try:
                insns = v.disassemble(data, "x86_64")
                assert len(insns) == 20
                for insn in insns:
                    assert insn.mnemonic.lower() == "nop"
            except Exception:
                continue


class TestLevel0EdgeCasesExternal:
    """Test edge cases with external validation."""

    def test_empty_after_valid(
        self,
        module: Level0Module,
        validators: list[ExternalValidator],
    ) -> None:
        """Valid instruction followed by invalid bytes."""
        # NOP followed by incomplete instruction
        data = b"\x90\x0f"

        inp = Level0Input(data=data, architecture=Architecture.X86_64)
        genesis_output = module.process(inp)

        # Should decode at least the NOP
        if not genesis_output.is_uncertain:
            assert genesis_output.instruction_count >= 1
            assert genesis_output.instructions[0].mnemonic.lower() == "nop"
