"""
Property-based tests for Level 1 using Hypothesis.

These tests generate random inputs to verify:
1. Module never crashes on any input
2. Output structure is always valid
3. Uncertain inputs always return is_uncertain=True
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly import (
    Level1Module,
    Level1Input,
    Level1Output,
    EffectOperation,
    ControlFlowType,
)

# Shared module instance (stateless, safe to reuse)
MODULE = Level1Module()

# Valid x86_64 registers for property tests
VALID_REGISTERS = [
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
    "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
    "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh",
]

VALID_MNEMONICS = [
    "mov", "add", "sub", "xor", "and", "or", "not",
    "push", "pop", "call", "ret", "jmp", "je", "jne",
    "cmp", "test", "lea", "inc", "dec", "neg",
    "shl", "shr", "mul", "div", "imul", "idiv",
]

INVALID_MNEMONICS = [
    "fakeinstr", "notreal", "xyz", "invalidop",
    "", "mov!", "@add", "123",
]


def make_input(mnemonic: str, operands: list[str]) -> Level1Input:
    """Create a Level1Input for testing."""
    instr = Instruction(
        offset=0,
        raw_bytes=b"\x90",
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )
    return Level1Input(instruction=instr)


class TestModuleNeverCrashes:
    """Property: Module should never crash on any input."""
    
    @given(
        mnemonic=st.text(min_size=0, max_size=20),
        operands=st.lists(st.text(min_size=0, max_size=30), min_size=0, max_size=4)
    )
    @settings(max_examples=200)
    def test_random_input_no_crash(self, mnemonic, operands):
        """Module handles arbitrary string input without crashing."""
        instr = make_input(mnemonic, operands)
        result = MODULE.analyze(instr)
        
        # Should always return a Level1Output
        assert isinstance(result, Level1Output)
    
    @given(
        mnemonic=st.sampled_from(VALID_MNEMONICS),
        operands=st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=3)
    )
    @settings(max_examples=100)
    def test_valid_mnemonic_random_operands(self, mnemonic, operands):
        """Valid mnemonic with random operands doesn't crash."""
        instr = make_input(mnemonic, operands)
        result = MODULE.analyze(instr)
        
        assert isinstance(result, Level1Output)


class TestOutputStructure:
    """Property: Output structure should always be valid."""
    
    @given(
        mnemonic=st.sampled_from(VALID_MNEMONICS),
        reg1=st.sampled_from(VALID_REGISTERS),
        reg2=st.sampled_from(VALID_REGISTERS),
    )
    @settings(max_examples=100)
    def test_output_has_required_fields(self, mnemonic, reg1, reg2):
        """Output always has required structure."""
        instr = make_input(mnemonic, [reg1, reg2])
        result = MODULE.analyze(instr)
        
        # Check structure
        assert hasattr(result, "register_effects")
        assert hasattr(result, "memory_effects")
        assert hasattr(result, "flag_effects")
        assert hasattr(result, "control_flow")
        assert hasattr(result, "is_uncertain")
        assert isinstance(result.is_uncertain, bool)
    
    @given(
        mnemonic=st.sampled_from(VALID_MNEMONICS),
        reg=st.sampled_from(VALID_REGISTERS),
    )
    @settings(max_examples=50)
    def test_register_effects_valid_operations(self, mnemonic, reg):
        """Register effects have valid operations."""
        instr = make_input(mnemonic, [reg])
        result = MODULE.analyze(instr)
        
        for effect in result.register_effects:
            assert effect.operation in EffectOperation


class TestUncertaintyHandling:
    """Property: Invalid/uncertain inputs should return is_uncertain=False."""
    
    @given(mnemonic=st.sampled_from(INVALID_MNEMONICS))
    @settings(max_examples=20)
    def test_invalid_mnemonic_is_uncertain(self, mnemonic):
        """Invalid mnemonics are marked uncertain."""
        instr = make_input(mnemonic, ["rax"])
        result = MODULE.analyze(instr)
        
        assert result.is_uncertain == True
    
    @given(
        mnemonic=st.sampled_from(["mov", "add", "sub"]),
        bad_operand=st.text(min_size=1, max_size=10).filter(
            lambda x: x.lower() not in VALID_REGISTERS and not x.startswith("0x")
        )
    )
    @settings(max_examples=50)
    def test_invalid_operands_handled(self, mnemonic, bad_operand):
        """Invalid operands are handled gracefully."""
        assume(not bad_operand.isdigit())  # Skip pure numbers (valid immediates)
        assume("[" not in bad_operand)  # Skip memory-like syntax
        
        instr = make_input(mnemonic, [bad_operand, bad_operand])
        result = MODULE.analyze(instr)
        
        # Should either be uncertain OR handle gracefully
        assert isinstance(result, Level1Output)


class TestDeterminism:
    """Property: Same input should always produce same output."""
    
    @given(
        mnemonic=st.sampled_from(VALID_MNEMONICS),
        reg1=st.sampled_from(VALID_REGISTERS[:8]),  # Limit for speed
        reg2=st.sampled_from(VALID_REGISTERS[:8]),
    )
    @settings(max_examples=50)
    def test_deterministic_output(self, mnemonic, reg1, reg2):
        """Same input produces identical output."""
        instr = make_input(mnemonic, [reg1, reg2])
        
        result1 = MODULE.analyze(instr)
        result2 = MODULE.analyze(instr)
        
        assert result1.is_uncertain == result2.is_uncertain
        assert len(result1.register_effects) == len(result2.register_effects)
        assert len(result1.flag_effects) == len(result2.flag_effects)


class TestControlFlowTypes:
    """Property: Control flow instructions have correct flow types."""
    
    @given(target=st.from_regex(r"0x[0-9a-f]{1,8}", fullmatch=True))
    @settings(max_examples=20)
    def test_jmp_is_unconditional(self, target):
        """JMP is always unconditional."""
        instr = make_input("jmp", [target])
        result = MODULE.analyze(instr)
        
        if not result.is_uncertain and result.control_flow:
            assert result.control_flow.type == ControlFlowType.JUMP
    
    @given(target=st.from_regex(r"0x[0-9a-f]{1,8}", fullmatch=True))
    @settings(max_examples=20)
    def test_call_is_call_type(self, target):
        """CALL has call flow type."""
        instr = make_input("call", [target])
        result = MODULE.analyze(instr)
        
        if not result.is_uncertain and result.control_flow:
            assert result.control_flow.type == ControlFlowType.CALL
    
    def test_ret_is_return_type(self):
        """RET has return flow type."""
        instr = make_input("ret", [])
        result = MODULE.analyze(instr)
        
        if not result.is_uncertain and result.control_flow:
            assert result.control_flow.type == ControlFlowType.RETURN
