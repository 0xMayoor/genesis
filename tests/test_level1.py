"""Tests for Level 1: Assembly Semantics."""

import pytest
from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly import (
    Level1Module,
    Level1Input,
    Level1Output,
    EffectOperation,
    FlagOperation,
    ControlFlowType,
)


@pytest.fixture
def module():
    """Create Level 1 module."""
    return Level1Module()


def make_instr(mnemonic: str, operands: list[str]) -> Instruction:
    """Helper to create instruction."""
    return Instruction(
        offset=0,
        raw_bytes=b"\x90",  # Placeholder bytes
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )


class TestDataMovement:
    """Test data movement instructions."""
    
    def test_mov_reg_reg(self, module):
        """MOV rax, rbx - should read rbx, write rax, no flags."""
        instr = make_instr("mov", ["rax", "rbx"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert "rax" in result.writes_registers
        assert "rbx" in result.reads_registers
        assert result.flag_effects == []
    
    def test_push_reg(self, module):
        """PUSH rax - should decrement rsp, write to stack."""
        instr = make_instr("push", ["rax"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert "rsp" in result.writes_registers
        assert "rax" in result.reads_registers
        assert len(result.memory_effects) == 1
        assert result.memory_effects[0].operation == EffectOperation.WRITE
    
    def test_pop_reg(self, module):
        """POP rax - should read from stack, increment rsp."""
        instr = make_instr("pop", ["rax"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert "rax" in result.writes_registers
        assert "rsp" in result.writes_registers
        assert len(result.memory_effects) == 1
        assert result.memory_effects[0].operation == EffectOperation.READ
    
    def test_lea(self, module):
        """LEA - should compute address, not access memory."""
        instr = make_instr("lea", ["rax", "[rbx + rcx*4]"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert "rax" in result.writes_registers
        assert result.memory_effects == []  # LEA doesn't access memory
    
    def test_xchg(self, module):
        """XCHG - should swap two registers."""
        instr = make_instr("xchg", ["rax", "rbx"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        # Both should be read-write
        assert any(e.register == "rax" and e.operation == EffectOperation.READ_WRITE 
                   for e in result.register_effects)
        assert any(e.register == "rbx" and e.operation == EffectOperation.READ_WRITE 
                   for e in result.register_effects)


class TestArithmetic:
    """Test arithmetic instructions."""
    
    def test_add_sets_flags(self, module):
        """ADD - should modify all arithmetic flags."""
        instr = make_instr("add", ["rax", "rbx"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        flags = result.modifies_flags
        assert "ZF" in flags
        assert "SF" in flags
        assert "CF" in flags
        assert "OF" in flags
    
    def test_inc_preserves_cf(self, module):
        """INC - should NOT modify CF (important edge case!)."""
        instr = make_instr("inc", ["rax"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        # CF should NOT be in modified flags for INC
        cf_effects = [e for e in result.flag_effects if e.flag == "CF"]
        assert len(cf_effects) == 0 or cf_effects[0].operation == FlagOperation.UNCHANGED
    
    def test_xor_self_clears_flags(self, module):
        """XOR rax, rax - clears CF and OF."""
        instr = make_instr("xor", ["rax", "rax"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        cf = next(e for e in result.flag_effects if e.flag == "CF")
        of = next(e for e in result.flag_effects if e.flag == "OF")
        assert cf.operation == FlagOperation.CLEAR
        assert of.operation == FlagOperation.CLEAR


class TestComparison:
    """Test comparison instructions."""
    
    def test_cmp_only_reads(self, module):
        """CMP - should only read operands, not write."""
        instr = make_instr("cmp", ["rax", "rbx"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        # CMP reads but doesn't write
        assert result.writes_registers == []
        assert "rax" in result.reads_registers
        assert "rbx" in result.reads_registers
        # But it sets flags
        assert len(result.flag_effects) > 0
    
    def test_test_clears_cf_of(self, module):
        """TEST - should clear CF and OF."""
        instr = make_instr("test", ["rax", "rax"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        cf = next(e for e in result.flag_effects if e.flag == "CF")
        of = next(e for e in result.flag_effects if e.flag == "OF")
        assert cf.operation == FlagOperation.CLEAR
        assert of.operation == FlagOperation.CLEAR


class TestControlFlow:
    """Test control flow instructions."""
    
    def test_jmp_unconditional(self, module):
        """JMP - unconditional jump."""
        instr = make_instr("jmp", ["0x1000"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert result.control_flow.type == ControlFlowType.JUMP
        assert result.control_flow.condition is None
    
    def test_je_conditional(self, module):
        """JE - conditional jump with ZF condition."""
        instr = make_instr("je", ["0x1000"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert result.control_flow.type == ControlFlowType.CONDITIONAL
        assert "ZF" in result.control_flow.condition
    
    def test_call_pushes_return(self, module):
        """CALL - should push return address."""
        instr = make_instr("call", ["0x1000"])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert result.control_flow.type == ControlFlowType.CALL
        assert "rsp" in result.writes_registers
        assert len(result.memory_effects) == 1
        assert result.memory_effects[0].operation == EffectOperation.WRITE
    
    def test_ret_pops_return(self, module):
        """RET - should pop return address."""
        instr = make_instr("ret", [])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert result.control_flow.type == ControlFlowType.RETURN
        assert "rsp" in result.writes_registers
        assert len(result.memory_effects) == 1
        assert result.memory_effects[0].operation == EffectOperation.READ


class TestStackFrame:
    """Test stack frame instructions."""
    
    def test_leave(self, module):
        """LEAVE - mov rsp, rbp; pop rbp."""
        instr = make_instr("leave", [])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain
        assert "rsp" in result.writes_registers
        assert "rbp" in result.writes_registers


class TestAdversarial:
    """Test adversarial cases."""
    
    def test_unknown_instruction(self, module):
        """Unknown instruction should be refused."""
        instr = make_instr("invalid_opcode", [])
        result = module.analyze(Level1Input(instruction=instr))
        
        assert result.is_uncertain
        assert result.confidence == 0.0
    
    def test_wrong_operand_count(self, module):
        """Wrong operand count should be refused."""
        instr = make_instr("mov", ["rax"])  # MOV needs 2 operands
        result = module.analyze(Level1Input(instruction=instr))
        
        assert result.is_uncertain


class TestIntegration:
    """Test integration with Level 0."""
    
    def test_from_level0_input(self, module):
        """Test creating Level1Input from Level0 output."""
        instr = make_instr("add", ["rax", "rbx"])
        l1_input = Level1Input.from_level0(instr, "x86_64")
        result = module.analyze(l1_input)
        
        assert not result.is_uncertain
        assert result.instruction == instr


class TestOutput:
    """Test output format."""
    
    def test_to_dict(self, module):
        """Test output serialization."""
        instr = make_instr("add", ["rax", "rbx"])
        result = module.analyze(Level1Input(instruction=instr))
        
        d = result.to_dict()
        assert "instruction" in d
        assert "register_effects" in d
        assert "flag_effects" in d
        assert "control_flow" in d
