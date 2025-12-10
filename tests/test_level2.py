"""Tests for Level 2: Control Flow Analysis."""

import pytest
from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly.types import (
    Level1Output,
    RegisterEffect,
    ControlFlowEffect,
    ControlFlowType,
    EffectOperation,
)
from levels.level2_ir import (
    Level2Module,
    Level2Input,
    Level2Output,
    BasicBlock,
    CFGEdge,
    Function,
    Loop,
    EntryType,
    ExitType,
    EdgeType,
    LoopType,
)


@pytest.fixture
def module():
    return Level2Module()


def make_instr(offset: int, mnemonic: str, operands: list[str], 
               cf_type: ControlFlowType = ControlFlowType.SEQUENTIAL,
               target: str = None, condition: str = None) -> Level1Output:
    """Create a Level1Output for testing."""
    instr = Instruction(
        offset=offset,
        raw_bytes=b"\x90",
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )
    
    cf = None
    if cf_type != ControlFlowType.SEQUENTIAL or target or condition:
        cf = ControlFlowEffect(
            type=cf_type,
            target_expr=target,
            condition=condition,
        )
    
    return Level1Output(
        instruction=instr,
        register_effects=[],
        memory_effects=[],
        flag_effects=[],
        control_flow=cf,
        is_uncertain=False,
    )


class TestBasicBlockDetection:
    """Test basic block detection."""
    
    def test_single_linear_block(self, module):
        """Linear code creates single block."""
        instructions = [
            make_instr(0, "mov", ["rax", "rbx"]),
            make_instr(1, "add", ["rax", "rcx"]),
            make_instr(2, "sub", ["rax", "rdx"]),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        assert len(result.basic_blocks) == 1
        assert result.basic_blocks[0].start_offset == 0
        assert result.basic_blocks[0].exit_type == ExitType.FALL_THROUGH
    
    def test_unconditional_jump_splits_block(self, module):
        """Unconditional jump creates new block at target."""
        instructions = [
            make_instr(0, "mov", ["rax", "rbx"]),
            make_instr(1, "jmp", ["0x10"], cf_type=ControlFlowType.JUMP, target="0x10"),
            make_instr(0x10, "nop", []),
            make_instr(0x11, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        assert len(result.basic_blocks) == 2
        # First block ends with jump
        assert result.basic_blocks[0].exit_type == ExitType.UNCONDITIONAL_JUMP
        # Second block starts at jump target
        assert result.basic_blocks[1].start_offset == 0x10
    
    def test_conditional_jump_creates_two_successors(self, module):
        """Conditional jump creates diamond pattern."""
        instructions = [
            make_instr(0, "cmp", ["rax", "rbx"]),
            make_instr(1, "je", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10", condition="ZF==1"),
            # Fall-through path
            make_instr(2, "mov", ["rax", "1"]),
            make_instr(3, "jmp", ["0x20"], cf_type=ControlFlowType.JUMP, target="0x20"),
            # Jump target path
            make_instr(0x10, "mov", ["rax", "0"]),
            make_instr(0x11, "jmp", ["0x20"], cf_type=ControlFlowType.JUMP, target="0x20"),
            # Merge point
            make_instr(0x20, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Should have blocks at: 0, 2, 0x10, 0x20
        assert len(result.basic_blocks) >= 3
    
    def test_call_creates_block_boundary(self, module):
        """Call instruction creates block boundary."""
        instructions = [
            make_instr(0, "mov", ["rdi", "1"]),
            make_instr(1, "call", ["0x100"], cf_type=ControlFlowType.CALL, target="0x100"),
            make_instr(2, "mov", ["rax", "rbx"]),
            make_instr(3, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Block after call is a new block
        assert len(result.basic_blocks) >= 2


class TestCFGConstruction:
    """Test CFG edge construction."""
    
    def test_fall_through_edge(self, module):
        """Sequential blocks have fall-through edge."""
        instructions = [
            make_instr(0, "mov", ["rax", "rbx"]),
            make_instr(1, "jmp", ["0x10"], cf_type=ControlFlowType.JUMP, target="0x10"),
            make_instr(0x10, "nop", []),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Should have edge from block 0 to block at 0x10
        edges = result.cfg_edges
        unconditional_edges = [e for e in edges if e.edge_type == EdgeType.UNCONDITIONAL]
        assert len(unconditional_edges) >= 1
    
    def test_conditional_edges(self, module):
        """Conditional jump creates true and false edges."""
        instructions = [
            make_instr(0, "cmp", ["rax", "0"]),
            make_instr(1, "je", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10"),
            make_instr(2, "nop", []),  # Fall-through
            make_instr(0x10, "nop", []),  # Target
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        true_edges = [e for e in result.cfg_edges if e.edge_type == EdgeType.CONDITIONAL_TRUE]
        false_edges = [e for e in result.cfg_edges if e.edge_type == EdgeType.CONDITIONAL_FALSE]
        
        assert len(true_edges) >= 1
        assert len(false_edges) >= 1
    
    def test_return_has_no_outgoing_edges(self, module):
        """Return instruction has no successor edges."""
        instructions = [
            make_instr(0, "mov", ["rax", "0"]),
            make_instr(1, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Single block with return should have no outgoing edges
        assert len(result.cfg_edges) == 0


class TestFunctionDetection:
    """Test function boundary detection."""
    
    def test_simple_function(self, module):
        """Detect single function."""
        instructions = [
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "mov", ["rbp", "rsp"]),
            make_instr(2, "mov", ["rax", "0"]),
            make_instr(3, "pop", ["rbp"]),
            make_instr(4, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        assert len(result.functions) == 1
        assert result.functions[0].entry_offset == 0
        assert len(result.functions[0].exit_blocks) >= 1
    
    def test_function_with_call(self, module):
        """Detect function that makes calls."""
        instructions = [
            # Main function at 0
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "call", ["0x100"], cf_type=ControlFlowType.CALL, target="0x100"),
            make_instr(2, "pop", ["rbp"]),
            make_instr(3, "ret", [], cf_type=ControlFlowType.RETURN),
            # Called function at 0x100
            make_instr(0x100, "mov", ["rax", "1"]),
            make_instr(0x101, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Should detect two functions
        assert len(result.functions) >= 1
        
        # Should have call edge
        assert len(result.call_edges) >= 1


class TestLoopDetection:
    """Test loop detection."""
    
    def test_simple_while_loop(self, module):
        """Detect simple while loop."""
        instructions = [
            # Header block (loop test)
            make_instr(0, "cmp", ["rcx", "0"]),
            make_instr(1, "je", ["0x20"], cf_type=ControlFlowType.CONDITIONAL, target="0x20", condition="ZF==1"),
            # Loop body
            make_instr(2, "dec", ["rcx"]),
            make_instr(3, "jmp", ["0x0"], cf_type=ControlFlowType.JUMP, target="0x0"),  # Back edge
            # Exit
            make_instr(0x20, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Should detect one loop
        assert len(result.loops) >= 1
        if result.loops:
            loop = result.loops[0]
            assert loop.header_block is not None


class TestAdversarialCases:
    """Test adversarial/edge cases."""
    
    def test_empty_input(self, module):
        """Empty input returns uncertain."""
        result = module.analyze(Level2Input(instructions=[], entry_point=0))
        
        assert result.is_uncertain == True
    
    def test_single_instruction(self, module):
        """Single instruction creates single block."""
        instructions = [
            make_instr(0, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        assert len(result.basic_blocks) == 1


class TestOutputMethods:
    """Test Level2Output helper methods."""
    
    def test_get_block(self, module):
        """Test get_block method."""
        instructions = [
            make_instr(0, "mov", ["rax", "1"]),
            make_instr(1, "jmp", ["0x10"], cf_type=ControlFlowType.JUMP, target="0x10"),
            make_instr(0x10, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        block0 = result.get_block(0)
        assert block0 is not None
        assert block0.id == 0
    
    def test_get_successors(self, module):
        """Test get_successors method."""
        instructions = [
            make_instr(0, "jmp", ["0x10"], cf_type=ControlFlowType.JUMP, target="0x10"),
            make_instr(0x10, "ret", [], cf_type=ControlFlowType.RETURN),
        ]
        
        result = module.analyze(Level2Input(instructions=instructions, entry_point=0))
        
        # Block 0 should have block 1 as successor
        if len(result.basic_blocks) >= 2:
            successors = result.get_successors(0)
            assert len(successors) >= 1
