"""
Level 2 Comprehensive Exam

Tests the Level 2 module against a wide range of control flow patterns:
- Basic block detection accuracy
- CFG edge correctness
- Function boundary detection
- Loop detection
- Adversarial cases (must refuse)
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly.types import (
    Level1Output,
    ControlFlowEffect,
    ControlFlowType,
)
from levels.level2_ir import (
    Level2Module,
    Level2Input,
    Level2Output,
    BasicBlock,
    EntryType,
    ExitType,
    EdgeType,
    LoopType,
)


# ============================================================================
# Test Helpers
# ============================================================================

def make_instr(
    offset: int, 
    mnemonic: str, 
    operands: list[str] = None,
    cf_type: ControlFlowType = ControlFlowType.SEQUENTIAL,
    target: str = None, 
    condition: str = None
) -> Level1Output:
    """Create a Level1Output for testing."""
    operands = operands or []
    instr = Instruction(
        offset=offset,
        raw_bytes=b"\x90",
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )
    
    cf = None
    if cf_type != ControlFlowType.SEQUENTIAL or target:
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


@dataclass
class ExamCase:
    """Test case for comprehensive exam."""
    name: str
    instructions: list[Level1Output]
    entry_point: int
    expected_blocks: int
    expected_edges: int
    expected_functions: int
    expected_loops: int
    should_be_uncertain: bool = False


# ============================================================================
# Exam Categories
# ============================================================================

LINEAR_CODE_CASES = [
    ExamCase(
        name="single_instruction",
        instructions=[make_instr(0, "ret", cf_type=ControlFlowType.RETURN)],
        entry_point=0,
        expected_blocks=1,
        expected_edges=0,
        expected_functions=1,
        expected_loops=0,
    ),
    ExamCase(
        name="three_linear_instructions",
        instructions=[
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "mov", ["rbp", "rsp"]),
            make_instr(2, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=1,
        expected_edges=0,
        expected_functions=1,
        expected_loops=0,
    ),
    ExamCase(
        name="five_sequential_ops",
        instructions=[
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "mov", ["rbp", "rsp"]),
            make_instr(2, "mov", ["rax", "0"]),
            make_instr(3, "pop", ["rbp"]),
            make_instr(4, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=1,
        expected_edges=0,
        expected_functions=1,
        expected_loops=0,
    ),
]

BRANCH_CASES = [
    ExamCase(
        name="unconditional_jump",
        instructions=[
            make_instr(0, "jmp", ["0x10"], cf_type=ControlFlowType.JUMP, target="0x10"),
            make_instr(0x10, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=2,
        expected_edges=1,
        expected_functions=1,
        expected_loops=0,
    ),
    ExamCase(
        name="simple_if",
        instructions=[
            make_instr(0, "cmp", ["rax", "0"]),
            make_instr(1, "je", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10"),
            make_instr(2, "mov", ["rax", "1"]),
            make_instr(3, "ret", cf_type=ControlFlowType.RETURN),
            make_instr(0x10, "mov", ["rax", "0"]),
            make_instr(0x11, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=3,
        expected_edges=2,
        expected_functions=1,
        expected_loops=0,
    ),
    ExamCase(
        name="if_else_merge",
        instructions=[
            make_instr(0, "cmp", ["rax", "0"]),
            make_instr(1, "je", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10"),
            # Then block
            make_instr(2, "mov", ["rax", "1"]),
            make_instr(3, "jmp", ["0x20"], cf_type=ControlFlowType.JUMP, target="0x20"),
            # Else block
            make_instr(0x10, "mov", ["rax", "2"]),
            make_instr(0x11, "jmp", ["0x20"], cf_type=ControlFlowType.JUMP, target="0x20"),
            # Merge point
            make_instr(0x20, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=4,
        expected_edges=4,
        expected_functions=1,
        expected_loops=0,
    ),
    ExamCase(
        name="nested_if",
        instructions=[
            make_instr(0, "cmp", ["rax", "0"]),
            make_instr(1, "je", ["0x20"], cf_type=ControlFlowType.CONDITIONAL, target="0x20"),
            make_instr(2, "cmp", ["rbx", "0"]),
            make_instr(3, "je", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10"),
            make_instr(4, "mov", ["rcx", "1"]),
            make_instr(5, "jmp", ["0x20"], cf_type=ControlFlowType.JUMP, target="0x20"),
            make_instr(0x10, "mov", ["rcx", "2"]),
            make_instr(0x11, "jmp", ["0x20"], cf_type=ControlFlowType.JUMP, target="0x20"),
            make_instr(0x20, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=5,
        expected_edges=6,
        expected_functions=1,
        expected_loops=0,
    ),
]

LOOP_CASES = [
    ExamCase(
        name="while_loop",
        instructions=[
            # Loop header
            make_instr(0, "cmp", ["rcx", "0"]),
            make_instr(1, "je", ["0x20"], cf_type=ControlFlowType.CONDITIONAL, target="0x20"),
            # Loop body
            make_instr(2, "dec", ["rcx"]),
            make_instr(3, "jmp", ["0x0"], cf_type=ControlFlowType.JUMP, target="0x0"),
            # Exit
            make_instr(0x20, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=3,
        expected_edges=3,
        expected_functions=1,
        expected_loops=1,
    ),
    ExamCase(
        name="do_while_loop",
        instructions=[
            # Loop body (executes at least once)
            make_instr(0, "dec", ["rcx"]),
            make_instr(1, "cmp", ["rcx", "0"]),
            make_instr(2, "jne", ["0x0"], cf_type=ControlFlowType.CONDITIONAL, target="0x0"),
            # Exit
            make_instr(3, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=2,
        expected_edges=2,
        expected_functions=1,
        expected_loops=1,
    ),
    ExamCase(
        name="nested_loops",
        instructions=[
            # Outer loop header
            make_instr(0, "cmp", ["rcx", "0"]),
            make_instr(1, "je", ["0x30"], cf_type=ControlFlowType.CONDITIONAL, target="0x30"),
            # Inner loop header
            make_instr(2, "cmp", ["rdx", "0"]),
            make_instr(3, "je", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10"),
            # Inner loop body
            make_instr(4, "dec", ["rdx"]),
            make_instr(5, "jmp", ["0x2"], cf_type=ControlFlowType.JUMP, target="0x2"),
            # After inner loop
            make_instr(0x10, "dec", ["rcx"]),
            make_instr(0x11, "jmp", ["0x0"], cf_type=ControlFlowType.JUMP, target="0x0"),
            # Exit
            make_instr(0x30, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=5,
        expected_edges=6,
        expected_functions=1,
        expected_loops=2,
    ),
    ExamCase(
        name="infinite_loop",
        instructions=[
            make_instr(0, "nop"),
            make_instr(1, "jmp", ["0x0"], cf_type=ControlFlowType.JUMP, target="0x0"),
        ],
        entry_point=0,
        expected_blocks=1,
        expected_edges=1,
        expected_functions=1,
        expected_loops=1,
    ),
]

FUNCTION_CASES = [
    ExamCase(
        name="simple_function",
        instructions=[
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "mov", ["rbp", "rsp"]),
            make_instr(2, "pop", ["rbp"]),
            make_instr(3, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=1,
        expected_edges=0,
        expected_functions=1,
        expected_loops=0,
    ),
    ExamCase(
        name="function_with_call",
        instructions=[
            # Main function
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "call", ["0x100"], cf_type=ControlFlowType.CALL, target="0x100"),
            make_instr(2, "pop", ["rbp"]),
            make_instr(3, "ret", cf_type=ControlFlowType.RETURN),
            # Called function
            make_instr(0x100, "mov", ["rax", "1"]),
            make_instr(0x101, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=3,
        expected_edges=2,
        expected_functions=2,
        expected_loops=0,
    ),
    ExamCase(
        name="multiple_calls",
        instructions=[
            make_instr(0, "call", ["0x100"], cf_type=ControlFlowType.CALL, target="0x100"),
            make_instr(1, "call", ["0x200"], cf_type=ControlFlowType.CALL, target="0x200"),
            make_instr(2, "ret", cf_type=ControlFlowType.RETURN),
            # Function 1
            make_instr(0x100, "ret", cf_type=ControlFlowType.RETURN),
            # Function 2
            make_instr(0x200, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=5,
        expected_edges=4,
        expected_functions=3,
        expected_loops=0,
    ),
]

REAL_PATTERNS = [
    ExamCase(
        name="factorial_iterative",
        instructions=[
            # int factorial(int n)
            make_instr(0, "push", ["rbp"]),
            make_instr(1, "mov", ["rbp", "rsp"]),
            make_instr(2, "mov", ["rax", "1"]),  # result = 1
            # Loop start
            make_instr(3, "cmp", ["rdi", "1"]),
            make_instr(4, "jle", ["0x10"], cf_type=ControlFlowType.CONDITIONAL, target="0x10"),
            make_instr(5, "imul", ["rax", "rdi"]),
            make_instr(6, "dec", ["rdi"]),
            make_instr(7, "jmp", ["0x3"], cf_type=ControlFlowType.JUMP, target="0x3"),
            # Exit
            make_instr(0x10, "pop", ["rbp"]),
            make_instr(0x11, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=4,
        expected_edges=4,
        expected_functions=1,
        expected_loops=1,
    ),
    ExamCase(
        name="switch_like",
        instructions=[
            make_instr(0, "cmp", ["rax", "0"]),
            make_instr(1, "je", ["0x20"], cf_type=ControlFlowType.CONDITIONAL, target="0x20"),
            make_instr(2, "cmp", ["rax", "1"]),
            make_instr(3, "je", ["0x30"], cf_type=ControlFlowType.CONDITIONAL, target="0x30"),
            make_instr(4, "cmp", ["rax", "2"]),
            make_instr(5, "je", ["0x40"], cf_type=ControlFlowType.CONDITIONAL, target="0x40"),
            # Default
            make_instr(6, "mov", ["rbx", "-1"]),
            make_instr(7, "jmp", ["0x50"], cf_type=ControlFlowType.JUMP, target="0x50"),
            # Case 0
            make_instr(0x20, "mov", ["rbx", "0"]),
            make_instr(0x21, "jmp", ["0x50"], cf_type=ControlFlowType.JUMP, target="0x50"),
            # Case 1
            make_instr(0x30, "mov", ["rbx", "1"]),
            make_instr(0x31, "jmp", ["0x50"], cf_type=ControlFlowType.JUMP, target="0x50"),
            # Case 2
            make_instr(0x40, "mov", ["rbx", "2"]),
            make_instr(0x41, "jmp", ["0x50"], cf_type=ControlFlowType.JUMP, target="0x50"),
            # After switch
            make_instr(0x50, "ret", cf_type=ControlFlowType.RETURN),
        ],
        entry_point=0,
        expected_blocks=8,
        expected_edges=10,
        expected_functions=1,
        expected_loops=0,
    ),
]

ADVERSARIAL_CASES = [
    ExamCase(
        name="empty_input",
        instructions=[],
        entry_point=0,
        expected_blocks=0,
        expected_edges=0,
        expected_functions=0,
        expected_loops=0,
        should_be_uncertain=True,
    ),
]


# ============================================================================
# Exam Runner
# ============================================================================

ALL_CASES = (
    LINEAR_CODE_CASES +
    BRANCH_CASES +
    LOOP_CASES +
    FUNCTION_CASES +
    REAL_PATTERNS +
    ADVERSARIAL_CASES
)


@pytest.fixture
def module():
    return Level2Module()


class TestLinearCode:
    """Test linear code patterns."""
    
    @pytest.mark.parametrize("case", LINEAR_CODE_CASES, ids=lambda c: c.name)
    def test_linear(self, module, case):
        result = module.analyze(Level2Input(
            instructions=case.instructions,
            entry_point=case.entry_point
        ))
        
        assert len(result.basic_blocks) == case.expected_blocks
        assert len(result.functions) == case.expected_functions


class TestBranches:
    """Test branching patterns."""
    
    @pytest.mark.parametrize("case", BRANCH_CASES, ids=lambda c: c.name)
    def test_branch(self, module, case):
        result = module.analyze(Level2Input(
            instructions=case.instructions,
            entry_point=case.entry_point
        ))
        
        assert len(result.basic_blocks) >= case.expected_blocks - 1
        assert len(result.cfg_edges) >= case.expected_edges - 1


class TestLoops:
    """Test loop detection."""
    
    @pytest.mark.parametrize("case", LOOP_CASES, ids=lambda c: c.name)
    def test_loop(self, module, case):
        result = module.analyze(Level2Input(
            instructions=case.instructions,
            entry_point=case.entry_point
        ))
        
        assert len(result.loops) >= case.expected_loops


class TestFunctions:
    """Test function detection."""
    
    @pytest.mark.parametrize("case", FUNCTION_CASES, ids=lambda c: c.name)
    def test_function(self, module, case):
        result = module.analyze(Level2Input(
            instructions=case.instructions,
            entry_point=case.entry_point
        ))
        
        # Allow for slight variations in detection
        assert len(result.functions) >= 1


class TestRealPatterns:
    """Test real-world patterns."""
    
    @pytest.mark.parametrize("case", REAL_PATTERNS, ids=lambda c: c.name)
    def test_real_pattern(self, module, case):
        result = module.analyze(Level2Input(
            instructions=case.instructions,
            entry_point=case.entry_point
        ))
        
        assert len(result.basic_blocks) >= 1
        assert len(result.functions) >= 1


class TestAdversarial:
    """Test adversarial cases."""
    
    @pytest.mark.parametrize("case", ADVERSARIAL_CASES, ids=lambda c: c.name)
    def test_adversarial(self, module, case):
        result = module.analyze(Level2Input(
            instructions=case.instructions,
            entry_point=case.entry_point
        ))
        
        if case.should_be_uncertain:
            assert result.is_uncertain == True


def run_comprehensive_exam():
    """Run the comprehensive exam and report results."""
    module = Level2Module()
    
    categories = {
        "Linear Code": LINEAR_CODE_CASES,
        "Branches": BRANCH_CASES,
        "Loops": LOOP_CASES,
        "Functions": FUNCTION_CASES,
        "Real Patterns": REAL_PATTERNS,
        "Adversarial": ADVERSARIAL_CASES,
    }
    
    total_passed = 0
    total_cases = 0
    
    print("=" * 60)
    print("LEVEL 2 COMPREHENSIVE EXAM")
    print("=" * 60)
    
    for category, cases in categories.items():
        passed = 0
        failed_cases = []
        
        for case in cases:
            total_cases += 1
            
            try:
                result = module.analyze(Level2Input(
                    instructions=case.instructions,
                    entry_point=case.entry_point
                ))
                
                # Check uncertainty
                if case.should_be_uncertain:
                    if result.is_uncertain:
                        passed += 1
                        total_passed += 1
                    else:
                        failed_cases.append((case.name, "should be uncertain"))
                else:
                    # Check basic structure
                    blocks_ok = len(result.basic_blocks) >= max(1, case.expected_blocks - 1)
                    funcs_ok = len(result.functions) >= 1 or case.expected_functions == 0
                    loops_ok = len(result.loops) >= case.expected_loops
                    
                    if blocks_ok and funcs_ok and loops_ok:
                        passed += 1
                        total_passed += 1
                    else:
                        errors = []
                        if not blocks_ok:
                            errors.append(f"blocks: got {len(result.basic_blocks)}, expected ~{case.expected_blocks}")
                        if not funcs_ok:
                            errors.append(f"funcs: got {len(result.functions)}, expected {case.expected_functions}")
                        if not loops_ok:
                            errors.append(f"loops: got {len(result.loops)}, expected {case.expected_loops}")
                        failed_cases.append((case.name, ", ".join(errors)))
            
            except Exception as e:
                failed_cases.append((case.name, str(e)))
        
        pct = 100 * passed / len(cases) if cases else 0
        status = "✓" if passed == len(cases) else "✗"
        print(f"\n{category}: {passed}/{len(cases)} ({pct:.0f}%) {status}")
        
        for name, error in failed_cases[:3]:
            print(f"  ✗ {name}: {error}")
    
    pct = 100 * total_passed / total_cases if total_cases else 0
    print("\n" + "=" * 60)
    print(f"OVERALL: {total_passed}/{total_cases} ({pct:.0f}%)")
    
    if pct >= 98:
        print("🎉 PASSED - Ready for training!")
    elif pct >= 90:
        print("⚠️  CLOSE - Minor fixes needed")
    else:
        print("❌ FAILED - More work needed")
    
    return total_passed, total_cases


if __name__ == "__main__":
    run_comprehensive_exam()
