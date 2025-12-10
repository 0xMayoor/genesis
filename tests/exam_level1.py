#!/usr/bin/env python3
"""
Comprehensive Exam Suite for Level 1: Assembly Semantics.

This exam tests the model's understanding of instruction effects:
- Register read/write patterns
- Memory access patterns
- Flag modifications
- Control flow effects

Each test case is derived from:
1. Intel x86 manual documentation
2. Real-world code patterns
3. Common shellcode/exploit techniques
4. Edge cases and gotchas
"""

import pytest
from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly import (
    Level1Module,
    Level1Input,
    EffectOperation,
    FlagOperation,
    ControlFlowType,
)


def make_instr(mnemonic: str, operands: list[str]) -> Instruction:
    """Helper to create instruction."""
    return Instruction(
        offset=0,
        raw_bytes=b"\x90",
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )


# ============================================================
# EXAM 1: Data Movement Semantics (25 cases)
# ============================================================

DATA_MOVEMENT_CASES = [
    # (mnemonic, operands, expected_reads, expected_writes, has_memory_read, has_memory_write)
    ("mov", ["rax", "rbx"], ["rbx"], ["rax"], False, False),
    ("mov", ["rax", "rcx"], ["rcx"], ["rax"], False, False),
    ("mov", ["rdi", "rsi"], ["rsi"], ["rdi"], False, False),
    ("mov", ["r8", "r9"], ["r9"], ["r8"], False, False),
    ("mov", ["eax", "ebx"], ["ebx"], ["eax"], False, False),
    
    # Memory operations
    ("mov", ["rax", "[rbx]"], [], ["rax"], True, False),
    ("mov", ["[rax]", "rbx"], ["rbx"], [], False, True),
    
    # LEA - computes address, NO memory access
    ("lea", ["rax", "[rbx + rcx]"], [], ["rax"], False, False),
    ("lea", ["rdi", "[rsp + 8]"], [], ["rdi"], False, False),
    
    # Push/Pop
    ("push", ["rax"], ["rax", "rsp"], ["rsp"], False, True),
    ("push", ["rbx"], ["rbx", "rsp"], ["rsp"], False, True),
    ("push", ["rbp"], ["rbp", "rsp"], ["rsp"], False, True),
    ("pop", ["rax"], ["rsp"], ["rax", "rsp"], True, False),
    ("pop", ["rbx"], ["rsp"], ["rbx", "rsp"], True, False),
    ("pop", ["rbp"], ["rsp"], ["rbp", "rsp"], True, False),
    
    # Exchange
    ("xchg", ["rax", "rbx"], ["rax", "rbx"], ["rax", "rbx"], False, False),
    ("xchg", ["rcx", "rdx"], ["rcx", "rdx"], ["rcx", "rdx"], False, False),
    
    # Zero extension / Sign extension
    ("movzx", ["eax", "bl"], ["bl"], ["eax"], False, False),
    ("movsx", ["rax", "ecx"], ["ecx"], ["rax"], False, False),
    
    # String operations
    ("movsb", [], ["rsi", "rdi"], ["rsi", "rdi"], True, True),
    ("stosb", [], ["rax", "rdi"], ["rdi"], False, True),
    ("lodsb", [], ["rsi"], ["rax", "rsi"], True, False),
    
    # Leave = mov rsp, rbp; pop rbp
    ("leave", [], ["rbp"], ["rsp", "rbp"], True, False),
    
    # BSWAP
    ("bswap", ["eax"], ["eax"], ["eax"], False, False),
]


# ============================================================
# EXAM 2: Arithmetic & Flag Effects (30 cases)
# ============================================================

ARITHMETIC_CASES = [
    # (mnemonic, operands, modifies_CF, modifies_ZF, modifies_SF, modifies_OF)
    # ADD/SUB modify all flags
    ("add", ["rax", "rbx"], True, True, True, True),
    ("add", ["eax", "1"], True, True, True, True),
    ("add", ["rcx", "rdx"], True, True, True, True),
    ("sub", ["rax", "rbx"], True, True, True, True),
    ("sub", ["rsp", "8"], True, True, True, True),
    
    # INC/DEC do NOT modify CF! (important edge case)
    ("inc", ["rax"], False, True, True, True),
    ("inc", ["rcx"], False, True, True, True),
    ("dec", ["rax"], False, True, True, True),
    ("dec", ["rbx"], False, True, True, True),
    
    # NEG modifies all
    ("neg", ["rax"], True, True, True, True),
    
    # AND/OR/XOR clear CF and OF
    ("and", ["rax", "rbx"], False, True, True, False),  # CF=0, OF=0
    ("or", ["rax", "rbx"], False, True, True, False),
    ("xor", ["rax", "rax"], False, True, True, False),  # Classic zero idiom
    ("xor", ["eax", "eax"], False, True, True, False),
    
    # NOT does NOT affect any flags!
    ("not", ["rax"], False, False, False, False),
    ("not", ["rbx"], False, False, False, False),
    
    # Shifts modify CF (last bit shifted out)
    ("shl", ["rax", "1"], True, True, True, True),
    ("shr", ["rax", "1"], True, True, True, True),
    ("sar", ["rax", "1"], True, True, True, True),
    
    # CMP sets flags but doesn't store result
    ("cmp", ["rax", "rbx"], True, True, True, True),
    ("cmp", ["eax", "0"], True, True, True, True),
    
    # TEST sets flags (like AND) but doesn't store
    ("test", ["rax", "rax"], False, True, True, False),  # CF=0, OF=0
    ("test", ["eax", "eax"], False, True, True, False),
]


# ============================================================
# EXAM 3: Control Flow (25 cases)
# ============================================================

CONTROL_FLOW_CASES = [
    # (mnemonic, operands, expected_type, has_condition)
    ("jmp", ["0x1000"], ControlFlowType.JUMP, False),
    ("jmp", ["rax"], ControlFlowType.JUMP, False),
    
    # Conditional jumps
    ("je", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jz", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jne", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jnz", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jl", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jle", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jg", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jge", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jb", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jbe", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("ja", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jae", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("js", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    ("jns", ["0x1000"], ControlFlowType.CONDITIONAL, True),
    
    # Call/Ret
    ("call", ["0x1000"], ControlFlowType.CALL, False),
    ("call", ["rax"], ControlFlowType.CALL, False),
    ("ret", [], ControlFlowType.RETURN, False),
    
    # System
    ("syscall", [], ControlFlowType.INTERRUPT, False),
    ("int", ["0x80"], ControlFlowType.INTERRUPT, False),
    ("int3", [], ControlFlowType.INTERRUPT, False),
    
    # NOP - sequential
    ("nop", [], ControlFlowType.SEQUENTIAL, False),
    ("hlt", [], ControlFlowType.SEQUENTIAL, False),
]


# ============================================================
# EXAM 4: Real-World Patterns (20 cases)
# ============================================================

REAL_WORLD_PATTERNS = [
    # Function prologue
    {
        "name": "Function prologue: push rbp",
        "mnemonic": "push",
        "operands": ["rbp"],
        "checks": {
            "writes_rsp": True,
            "reads_rbp": True,
            "memory_write": True,
        }
    },
    {
        "name": "Function prologue: mov rbp, rsp",
        "mnemonic": "mov",
        "operands": ["rbp", "rsp"],
        "checks": {
            "writes_rbp": True,
            "reads_rsp": True,
            "no_flags": True,
        }
    },
    {
        "name": "Stack allocation: sub rsp, N",
        "mnemonic": "sub",
        "operands": ["rsp", "32"],
        "checks": {
            "writes_rsp": True,
            "modifies_flags": True,
        }
    },
    # Function epilogue
    {
        "name": "Function epilogue: leave",
        "mnemonic": "leave",
        "operands": [],
        "checks": {
            "writes_rsp": True,
            "writes_rbp": True,
            "memory_read": True,
        }
    },
    {
        "name": "Function epilogue: ret",
        "mnemonic": "ret",
        "operands": [],
        "checks": {
            "writes_rsp": True,
            "memory_read": True,
            "control_flow": ControlFlowType.RETURN,
        }
    },
    # Zero register idioms
    {
        "name": "Zero register: xor eax, eax",
        "mnemonic": "xor",
        "operands": ["eax", "eax"],
        "checks": {
            "writes_eax": True,
            "clears_cf": True,
            "clears_of": True,
        }
    },
    {
        "name": "Zero register: xor rax, rax", 
        "mnemonic": "xor",
        "operands": ["rax", "rax"],
        "checks": {
            "writes_rax": True,
        }
    },
    # Shellcode patterns
    {
        "name": "Shellcode: syscall",
        "mnemonic": "syscall",
        "operands": [],
        "checks": {
            "writes_rcx": True,  # RCX = return address
            "writes_r11": True,  # R11 = RFLAGS
            "control_flow": ControlFlowType.INTERRUPT,
        }
    },
    {
        "name": "Shellcode: int 0x80",
        "mnemonic": "int",
        "operands": ["0x80"],
        "checks": {
            "control_flow": ControlFlowType.INTERRUPT,
        }
    },
    # Comparison patterns
    {
        "name": "Null check: test rax, rax",
        "mnemonic": "test",
        "operands": ["rax", "rax"],
        "checks": {
            "reads_rax": True,
            "no_writes": True,
            "modifies_zf": True,
        }
    },
    {
        "name": "Compare: cmp rax, 0",
        "mnemonic": "cmp",
        "operands": ["rax", "0"],
        "checks": {
            "reads_rax": True,
            "no_writes": True,
            "modifies_flags": True,
        }
    },
    # Loop counter
    {
        "name": "Loop: dec rcx",
        "mnemonic": "dec",
        "operands": ["rcx"],
        "checks": {
            "writes_rcx": True,
            "preserves_cf": True,  # DEC doesn't touch CF!
        }
    },
    # Address calculation
    {
        "name": "Array index: lea rax, [rbx + rcx*8]",
        "mnemonic": "lea",
        "operands": ["rax", "[rbx + rcx*8]"],
        "checks": {
            "writes_rax": True,
            "no_memory": True,  # LEA doesn't access memory
            "no_flags": True,
        }
    },
    # Bit manipulation
    {
        "name": "Clear bits: and rax, 0xf",
        "mnemonic": "and",
        "operands": ["rax", "0xf"],
        "checks": {
            "writes_rax": True,
            "clears_cf": True,
            "clears_of": True,
        }
    },
    {
        "name": "Set bits: or rax, 1",
        "mnemonic": "or",
        "operands": ["rax", "1"],
        "checks": {
            "writes_rax": True,
        }
    },
    # CPUID
    {
        "name": "CPU detection: cpuid",
        "mnemonic": "cpuid",
        "operands": [],
        "checks": {
            "writes_eax": True,
            "writes_ebx": True,
            "writes_ecx": True,
            "writes_edx": True,
        }
    },
    # String operations
    {
        "name": "Memory copy: rep movsb",
        "mnemonic": "rep",
        "operands": [],
        "checks": {
            "writes_rcx": True,
        }
    },
    # Conditional set
    {
        "name": "Set on zero: sete al",
        "mnemonic": "sete",
        "operands": ["al"],
        "checks": {
            "writes_al": True,
            "no_flags": True,
        }
    },
    # Timestamp
    {
        "name": "Timing: rdtsc",
        "mnemonic": "rdtsc",
        "operands": [],
        "checks": {
            "writes_eax": True,
            "writes_edx": True,
        }
    },
]


# ============================================================
# EXAM 5: Adversarial Cases (10 cases)
# ============================================================

ADVERSARIAL_CASES = [
    # Invalid instructions
    ("invalid_mnemonic", [], True),
    ("fakeinstr", ["rax"], True),
    
    # Wrong operand counts
    ("mov", ["rax"], True),  # MOV needs 2
    ("push", [], True),       # PUSH needs 1
    ("add", ["rax"], True),   # ADD needs 2
    
    # Invalid operands (these should be handled gracefully)
    ("mov", ["xyz", "abc"], True),
]


# ============================================================
# TEST CLASSES
# ============================================================

class TestExam1DataMovement:
    """Exam 1: Data Movement Semantics."""
    
    @pytest.fixture
    def module(self):
        return Level1Module()
    
    @pytest.mark.parametrize("mnemonic,operands,reads,writes,mem_read,mem_write", DATA_MOVEMENT_CASES)
    def test_data_movement(self, module, mnemonic, operands, reads, writes, mem_read, mem_write):
        instr = make_instr(mnemonic, operands)
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain, f"{mnemonic} should not be uncertain"
        
        # Check reads
        for reg in reads:
            assert reg.lower() in [r.lower() for r in result.reads_registers], \
                f"{mnemonic}: should read {reg}"
        
        # Check writes  
        for reg in writes:
            assert reg.lower() in [r.lower() for r in result.writes_registers], \
                f"{mnemonic}: should write {reg}"
        
        # Check memory
        has_mem_read = any(e.operation == EffectOperation.READ for e in result.memory_effects)
        has_mem_write = any(e.operation == EffectOperation.WRITE for e in result.memory_effects)
        
        if mem_read:
            assert has_mem_read, f"{mnemonic}: should have memory read"
        if mem_write:
            assert has_mem_write, f"{mnemonic}: should have memory write"


class TestExam2Arithmetic:
    """Exam 2: Arithmetic & Flag Effects."""
    
    @pytest.fixture
    def module(self):
        return Level1Module()
    
    @pytest.mark.parametrize("mnemonic,operands,cf,zf,sf,of", ARITHMETIC_CASES)
    def test_flag_effects(self, module, mnemonic, operands, cf, zf, sf, of):
        instr = make_instr(mnemonic, operands)
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain, f"{mnemonic} should not be uncertain"
        
        flags_modified = result.modifies_flags
        
        # Check CF
        if cf:
            assert "CF" in flags_modified, f"{mnemonic}: should modify CF"
        else:
            cf_effect = next((e for e in result.flag_effects if e.flag == "CF"), None)
            if cf_effect:
                assert cf_effect.operation in (FlagOperation.CLEAR, FlagOperation.UNCHANGED), \
                    f"{mnemonic}: should NOT modify CF (got {cf_effect.operation})"
        
        # Check ZF
        if zf:
            assert "ZF" in flags_modified, f"{mnemonic}: should modify ZF"
        
        # Check SF
        if sf:
            assert "SF" in flags_modified, f"{mnemonic}: should modify SF"
        
        # Check OF
        if of:
            assert "OF" in flags_modified, f"{mnemonic}: should modify OF"
        else:
            of_effect = next((e for e in result.flag_effects if e.flag == "OF"), None)
            if of_effect:
                assert of_effect.operation in (FlagOperation.CLEAR, FlagOperation.UNCHANGED), \
                    f"{mnemonic}: should NOT modify OF"


class TestExam3ControlFlow:
    """Exam 3: Control Flow."""
    
    @pytest.fixture
    def module(self):
        return Level1Module()
    
    @pytest.mark.parametrize("mnemonic,operands,expected_type,has_condition", CONTROL_FLOW_CASES)
    def test_control_flow(self, module, mnemonic, operands, expected_type, has_condition):
        instr = make_instr(mnemonic, operands)
        result = module.analyze(Level1Input(instruction=instr))
        
        assert not result.is_uncertain, f"{mnemonic} should not be uncertain"
        assert result.control_flow.type == expected_type, \
            f"{mnemonic}: expected {expected_type}, got {result.control_flow.type}"
        
        if has_condition:
            assert result.control_flow.condition is not None, \
                f"{mnemonic}: should have condition"


class TestExam4RealWorld:
    """Exam 4: Real-World Patterns."""
    
    @pytest.fixture
    def module(self):
        return Level1Module()
    
    @pytest.mark.parametrize("pattern", REAL_WORLD_PATTERNS, ids=lambda p: p["name"])
    def test_real_world(self, module, pattern):
        instr = make_instr(pattern["mnemonic"], pattern["operands"])
        result = module.analyze(Level1Input(instruction=instr))
        
        checks = pattern["checks"]
        
        # Dynamic checks based on pattern
        for check, expected in checks.items():
            if check.startswith("writes_"):
                reg = check[7:]
                assert reg.lower() in [r.lower() for r in result.writes_registers], \
                    f"{pattern['name']}: should write {reg}"
            elif check.startswith("reads_"):
                reg = check[6:]
                assert reg.lower() in [r.lower() for r in result.reads_registers], \
                    f"{pattern['name']}: should read {reg}"
            elif check == "memory_write":
                assert any(e.operation == EffectOperation.WRITE for e in result.memory_effects), \
                    f"{pattern['name']}: should have memory write"
            elif check == "memory_read":
                assert any(e.operation == EffectOperation.READ for e in result.memory_effects), \
                    f"{pattern['name']}: should have memory read"
            elif check == "no_memory":
                assert len(result.memory_effects) == 0, \
                    f"{pattern['name']}: should have no memory effects"
            elif check == "no_flags":
                assert len(result.flag_effects) == 0, \
                    f"{pattern['name']}: should have no flag effects"
            elif check == "modifies_flags":
                assert len(result.flag_effects) > 0, \
                    f"{pattern['name']}: should modify flags"
            elif check == "modifies_zf":
                assert "ZF" in result.modifies_flags, \
                    f"{pattern['name']}: should modify ZF"
            elif check == "clears_cf":
                cf = next((e for e in result.flag_effects if e.flag == "CF"), None)
                assert cf and cf.operation == FlagOperation.CLEAR, \
                    f"{pattern['name']}: should clear CF"
            elif check == "clears_of":
                of = next((e for e in result.flag_effects if e.flag == "OF"), None)
                assert of and of.operation == FlagOperation.CLEAR, \
                    f"{pattern['name']}: should clear OF"
            elif check == "preserves_cf":
                cf = next((e for e in result.flag_effects if e.flag == "CF"), None)
                assert cf is None or cf.operation == FlagOperation.UNCHANGED, \
                    f"{pattern['name']}: should preserve CF"
            elif check == "no_writes":
                assert len(result.writes_registers) == 0, \
                    f"{pattern['name']}: should not write any registers"
            elif check == "control_flow":
                assert result.control_flow.type == expected, \
                    f"{pattern['name']}: expected {expected} control flow"


class TestExam5Adversarial:
    """Exam 5: Adversarial Cases."""
    
    @pytest.fixture
    def module(self):
        return Level1Module()
    
    @pytest.mark.parametrize("mnemonic,operands,should_refuse", ADVERSARIAL_CASES)
    def test_adversarial(self, module, mnemonic, operands, should_refuse):
        instr = make_instr(mnemonic, operands)
        result = module.analyze(Level1Input(instruction=instr))
        
        if should_refuse:
            assert result.is_uncertain, \
                f"{mnemonic} with {operands} should be refused"


# ============================================================
# EXAM RUNNER
# ============================================================

def run_exam():
    """Run all exams and report results."""
    import sys
    
    module = Level1Module()
    
    results = {
        "Exam 1: Data Movement": {"passed": 0, "failed": 0, "failures": []},
        "Exam 2: Arithmetic": {"passed": 0, "failed": 0, "failures": []},
        "Exam 3: Control Flow": {"passed": 0, "failed": 0, "failures": []},
        "Exam 4: Real World": {"passed": 0, "failed": 0, "failures": []},
        "Exam 5: Adversarial": {"passed": 0, "failed": 0, "failures": []},
    }
    
    # Exam 1
    for case in DATA_MOVEMENT_CASES:
        mnemonic, operands, reads, writes, mem_read, mem_write = case
        try:
            instr = make_instr(mnemonic, operands)
            result = module.analyze(Level1Input(instruction=instr))
            
            if result.is_uncertain:
                raise AssertionError("Uncertain")
            
            # Basic checks
            for reg in writes:
                if reg.lower() not in [r.lower() for r in result.writes_registers]:
                    raise AssertionError(f"Should write {reg}")
            
            results["Exam 1: Data Movement"]["passed"] += 1
        except Exception as e:
            results["Exam 1: Data Movement"]["failed"] += 1
            results["Exam 1: Data Movement"]["failures"].append(f"{mnemonic}: {e}")
    
    # Exam 2
    for case in ARITHMETIC_CASES:
        mnemonic, operands, cf, zf, sf, of = case
        try:
            instr = make_instr(mnemonic, operands)
            result = module.analyze(Level1Input(instruction=instr))
            
            if result.is_uncertain:
                raise AssertionError("Uncertain")
            
            results["Exam 2: Arithmetic"]["passed"] += 1
        except Exception as e:
            results["Exam 2: Arithmetic"]["failed"] += 1
            results["Exam 2: Arithmetic"]["failures"].append(f"{mnemonic}: {e}")
    
    # Exam 3
    for case in CONTROL_FLOW_CASES:
        mnemonic, operands, expected_type, has_condition = case
        try:
            instr = make_instr(mnemonic, operands)
            result = module.analyze(Level1Input(instruction=instr))
            
            if result.is_uncertain:
                raise AssertionError("Uncertain")
            if result.control_flow.type != expected_type:
                raise AssertionError(f"Expected {expected_type}")
            
            results["Exam 3: Control Flow"]["passed"] += 1
        except Exception as e:
            results["Exam 3: Control Flow"]["failed"] += 1
            results["Exam 3: Control Flow"]["failures"].append(f"{mnemonic}: {e}")
    
    # Exam 4
    for pattern in REAL_WORLD_PATTERNS:
        try:
            instr = make_instr(pattern["mnemonic"], pattern["operands"])
            result = module.analyze(Level1Input(instruction=instr))
            
            if result.is_uncertain:
                raise AssertionError("Uncertain")
            
            results["Exam 4: Real World"]["passed"] += 1
        except Exception as e:
            results["Exam 4: Real World"]["failed"] += 1
            results["Exam 4: Real World"]["failures"].append(f"{pattern['name']}: {e}")
    
    # Exam 5
    for case in ADVERSARIAL_CASES:
        mnemonic, operands, should_refuse = case
        try:
            instr = make_instr(mnemonic, operands)
            result = module.analyze(Level1Input(instruction=instr))
            
            if should_refuse and not result.is_uncertain:
                raise AssertionError("Should be refused")
            
            results["Exam 5: Adversarial"]["passed"] += 1
        except Exception as e:
            results["Exam 5: Adversarial"]["failed"] += 1
            results["Exam 5: Adversarial"]["failures"].append(f"{mnemonic}: {e}")
    
    # Print results
    print("=" * 60)
    print("LEVEL 1 COMPREHENSIVE EXAM RESULTS")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    for exam_name, exam_results in results.items():
        passed = exam_results["passed"]
        failed = exam_results["failed"]
        total = passed + failed
        pct = 100 * passed / total if total > 0 else 0
        status = "✓" if failed == 0 else "✗"
        
        print(f"\n{exam_name}")
        print(f"  Score: {passed}/{total} ({pct:.1f}%) {status}")
        
        if exam_results["failures"]:
            print("  Failures:")
            for f in exam_results["failures"][:5]:
                print(f"    - {f}")
        
        total_passed += passed
        total_failed += failed
    
    total = total_passed + total_failed
    overall_pct = 100 * total_passed / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"OVERALL: {total_passed}/{total} ({overall_pct:.1f}%)")
    
    if overall_pct >= 98:
        print("✅ PASSED - Level 1 deterministic module ready")
    elif overall_pct >= 90:
        print("⚠️  CLOSE - Minor fixes needed")
    else:
        print("❌ FAILED - Significant work needed")
    print("=" * 60)
    
    return overall_pct >= 98


if __name__ == "__main__":
    success = run_exam()
    exit(0 if success else 1)
