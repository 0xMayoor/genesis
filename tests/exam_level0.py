"""
Level 0 Comprehensive Exam Suite

Real-world test cases from:
1. Common function prologues/epilogues
2. Known shellcode patterns
3. Intel manual examples
4. CTF-style challenges
"""

import pytest
from pathlib import Path
import json

# ============================================================================
# EXAM 1: Function Prologue/Epilogue Recognition
# These appear in virtually every compiled binary
# ============================================================================

FUNCTION_PATTERNS = [
    # Standard x86-64 prologue
    ("55", "push", "Function prologue: push rbp"),
    ("4889e5", "mov", "Function prologue: mov rbp, rsp"),
    ("4883ec", "sub", "Function prologue: sub rsp, imm (stack allocation)"),
    
    # Standard epilogue
    ("c9", "leave", "Function epilogue: leave"),
    ("c3", "ret", "Function epilogue: ret"),
    ("5d", "pop", "Function epilogue: pop rbp"),
    
    # Callee-saved registers (common in optimized code)
    ("53", "push", "Save rbx"),
    ("4154", "push", "Save r12"),
    ("4155", "push", "Save r13"),
    ("4156", "push", "Save r14"),
    ("4157", "push", "Save r15"),
    
    # Restore callee-saved
    ("5b", "pop", "Restore rbx"),
    ("415c", "pop", "Restore r12"),
    ("415d", "pop", "Restore r13"),
    ("415e", "pop", "Restore r14"),
    ("415f", "pop", "Restore r15"),
]

# ============================================================================
# EXAM 2: Shellcode Recognition
# Common patterns in exploits - security-critical to identify
# ============================================================================

SHELLCODE_PATTERNS = [
    # NOP sled variations
    ("90", "nop", "Classic NOP"),
    ("9090909090", "nop", "NOP sled (5 bytes)"),
    
    # System call setup (Linux x86-64)
    ("0f05", "syscall", "Linux syscall instruction"),
    ("cd80", "int", "Linux int 0x80 (32-bit syscall)"),
    
    # Common shellcode instructions
    ("31c0", "xor", "Zero eax (xor eax, eax)"),
    ("31db", "xor", "Zero ebx"),
    ("31c9", "xor", "Zero ecx"),
    ("31d2", "xor", "Zero edx"),
    ("31f6", "xor", "Zero esi"),
    ("31ff", "xor", "Zero edi"),
    ("4831c0", "xor", "Zero rax (64-bit)"),
    
    # execve setup patterns
    ("b03b", "mov", "mov al, 59 (execve syscall number)"),
    ("b801000000", "mov", "mov eax, 1 (exit syscall)"),
    
    # Stack manipulation for shellcode
    ("6a00", "push", "push 0 (null terminator)"),
    ("6a68", "push", "push 'h' (part of /bin/sh)"),
    
    # INT3 breakpoint (debugger detection)
    ("cc", "int3", "Breakpoint/debugger trap"),
]

# ============================================================================
# EXAM 3: Intel Manual Examples
# Canonical encodings from Intel SDM
# ============================================================================

INTEL_MANUAL_EXAMPLES = [
    # Basic data movement
    ("8b00", "mov", "mov eax, [rax]"),
    ("8b4004", "mov", "mov eax, [rax+4]"),
    ("890424", "mov", "mov [rsp], eax"),
    
    # Arithmetic
    ("01c0", "add", "add eax, eax"),
    ("29c0", "sub", "sub eax, eax"),
    ("f7d0", "not", "not eax"),
    ("f7d8", "neg", "neg eax"),
    
    # Logical
    ("21c0", "and", "and eax, eax"),
    ("09c0", "or", "or eax, eax"),
    ("31c0", "xor", "xor eax, eax"),
    
    # Shifts
    ("d1e0", "shl", "shl eax, 1"),
    ("d1e8", "shr", "shr eax, 1"),
    ("d1f8", "sar", "sar eax, 1"),
    
    # Control flow
    ("eb00", "jmp", "jmp short (offset 0)"),
    ("e900000000", "jmp", "jmp near (offset 0)"),
    ("7400", "je", "je/jz short"),
    ("7500", "jne", "jne/jnz short"),
    
    # Comparison
    ("39c0", "cmp", "cmp eax, eax"),
    ("85c0", "test", "test eax, eax"),
    
    # Stack
    ("50", "push", "push rax"),
    ("51", "push", "push rcx"),
    ("52", "push", "push rdx"),
    ("58", "pop", "pop rax"),
    ("59", "pop", "pop rcx"),
    ("5a", "pop", "pop rdx"),
    
    # String operations
    ("a4", "movsb", "Move string byte"),
    ("a5", "movsd", "Move string dword"),
    ("aa", "stosb", "Store string byte"),
    ("ab", "stosd", "Store string dword"),
    ("ac", "lodsb", "Load string byte"),
    ("ad", "lodsd", "Load string dword"),
    ("ae", "scasb", "Scan string byte"),
    ("af", "scasd", "Scan string dword"),
    
    # REP prefixes
    ("f3a4", "rep", "rep movsb"),
    ("f3ab", "rep", "rep stosd"),
    
    # Misc
    ("f4", "hlt", "Halt"),
    ("9c", "pushf", "Push flags"),  # Could be pushfq in 64-bit
    ("9d", "popf", "Pop flags"),    # Could be popfq in 64-bit
    ("fc", "cld", "Clear direction flag"),
    ("fd", "std", "Set direction flag"),
]

# ============================================================================
# EXAM 4: CTF-Style Challenges
# Tricky encodings that often appear in CTF reverse engineering
# ============================================================================

CTF_CHALLENGES = [
    # Self-modifying code patterns
    ("c6042490", "mov", "mov byte [rsp], 0x90 (write NOP)"),
    
    # Anti-disassembly tricks (valid but unusual)
    ("eb01", "jmp", "jmp +1 (skip next byte)"),
    
    # Obfuscated zero
    ("2bc0", "sub", "sub eax, eax (alternative zero)"),
    ("33c0", "xor", "xor eax, eax (alternative encoding)"),
    
    # LEA tricks (used instead of arithmetic)
    ("8d0400", "lea", "lea eax, [rax+rax] (multiply by 2)"),
    ("8d0480", "lea", "lea eax, [rax+rax*4] (multiply by 5)"),
    
    # XCHG with accumulator (compact encoding)
    ("91", "xchg", "xchg eax, ecx"),
    ("92", "xchg", "xchg eax, edx"),
    ("93", "xchg", "xchg eax, ebx"),
    
    # BSWAP (endianness swap)
    ("0fc8", "bswap", "bswap eax"),
    ("0fc9", "bswap", "bswap ecx"),
    
    # Bit manipulation
    ("0fbcc0", "bsf", "bsf eax, eax (bit scan forward)"),
    ("0fbdc0", "bsr", "bsr eax, eax (bit scan reverse)"),
    
    # CPUID (often used for VM detection)
    ("0fa2", "cpuid", "cpuid"),
    
    # RDTSC (timing attacks)
    ("0f31", "rdtsc", "rdtsc (read timestamp counter)"),
]

# ============================================================================
# EXAM 5: Adversarial Cases (Must Refuse)
# Invalid or ambiguous inputs the model should NOT try to decode
# ============================================================================

ADVERSARIAL_CASES = [
    # Invalid hex
    ("gg", "Invalid hex characters"),
    ("xyz", "Non-hex string"),
    ("0x90", "Has 0x prefix (should be raw hex)"),
    
    # Truncated instructions
    ("0f", "Incomplete two-byte opcode"),
    ("f3", "Lone REP prefix"),
    ("66", "Lone operand size prefix"),
    ("67", "Lone address size prefix"),
    ("48", "Lone REX prefix"),
    
    # Empty/whitespace
    ("", "Empty input"),
    ("   ", "Whitespace only"),
    
    # Too long (likely garbage)
    ("90" * 100, "Suspiciously long input"),
]


# ============================================================================
# Test Functions
# ============================================================================

class TestFunctionPatterns:
    """Exam 1: Function prologue/epilogue recognition"""
    
    @pytest.mark.parametrize("hex_bytes,expected_mnemonic,description", FUNCTION_PATTERNS)
    def test_function_pattern(self, hex_bytes, expected_mnemonic, description):
        """Test recognition of common function patterns"""
        # This will be filled in when we have the model inference function
        pytest.skip("Model inference not yet integrated")


class TestShellcodePatterns:
    """Exam 2: Shellcode pattern recognition"""
    
    @pytest.mark.parametrize("hex_bytes,expected_mnemonic,description", SHELLCODE_PATTERNS)
    def test_shellcode_pattern(self, hex_bytes, expected_mnemonic, description):
        """Test recognition of shellcode patterns"""
        pytest.skip("Model inference not yet integrated")


class TestIntelManualExamples:
    """Exam 3: Intel manual canonical examples"""
    
    @pytest.mark.parametrize("hex_bytes,expected_mnemonic,description", INTEL_MANUAL_EXAMPLES)
    def test_intel_example(self, hex_bytes, expected_mnemonic, description):
        """Test Intel manual examples"""
        pytest.skip("Model inference not yet integrated")


class TestCTFChallenges:
    """Exam 4: CTF-style tricky encodings"""
    
    @pytest.mark.parametrize("hex_bytes,expected_mnemonic,description", CTF_CHALLENGES)
    def test_ctf_challenge(self, hex_bytes, expected_mnemonic, description):
        """Test CTF-style challenges"""
        pytest.skip("Model inference not yet integrated")


class TestAdversarialCases:
    """Exam 5: Adversarial cases (must refuse)"""
    
    @pytest.mark.parametrize("hex_bytes,description", ADVERSARIAL_CASES)
    def test_adversarial_refusal(self, hex_bytes, description):
        """Test that model refuses invalid inputs"""
        pytest.skip("Model inference not yet integrated")


# ============================================================================
# Standalone Exam Runner (for use with trained model)
# ============================================================================

def run_exam_with_model(inference_fn):
    """
    Run full exam suite with a model inference function.
    
    Args:
        inference_fn: Function that takes hex_bytes and returns predicted mnemonic
                     Should return "INVALID" or similar for adversarial cases
    
    Returns:
        dict with exam results
    """
    results = {
        "function_patterns": {"correct": 0, "total": len(FUNCTION_PATTERNS), "failures": []},
        "shellcode_patterns": {"correct": 0, "total": len(SHELLCODE_PATTERNS), "failures": []},
        "intel_examples": {"correct": 0, "total": len(INTEL_MANUAL_EXAMPLES), "failures": []},
        "ctf_challenges": {"correct": 0, "total": len(CTF_CHALLENGES), "failures": []},
        "adversarial_refusal": {"correct": 0, "total": len(ADVERSARIAL_CASES), "failures": []},
    }
    
    # Exam 1: Function patterns
    for hex_bytes, expected, desc in FUNCTION_PATTERNS:
        pred = inference_fn(hex_bytes)
        if pred.lower() == expected.lower():
            results["function_patterns"]["correct"] += 1
        else:
            results["function_patterns"]["failures"].append((hex_bytes, expected, pred, desc))
    
    # Exam 2: Shellcode patterns
    for hex_bytes, expected, desc in SHELLCODE_PATTERNS:
        pred = inference_fn(hex_bytes)
        if pred.lower() == expected.lower():
            results["shellcode_patterns"]["correct"] += 1
        else:
            results["shellcode_patterns"]["failures"].append((hex_bytes, expected, pred, desc))
    
    # Exam 3: Intel manual examples
    for hex_bytes, expected, desc in INTEL_MANUAL_EXAMPLES:
        pred = inference_fn(hex_bytes)
        if pred.lower() == expected.lower():
            results["intel_examples"]["correct"] += 1
        else:
            results["intel_examples"]["failures"].append((hex_bytes, expected, pred, desc))
    
    # Exam 4: CTF challenges
    for hex_bytes, expected, desc in CTF_CHALLENGES:
        pred = inference_fn(hex_bytes)
        if pred.lower() == expected.lower():
            results["ctf_challenges"]["correct"] += 1
        else:
            results["ctf_challenges"]["failures"].append((hex_bytes, expected, pred, desc))
    
    # Exam 5: Adversarial (should refuse)
    refusal_indicators = ["invalid", "unknown", "error", "cannot", "refuse", "uncertain"]
    for hex_bytes, desc in ADVERSARIAL_CASES:
        pred = inference_fn(hex_bytes).lower()
        if any(r in pred for r in refusal_indicators):
            results["adversarial_refusal"]["correct"] += 1
        else:
            results["adversarial_refusal"]["failures"].append((hex_bytes, "REFUSE", pred, desc))
    
    return results


def print_exam_results(results):
    """Pretty print exam results"""
    print("\n" + "=" * 60)
    print("LEVEL 0 COMPREHENSIVE EXAM RESULTS")
    print("=" * 60)
    
    total_correct = 0
    total_questions = 0
    
    exam_names = {
        "function_patterns": "Exam 1: Function Patterns",
        "shellcode_patterns": "Exam 2: Shellcode Patterns", 
        "intel_examples": "Exam 3: Intel Manual Examples",
        "ctf_challenges": "Exam 4: CTF Challenges",
        "adversarial_refusal": "Exam 5: Adversarial Refusal",
    }
    
    for key, name in exam_names.items():
        r = results[key]
        pct = 100 * r["correct"] / r["total"] if r["total"] > 0 else 0
        status = "✓" if pct >= 98 else "✗"
        print(f"\n{name}")
        print(f"  Score: {r['correct']}/{r['total']} ({pct:.1f}%) {status}")
        
        if r["failures"] and len(r["failures"]) <= 5:
            print("  Failures:")
            for f in r["failures"][:5]:
                print(f"    - {f[0]}: expected '{f[1]}', got '{f[2]}' ({f[3]})")
        elif r["failures"]:
            print(f"  Failures: {len(r['failures'])} (showing first 5)")
            for f in r["failures"][:5]:
                print(f"    - {f[0]}: expected '{f[1]}', got '{f[2]}' ({f[3]})")
        
        total_correct += r["correct"]
        total_questions += r["total"]
    
    overall_pct = 100 * total_correct / total_questions if total_questions > 0 else 0
    print("\n" + "=" * 60)
    print(f"OVERALL: {total_correct}/{total_questions} ({overall_pct:.1f}%)")
    
    if overall_pct >= 98:
        print("🎉 PASSED - Ready for Level 1!")
    elif overall_pct >= 95:
        print("⚠️  CLOSE - Minor improvements needed")
    else:
        print("❌ FAILED - More training required")
    
    print("=" * 60)
    
    return overall_pct


if __name__ == "__main__":
    # Example usage with a dummy inference function
    def dummy_inference(hex_bytes):
        """Placeholder - replace with actual model inference"""
        return "unknown"
    
    print("Run with: pytest tests/exam_level0.py -v")
    print("Or import and use run_exam_with_model() with your inference function")
