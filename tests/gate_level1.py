#!/usr/bin/env python3
"""
Level 1 Gate Test: Instruction Semantics

FROZEN TEST - DO NOT MODIFY AFTER TRAINING BEGINS

This test verifies that the Level 1 model correctly predicts:
1. Registers read by an instruction
2. Registers written by an instruction
3. Whether memory is read
4. Whether memory is written
5. Whether flags are affected

Ground truth: Capstone detail mode (authoritative x86 decoder)
Test programs: Completely separate from training data
"""

import subprocess
import tempfile
from pathlib import Path

from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_MEM, X86_OP_REG

# =============================================================================
# GATE TEST PROGRAMS - NEVER USE THESE IN TRAINING
# =============================================================================

GATE_PROGRAMS = {
    "fibonacci": """
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}
int main() { return fib(10); }
""",
    "binary_search": """
int bsearch(int* arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
int main() { int a[] = {1,2,3,4,5}; return bsearch(a, 5, 3); }
""",
    "quicksort_partition": """
void swap(int* a, int* b) { int t = *a; *a = *b; *b = t; }
int partition(int* arr, int lo, int hi) {
    int pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) { i++; swap(&arr[i], &arr[j]); }
    }
    swap(&arr[i+1], &arr[hi]);
    return i + 1;
}
int main() { int a[] = {3,1,4,1,5}; return partition(a, 0, 4); }
""",
    "matrix_multiply": """
void matmul(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i*n+j] = 0;
            for (int k = 0; k < n; k++)
                C[i*n+j] += A[i*n+k] * B[k*n+j];
        }
}
int main() { int A[4]={1,2,3,4}, B[4]={1,0,0,1}, C[4]; matmul(A,B,C,2); return C[0]; }
""",
    "linked_list": """
struct Node { int val; struct Node* next; };
int sum_list(struct Node* head) {
    int s = 0;
    while (head) { s += head->val; head = head->next; }
    return s;
}
int main() { return 0; }
""",
}

# Compilers and optimization levels for test
TEST_COMPILERS = ["gcc", "clang"]
TEST_OPT_LEVELS = ["-O0", "-O2"]  # Only 2 levels for gate test


# =============================================================================
# GROUND TRUTH EXTRACTION (Capstone)
# =============================================================================

def get_instruction_semantics(insn) -> dict:
    """Extract ground truth semantics from Capstone instruction.
    
    Returns dict with:
        reads: set of register names read
        writes: set of register names written  
        mem_read: bool
        mem_write: bool
        flags_written: bool
    """
    reads = set(insn.reg_name(r) for r in insn.regs_read)
    writes = set(insn.reg_name(r) for r in insn.regs_write)
    mem_read = False
    mem_write = False
    
    # Parse operands for explicit registers and memory
    for i, op in enumerate(insn.operands):
        if op.type == X86_OP_REG:
            reg = insn.reg_name(op.reg)
            # Destination operand (first) is write, except for cmp/test/push/jmp/call
            if i == 0 and insn.mnemonic not in ['push', 'cmp', 'test', 'jmp', 'call', 'ja', 'jae', 'jb', 'jbe', 'je', 'jne', 'jg', 'jge', 'jl', 'jle', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp', 'jz', 'jnz']:
                writes.add(reg)
            else:
                reads.add(reg)
        elif op.type == X86_OP_MEM:
            # Memory base/index registers are read
            if op.mem.base:
                reads.add(insn.reg_name(op.mem.base))
            if op.mem.index:
                reads.add(insn.reg_name(op.mem.index))
            # First operand memory = write (except lea, cmp, test)
            if i == 0 and insn.mnemonic not in ['lea', 'cmp', 'test']:
                mem_write = True
            else:
                mem_read = True
    
    # Special cases
    if insn.mnemonic == 'push':
        mem_write = True
    elif insn.mnemonic == 'pop':
        mem_read = True
    elif insn.mnemonic == 'ret':
        mem_read = True  # reads return address from stack
    elif insn.mnemonic == 'call':
        mem_write = True  # writes return address to stack
    
    # Flags
    flags_written = insn.eflags != 0
    
    # Normalize: remove 'rflags' from writes if flags_written is True
    writes.discard('rflags')
    
    return {
        'reads': frozenset(reads),
        'writes': frozenset(writes),
        'mem_read': mem_read,
        'mem_write': mem_write,
        'flags_written': flags_written,
    }


def compile_and_extract(code: str, compiler: str, opt: str) -> list[tuple[bytes, str, dict]]:
    """Compile C code and extract instructions with ground truth semantics.
    
    Returns list of (bytes, mnemonic, semantics_dict)
    """
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src = tmpdir / "test.c"
        binary = tmpdir / "test"
        
        src.write_text(code)
        
        # Compile
        result = subprocess.run(
            [compiler, opt, "-w", "-o", str(binary), str(src)],
            capture_output=True,
        )
        if result.returncode != 0:
            return []
        
        # Extract .text section
        text_result = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", str(binary), "/dev/stdout"],
            capture_output=True,
        )
        if not text_result.stdout:
            return []
        
        # Disassemble with Capstone detail mode
        md = Cs(CS_ARCH_X86, CS_MODE_64)
        md.detail = True
        
        for insn in md.disasm(text_result.stdout, 0):
            semantics = get_instruction_semantics(insn)
            results.append((bytes(insn.bytes), insn.mnemonic, semantics))
    
    return results


# =============================================================================
# GATE TEST
# =============================================================================

class Level1GateTest:
    """Gate test for Level 1 model.
    
    Must achieve 100% accuracy on ALL metrics to pass.
    """
    
    def __init__(self, model):
        """Initialize with a Level 1 model.
        
        Model must have a predict(bytes) -> dict method that returns:
            {
                'reads': set of register names,
                'writes': set of register names,
                'mem_read': bool,
                'mem_write': bool,
                'flags_written': bool,
            }
        """
        self.model = model
        self.test_data = []
        self._load_test_data()
    
    def _load_test_data(self):
        """Load test data from gate programs."""
        print("Loading gate test data...")
        for name, code in GATE_PROGRAMS.items():
            for compiler in TEST_COMPILERS:
                for opt in TEST_OPT_LEVELS:
                    samples = compile_and_extract(code, compiler, opt)
                    self.test_data.extend(samples)
                    print(f"  {name}/{compiler}/{opt}: {len(samples)} instructions")
        print(f"Total gate test samples: {len(self.test_data)}")
    
    def run(self) -> dict:
        """Run the gate test.
        
        Returns dict with:
            passed: bool
            accuracy: dict of per-field accuracies
            errors: list of error details
        """
        results = {
            'reads_correct': 0,
            'writes_correct': 0,
            'mem_read_correct': 0,
            'mem_write_correct': 0,
            'flags_correct': 0,
            'total': 0,
            'errors': [],
        }
        
        for instr_bytes, mnemonic, ground_truth in self.test_data:
            pred = self.model.predict(instr_bytes)
            results['total'] += 1
            
            # Check each field
            if set(pred.get('reads', set())) == ground_truth['reads']:
                results['reads_correct'] += 1
            else:
                results['errors'].append({
                    'bytes': instr_bytes.hex(),
                    'mnemonic': mnemonic,
                    'field': 'reads',
                    'predicted': pred.get('reads'),
                    'expected': ground_truth['reads'],
                })
            
            if set(pred.get('writes', set())) == ground_truth['writes']:
                results['writes_correct'] += 1
            else:
                results['errors'].append({
                    'bytes': instr_bytes.hex(),
                    'mnemonic': mnemonic,
                    'field': 'writes',
                    'predicted': pred.get('writes'),
                    'expected': ground_truth['writes'],
                })
            
            if pred.get('mem_read') == ground_truth['mem_read']:
                results['mem_read_correct'] += 1
            
            if pred.get('mem_write') == ground_truth['mem_write']:
                results['mem_write_correct'] += 1
            
            if pred.get('flags_written') == ground_truth['flags_written']:
                results['flags_correct'] += 1
        
        # Calculate accuracies
        total = results['total']
        accuracy = {
            'reads': results['reads_correct'] / total if total > 0 else 0,
            'writes': results['writes_correct'] / total if total > 0 else 0,
            'mem_read': results['mem_read_correct'] / total if total > 0 else 0,
            'mem_write': results['mem_write_correct'] / total if total > 0 else 0,
            'flags': results['flags_correct'] / total if total > 0 else 0,
        }
        
        # Overall pass requires 100% on ALL fields
        passed = all(acc == 1.0 for acc in accuracy.values())
        
        return {
            'passed': passed,
            'accuracy': accuracy,
            'total': total,
            'errors': results['errors'][:20],  # Limit errors shown
        }


def print_gate_test_results(results: dict):
    """Pretty print gate test results."""
    print("\n" + "=" * 60)
    print("LEVEL 1 GATE TEST RESULTS")
    print("=" * 60)
    
    print(f"\nTotal instructions tested: {results['total']}")
    print("\nPer-field accuracy:")
    for field, acc in results['accuracy'].items():
        status = "✓" if acc == 1.0 else "✗"
        print(f"  {field:12}: {acc*100:6.2f}% {status}")
    
    print(f"\nOverall: {'PASSED ✓' if results['passed'] else 'FAILED ✗'}")
    
    if results['errors']:
        print(f"\nSample errors ({len(results['errors'])} shown):")
        for err in results['errors'][:10]:
            print(f"  {err['bytes']}: {err['mnemonic']} - {err['field']}")
            print(f"    predicted: {err['predicted']}")
            print(f"    expected:  {err['expected']}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test ground truth extraction
    print("Testing ground truth extraction...")
    
    test_code = "int main() { int x = 0; for(int i=0; i<10; i++) x += i; return x; }"
    samples = compile_and_extract(test_code, "gcc", "-O0")
    
    print(f"\nExtracted {len(samples)} instructions from test program")
    print("\nSample semantics:")
    for bytez, mnem, sem in samples[:10]:
        print(f"  {bytez.hex():20} {mnem:10} R={list(sem['reads'])[:3]} W={list(sem['writes'])[:3]} MR={sem['mem_read']} MW={sem['mem_write']} F={sem['flags_written']}")
    
    print("\n[Gate test ready - waiting for Level 1 model]")
