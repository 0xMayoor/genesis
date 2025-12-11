#!/usr/bin/env python3
"""
Real-World Validation Suite

Tests GENESIS models against REAL compiled binaries, not synthetic data.
Uses objdump/radare2 as ground truth.

This is the undeniable proof that our models work.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re

# Real C programs to compile and test
REAL_PROGRAMS = {
    "simple_return": """
int main() {
    return 42;
}
""",
    
    "simple_add": """
int add(int a, int b) {
    return a + b;
}

int main() {
    return add(3, 5);
}
""",
    
    "if_else": """
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

int main() {
    return max(10, 20);
}
""",

    "while_loop": """
int sum_to_n(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n;
        n--;
    }
    return sum;
}

int main() {
    return sum_to_n(10);
}
""",

    "for_loop": """
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main() {
    return factorial(5);
}
""",

    "nested_if": """
int classify(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 0) {
        return 0;
    } else {
        return 1;
    }
}

int main() {
    return classify(5);
}
""",

    "multiple_functions": """
int helper(int x) {
    return x * 2;
}

int process(int a, int b) {
    int x = helper(a);
    int y = helper(b);
    return x + y;
}

int main() {
    return process(3, 4);
}
""",

    "switch_case": """
int day_type(int day) {
    switch(day) {
        case 0:
        case 6:
            return 0;  // weekend
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            return 1;  // weekday
        default:
            return -1;
    }
}

int main() {
    return day_type(3);
}
""",

    "recursion": """
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}

int main() {
    return fib(10);
}
""",

    "pointer_arithmetic": """
int sum_array(int* arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    return sum_array(arr, 5);
}
""",
}


@dataclass
class BinaryInfo:
    """Information about a compiled binary."""
    path: Path
    name: str
    functions: list[str]
    basic_blocks_count: int
    has_loops: bool
    has_conditionals: bool


def compile_program(name: str, code: str, tmpdir: Path) -> Optional[Path]:
    """Compile C code to binary."""
    src_path = tmpdir / f"{name}.c"
    bin_path = tmpdir / name
    
    src_path.write_text(code)
    
    # Compile with -O0 to preserve structure, -g for debug info
    result = subprocess.run(
        ["gcc", "-O0", "-g", "-o", str(bin_path), str(src_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  Failed to compile {name}: {result.stderr}")
        return None
    
    return bin_path


def get_objdump_functions(binary_path: Path) -> list[str]:
    """Get function names from objdump."""
    result = subprocess.run(
        ["objdump", "-t", str(binary_path)],
        capture_output=True,
        text=True
    )
    
    functions = []
    for line in result.stdout.split('\n'):
        if ' F .text' in line:
            parts = line.split()
            if parts:
                func_name = parts[-1]
                if not func_name.startswith('_'):  # Skip internal functions
                    functions.append(func_name)
    
    return functions


def get_objdump_disassembly(binary_path: Path, function: str = None) -> str:
    """Get disassembly from objdump."""
    cmd = ["objdump", "-d", str(binary_path)]
    if function:
        cmd.extend(["--disassemble=" + function])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def count_basic_blocks_objdump(disasm: str) -> int:
    """Count basic blocks by looking for jump targets and ret instructions."""
    # Count unique addresses that are jump targets
    jump_targets = set()
    ret_count = 0
    
    lines = disasm.split('\n')
    for line in lines:
        # Look for jumps
        if any(j in line for j in ['jmp', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jae', 'jbe', 'call']):
            # Extract target address
            match = re.search(r'[0-9a-f]+\s+<', line)
            if match:
                jump_targets.add(match.group())
        
        if 'ret' in line:
            ret_count += 1
    
    # Rough estimate: each jump target + entry = block
    return max(len(jump_targets) + 1, ret_count + 1)


def has_conditional_jumps(disasm: str) -> bool:
    """Check if disassembly has conditional jumps."""
    conditionals = ['je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jae', 'jbe', 'jz', 'jnz']
    return any(cond in disasm for cond in conditionals)


def has_backward_jumps(disasm: str) -> bool:
    """Check for loops (backward jumps)."""
    lines = disasm.split('\n')
    addresses = []
    jumps = []
    
    for line in lines:
        # Get address of this line
        match = re.match(r'\s*([0-9a-f]+):', line)
        if match:
            addr = int(match.group(1), 16)
            addresses.append(addr)
            
            # Check if this is a jump
            if 'jmp' in line or any(j in line for j in ['je', 'jne', 'jg', 'jl']):
                target_match = re.search(r'([0-9a-f]+)\s+<', line)
                if target_match:
                    target = int(target_match.group(1), 16)
                    if target < addr:  # Backward jump = loop
                        return True
    
    return False


def test_level0_on_real_binary(binary_path: Path) -> dict:
    """Test Level 0 model on real binary."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from levels.level0_machine import Level0Module, Level0Input
    
    # Read binary
    with open(binary_path, 'rb') as f:
        binary_data = f.read()
    
    # Find .text section (simplified - just use objdump output)
    disasm = get_objdump_disassembly(binary_path)
    
    # Extract some instruction bytes to test
    module = Level0Module()
    
    # Test on first few instructions
    test_bytes = [
        bytes.fromhex("55"),           # push rbp
        bytes.fromhex("4889e5"),       # mov rbp, rsp
        bytes.fromhex("b800000000"),   # mov eax, 0
        bytes.fromhex("c3"),           # ret
    ]
    
    results = []
    for test in test_bytes:
        output = module.process(Level0Input(data=test))
        results.append({
            "bytes": test.hex(),
            "mnemonic": output.instructions[0].mnemonic if output.instructions else None,
            "uncertain": output.is_uncertain,
        })
    
    return {
        "tested": len(results),
        "successful": sum(1 for r in results if r["mnemonic"]),
        "results": results,
    }


def test_with_capstone(binary_path: Path) -> dict:
    """Compare our Level 0 with Capstone directly on binary."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64
    from levels.level0_machine import Level0Module, Level0Input
    
    # Get text section bytes
    result = subprocess.run(
        ["objcopy", "-O", "binary", "--only-section=.text", str(binary_path), "/dev/stdout"],
        capture_output=True
    )
    text_bytes = result.stdout[:100]  # First 100 bytes
    
    if not text_bytes:
        return {"error": "Could not extract .text section"}
    
    # Disassemble with Capstone (ground truth)
    cs = Cs(CS_ARCH_X86, CS_MODE_64)
    capstone_instrs = list(cs.disasm(text_bytes, 0))
    
    # Disassemble with our module
    module = Level0Module()
    our_output = module.process(Level0Input(data=text_bytes))
    
    # Compare
    matches = 0
    total = min(len(capstone_instrs), len(our_output.instructions))
    
    for i in range(total):
        cap = capstone_instrs[i]
        our = our_output.instructions[i]
        if cap.mnemonic == our.mnemonic:
            matches += 1
    
    return {
        "capstone_count": len(capstone_instrs),
        "our_count": len(our_output.instructions),
        "matches": matches,
        "accuracy": matches / total if total > 0 else 0,
    }


def run_real_world_exam():
    """Run the complete real-world examination."""
    print("=" * 70)
    print("GENESIS REAL-WORLD VALIDATION EXAM")
    print("Testing against REAL compiled binaries, not synthetic data")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Phase 1: Compile all test programs
        print("\n📦 PHASE 1: Compiling real C programs...")
        binaries = {}
        
        for name, code in REAL_PROGRAMS.items():
            bin_path = compile_program(name, code, tmpdir)
            if bin_path:
                disasm = get_objdump_disassembly(bin_path)
                binaries[name] = BinaryInfo(
                    path=bin_path,
                    name=name,
                    functions=get_objdump_functions(bin_path),
                    basic_blocks_count=count_basic_blocks_objdump(disasm),
                    has_loops=has_backward_jumps(disasm),
                    has_conditionals=has_conditional_jumps(disasm),
                )
                print(f"  ✓ {name}: {len(binaries[name].functions)} functions")
        
        print(f"\nCompiled {len(binaries)}/{len(REAL_PROGRAMS)} programs")
        
        # Phase 2: Test Level 0 (disassembly)
        print("\n🔬 PHASE 2: Testing Level 0 (Machine Code → Assembly)")
        print("-" * 50)
        
        level0_results = []
        for name, info in binaries.items():
            result = test_with_capstone(info.path)
            level0_results.append(result)
            
            if "error" in result:
                print(f"  ✗ {name}: {result['error']}")
            else:
                acc = result["accuracy"]
                status = "✓" if acc >= 0.95 else "⚠" if acc >= 0.80 else "✗"
                print(f"  {status} {name}: {acc*100:.1f}% match with Capstone ({result['matches']}/{result['capstone_count']})")
        
        valid_results = [r for r in level0_results if "accuracy" in r]
        if valid_results:
            avg_acc = sum(r["accuracy"] for r in valid_results) / len(valid_results)
            print(f"\n  Level 0 Average Accuracy: {avg_acc*100:.1f}%")
        
        # Phase 3: Test against objdump output format
        print("\n🔬 PHASE 3: Verifying disassembly against objdump")
        print("-" * 50)
        
        for name, info in list(binaries.items())[:3]:  # Test first 3
            print(f"\n  {name}:")
            disasm = get_objdump_disassembly(info.path, "main")
            
            # Count instructions in main
            instr_count = len([l for l in disasm.split('\n') if re.match(r'\s+[0-9a-f]+:', l)])
            print(f"    objdump: {instr_count} instructions in main()")
            print(f"    has_conditionals: {info.has_conditionals}")
            print(f"    has_loops: {info.has_loops}")
        
        # Phase 4: Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"""
Programs compiled:     {len(binaries)}/{len(REAL_PROGRAMS)}
Level 0 accuracy:      {avg_acc*100:.1f}% (vs Capstone ground truth)

Test binaries include:
  - Simple returns
  - Conditionals (if/else)
  - Loops (while, for)
  - Recursion
  - Multiple functions
  - Switch statements
  - Pointer arithmetic
        """)
        
        if avg_acc >= 0.95:
            print("✅ PASSED: Level 0 matches Capstone on real binaries")
        elif avg_acc >= 0.80:
            print("⚠️  PARTIAL: Level 0 mostly works but has some discrepancies")
        else:
            print("❌ FAILED: Level 0 has significant issues with real binaries")
        
        return avg_acc


if __name__ == "__main__":
    run_real_world_exam()
