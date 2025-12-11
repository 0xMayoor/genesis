#!/usr/bin/env python3
"""
Real Binary Dataset Generator

Generates training data from REAL compiled binaries, not synthetic patterns.
- Multiple compilers (gcc, clang)
- Multiple optimization levels (-O0, -O1, -O2, -O3)
- AT&T and Intel syntax
- Real instruction sequences from actual programs
"""

import subprocess
import tempfile
import json
import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import hashlib

# ============================================================================
# Real C Programs - Diverse patterns
# ============================================================================

REAL_C_PROGRAMS = {
    # Basic patterns
    "empty_main": "int main() { return 0; }",
    "return_value": "int main() { return 42; }",
    "simple_var": "int main() { int x = 5; return x; }",
    
    # Arithmetic
    "add_two": "int add(int a, int b) { return a + b; } int main() { return add(3, 5); }",
    "multiply": "int mul(int a, int b) { return a * b; } int main() { return mul(6, 7); }",
    "complex_math": "int calc(int a, int b, int c) { return (a + b) * c - a / 2; } int main() { return calc(10, 20, 3); }",
    
    # Conditionals
    "simple_if": "int main() { int x = 5; if (x > 0) return 1; return 0; }",
    "if_else": "int max(int a, int b) { if (a > b) return a; else return b; } int main() { return max(10, 20); }",
    "nested_if": """
int classify(int x) {
    if (x < 0) return -1;
    else if (x == 0) return 0;
    else if (x < 10) return 1;
    else return 2;
}
int main() { return classify(5); }
""",
    "ternary": "int abs_val(int x) { return x >= 0 ? x : -x; } int main() { return abs_val(-5); }",
    
    # Loops
    "while_simple": """
int sum_to(int n) {
    int sum = 0;
    while (n > 0) { sum += n; n--; }
    return sum;
}
int main() { return sum_to(10); }
""",
    "for_loop": """
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) result *= i;
    return result;
}
int main() { return factorial(5); }
""",
    "do_while": """
int count_digits(int n) {
    int count = 0;
    do { count++; n /= 10; } while (n > 0);
    return count;
}
int main() { return count_digits(12345); }
""",
    "nested_loops": """
int matrix_sum(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            sum += i * j;
    return sum;
}
int main() { return matrix_sum(5); }
""",
    
    # Function calls
    "multi_func": """
int helper(int x) { return x * 2; }
int process(int a, int b) { return helper(a) + helper(b); }
int main() { return process(3, 4); }
""",
    "recursive": """
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
}
int main() { return fib(10); }
""",
    
    # Switch
    "switch_simple": """
int day_type(int d) {
    switch(d) {
        case 0: case 6: return 0;
        case 1: case 2: case 3: case 4: case 5: return 1;
        default: return -1;
    }
}
int main() { return day_type(3); }
""",
    
    # Pointers and arrays
    "array_sum": """
int sum_arr(int* arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) sum += arr[i];
    return sum;
}
int main() { int arr[] = {1,2,3,4,5}; return sum_arr(arr, 5); }
""",
    "pointer_swap": """
void swap(int* a, int* b) { int t = *a; *a = *b; *b = t; }
int main() { int x = 5, y = 10; swap(&x, &y); return x; }
""",
    
    # Structs
    "struct_basic": """
struct Point { int x; int y; };
int distance(struct Point p) { return p.x * p.x + p.y * p.y; }
int main() { struct Point p = {3, 4}; return distance(p); }
""",
    
    # Bitwise
    "bitwise_ops": """
int bits(int x) {
    int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}
int main() { return bits(255); }
""",
    
    # String operations
    "strlen_manual": """
int my_strlen(const char* s) {
    int len = 0;
    while (s[len]) len++;
    return len;
}
int main() { return my_strlen("hello"); }
""",
}

# Additional algorithmic programs for diversity
# More programs for diversity
MORE_PROGRAMS = {
    "abs_value": "int abs_v(int x) { return x < 0 ? -x : x; } int main() { return abs_v(-5); }",
    "min_val": "int min_v(int a, int b) { return a < b ? a : b; } int main() { return min_v(3, 7); }",
    "max3": "int max3(int a, int b, int c) { int m = a; if (b > m) m = b; if (c > m) m = c; return m; } int main() { return max3(1,5,3); }",
    "clamp": "int clamp(int x, int lo, int hi) { if (x < lo) return lo; if (x > hi) return hi; return x; } int main() { return clamp(15, 0, 10); }",
    "sign": "int sign(int x) { if (x > 0) return 1; if (x < 0) return -1; return 0; } int main() { return sign(-5); }",
    "div_ceil": "int div_ceil(int a, int b) { return (a + b - 1) / b; } int main() { return div_ceil(10, 3); }",
    "mod_pow2": "int mod_pow2(int x, int p) { return x & ((1 << p) - 1); } int main() { return mod_pow2(100, 4); }",
    "is_pow2": "int is_pow2(int x) { return x > 0 && (x & (x-1)) == 0; } int main() { return is_pow2(16); }",
    "next_pow2": """
unsigned next_pow2(unsigned x) {
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}
int main() { return next_pow2(17); }
""",
    "count_zeros": "int count_zeros(int* a, int n) { int c = 0; for (int i = 0; i < n; i++) if (a[i] == 0) c++; return c; } int main() { int a[] = {1,0,2,0,3}; return count_zeros(a, 5); }",
    "sum_squares": "int sum_sq(int n) { int s = 0; for (int i = 1; i <= n; i++) s += i*i; return s; } int main() { return sum_sq(5); }",
    "sum_cubes": "int sum_cu(int n) { int s = 0; for (int i = 1; i <= n; i++) s += i*i*i; return s; } int main() { return sum_cu(4); }",
    "avg_array": "int avg(int* a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s / n; } int main() { int a[] = {10,20,30}; return avg(a, 3); }",
    "find_first": "int find(int* a, int n, int x) { for (int i = 0; i < n; i++) if (a[i] == x) return i; return -1; } int main() { int a[] = {1,2,3,4,5}; return find(a, 5, 3); }",
    "all_positive": "int all_pos(int* a, int n) { for (int i = 0; i < n; i++) if (a[i] <= 0) return 0; return 1; } int main() { int a[] = {1,2,3}; return all_pos(a, 3); }",
    "any_negative": "int any_neg(int* a, int n) { for (int i = 0; i < n; i++) if (a[i] < 0) return 1; return 0; } int main() { int a[] = {1,-2,3}; return any_neg(a, 3); }",
    "rotate_left": "void rot_l(int* a, int n) { int t = a[0]; for (int i = 0; i < n-1; i++) a[i] = a[i+1]; a[n-1] = t; } int main() { int a[] = {1,2,3}; rot_l(a, 3); return a[0]; }",
    "rotate_right": "void rot_r(int* a, int n) { int t = a[n-1]; for (int i = n-1; i > 0; i--) a[i] = a[i-1]; a[0] = t; } int main() { int a[] = {1,2,3}; rot_r(a, 3); return a[0]; }",
    "copy_array": "void copy(int* d, int* s, int n) { for (int i = 0; i < n; i++) d[i] = s[i]; } int main() { int a[] = {1,2,3}, b[3]; copy(b, a, 3); return b[1]; }",
    "fill_array": "void fill(int* a, int n, int v) { for (int i = 0; i < n; i++) a[i] = v; } int main() { int a[5]; fill(a, 5, 42); return a[2]; }",
    "dot_product": "int dot(int* a, int* b, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; } int main() { int a[] = {1,2,3}, b[] = {4,5,6}; return dot(a, b, 3); }",
    "matrix_trace": "int trace(int m[3][3]) { return m[0][0] + m[1][1] + m[2][2]; } int main() { int m[3][3] = {{1,0,0},{0,2,0},{0,0,3}}; return trace(m); }",
    "str_equal": "int str_eq(char* a, char* b) { while (*a && *b) { if (*a++ != *b++) return 0; } return *a == *b; } int main() { return str_eq(\"hi\", \"hi\"); }",
    "str_copy": "void str_cp(char* d, char* s) { while ((*d++ = *s++)); } int main() { char d[10]; str_cp(d, \"test\"); return d[0]; }",
    "char_count": "int chr_cnt(char* s, char c) { int n = 0; while (*s) if (*s++ == c) n++; return n; } int main() { return chr_cnt(\"hello\", 'l'); }",
    "to_upper": "char to_up(char c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; } int main() { return to_up('a'); }",
    "to_lower": "char to_lo(char c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; } int main() { return to_lo('A'); }",
    "is_digit": "int is_dig(char c) { return c >= '0' && c <= '9'; } int main() { return is_dig('5'); }",
    "is_alpha": "int is_alp(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); } int main() { return is_alp('x'); }",
    "atoi_simple": "int my_atoi(char* s) { int n = 0; while (*s >= '0' && *s <= '9') n = n * 10 + (*s++ - '0'); return n; } int main() { return my_atoi(\"123\"); }",
    "itoa_simple": """
void my_itoa(int n, char* s) {
    int i = 0, t = n;
    do { s[i++] = t % 10 + '0'; t /= 10; } while (t);
    s[i] = 0;
    for (int j = 0; j < i/2; j++) { char c = s[j]; s[j] = s[i-1-j]; s[i-1-j] = c; }
}
int main() { char s[20]; my_itoa(123, s); return s[0]; }
""",
    "hex_digit": "char hex_dig(int n) { return n < 10 ? '0' + n : 'a' + n - 10; } int main() { return hex_dig(12); }",
    "bit_set": "int bit_set(int x, int n) { return x | (1 << n); } int main() { return bit_set(0, 3); }",
    "bit_clear": "int bit_clr(int x, int n) { return x & ~(1 << n); } int main() { return bit_clr(15, 1); }",
    "bit_toggle": "int bit_tog(int x, int n) { return x ^ (1 << n); } int main() { return bit_tog(5, 1); }",
    "bit_check": "int bit_chk(int x, int n) { return (x >> n) & 1; } int main() { return bit_chk(5, 2); }",
    "leading_zeros": """
int clz(unsigned x) {
    if (x == 0) return 32;
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
}
int main() { return clz(256); }
""",
    "swap_bits": "unsigned swap_bits(unsigned x) { return ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1); } int main() { return swap_bits(0xAA) & 0xFF; }",
    "parity": "int parity(unsigned x) { x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1; return x & 1; } int main() { return parity(7); }",
}

ALGORITHMIC_PROGRAMS = {
    "bubble_sort": """
void sort(int* arr, int n) {
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (arr[j] > arr[j+1]) {
                int t = arr[j]; arr[j] = arr[j+1]; arr[j+1] = t;
            }
}
int main() { int a[] = {5,2,8,1,9}; sort(a, 5); return a[0]; }
""",
    "binary_search": """
int search(int* arr, int n, int x) {
    int l = 0, r = n - 1;
    while (l <= r) {
        int m = (l + r) / 2;
        if (arr[m] == x) return m;
        if (arr[m] < x) l = m + 1;
        else r = m - 1;
    }
    return -1;
}
int main() { int a[] = {1,2,3,4,5}; return search(a, 5, 3); }
""",
    "gcd": """
int gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}
int main() { return gcd(48, 18); }
""",
    "power": """
int power(int base, int exp) {
    int result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
int main() { return power(2, 10); }
""",
    "prime_check": """
int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) return 0;
    return 1;
}
int main() { return is_prime(17); }
""",
}

ALL_PROGRAMS = {**REAL_C_PROGRAMS, **MORE_PROGRAMS, **ALGORITHMIC_PROGRAMS}


@dataclass
class CompiledBinary:
    name: str
    compiler: str
    opt_level: str
    path: Path
    disasm_intel: str
    disasm_att: str


def check_compiler(compiler: str) -> bool:
    """Check if compiler is available."""
    try:
        subprocess.run([compiler, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def compile_program(name: str, code: str, compiler: str, opt: str, tmpdir: Path) -> Optional[CompiledBinary]:
    """Compile C program with specified compiler and optimization."""
    src = tmpdir / f"{name}_{compiler}_{opt}.c"
    bin_path = tmpdir / f"{name}_{compiler}_{opt}"
    
    src.write_text(code)
    
    result = subprocess.run(
        [compiler, opt, "-o", str(bin_path), str(src)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return None
    
    # Get Intel syntax disassembly
    intel = subprocess.run(
        ["objdump", "-d", "-M", "intel", str(bin_path)],
        capture_output=True, text=True
    ).stdout
    
    # Get AT&T syntax disassembly
    att = subprocess.run(
        ["objdump", "-d", str(bin_path)],
        capture_output=True, text=True
    ).stdout
    
    return CompiledBinary(
        name=name,
        compiler=compiler,
        opt_level=opt,
        path=bin_path,
        disasm_intel=intel,
        disasm_att=att,
    )


def parse_disassembly(disasm: str, syntax: str) -> list[dict]:
    """Parse objdump output into instruction list."""
    instructions = []
    current_func = None
    
    for line in disasm.split('\n'):
        # Function header
        func_match = re.match(r'^[0-9a-f]+ <(\w+)>:', line)
        if func_match:
            current_func = func_match.group(1)
            continue
        
        # Instruction line
        instr_match = re.match(r'\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S+)\s*(.*)', line)
        if instr_match:
            addr, bytes_hex, mnemonic, operands = instr_match.groups()
            instructions.append({
                "offset": int(addr, 16),
                "bytes": bytes_hex.strip().replace(" ", ""),
                "mnemonic": mnemonic.strip(),
                "operands": operands.strip(),
                "function": current_func,
                "syntax": syntax,
            })
    
    return instructions


def generate_level0_samples(binary: CompiledBinary) -> list[dict]:
    """Generate Level 0 samples (bytes -> mnemonic) from binary."""
    samples = []
    
    # Intel syntax
    for instr in parse_disassembly(binary.disasm_intel, "intel"):
        if len(instr["bytes"]) > 0 and len(instr["bytes"]) <= 30:
            samples.append({
                "input": f"Bytes: {instr['bytes']}",
                "output": f"Instruction: {instr['mnemonic']}",
                "metadata": {
                    "syntax": "intel",
                    "compiler": binary.compiler,
                    "opt": binary.opt_level,
                    "function": instr["function"],
                }
            })
    
    # AT&T syntax
    for instr in parse_disassembly(binary.disasm_att, "att"):
        if len(instr["bytes"]) > 0 and len(instr["bytes"]) <= 30:
            samples.append({
                "input": f"Bytes: {instr['bytes']}",
                "output": f"Instruction: {instr['mnemonic']}",
                "metadata": {
                    "syntax": "att",
                    "compiler": binary.compiler,
                    "opt": binary.opt_level,
                    "function": instr["function"],
                }
            })
    
    return samples


def generate_level1_samples(binary: CompiledBinary) -> list[dict]:
    """Generate Level 1 samples (instruction -> semantics) from binary."""
    samples = []
    
    for syntax, disasm in [("intel", binary.disasm_intel), ("att", binary.disasm_att)]:
        for instr in parse_disassembly(disasm, syntax):
            m = instr["mnemonic"].lower()
            ops = instr["operands"]
            
            # Determine semantic category
            if m in ["push", "pop"]:
                category = "stack_operation"
            elif m in ["mov", "movl", "movq", "movb", "movw", "lea", "leaq"]:
                category = "data_transfer"
            elif m in ["add", "addl", "sub", "subl", "imul", "mul", "div", "idiv", "inc", "dec"]:
                category = "arithmetic"
            elif m in ["and", "or", "xor", "not", "shl", "shr", "sal", "sar", "rol", "ror"]:
                category = "bitwise"
            elif m in ["cmp", "cmpl", "test", "testl"]:
                category = "comparison"
            elif m.startswith("j") or m in ["jmp", "jmpq"]:
                category = "control_flow"
            elif m in ["call", "callq", "ret", "retq"]:
                category = "function_call"
            elif m in ["nop", "endbr64", "endbr32"]:
                category = "no_operation"
            else:
                category = "other"
            
            samples.append({
                "input": f"Instruction: {instr['mnemonic']} {ops}",
                "output": f"Category: {category}",
                "metadata": {
                    "syntax": syntax,
                    "compiler": binary.compiler,
                    "opt": binary.opt_level,
                    "mnemonic": m,
                }
            })
    
    return samples


def generate_level2_samples(binary: CompiledBinary) -> list[dict]:
    """Generate Level 2 samples (instruction sequence -> CFG) from binary."""
    samples = []
    
    for syntax, disasm in [("intel", binary.disasm_intel), ("att", binary.disasm_att)]:
        instructions = parse_disassembly(disasm, syntax)
        
        # Group by function
        functions = {}
        for instr in instructions:
            func = instr.get("function", "unknown")
            if func not in functions:
                functions[func] = []
            functions[func].append(instr)
        
        for func_name, func_instrs in functions.items():
            if len(func_instrs) < 3 or func_name.startswith("_"):
                continue
            
            # Analyze CFG properties
            has_loop = False
            has_conditional = False
            has_call = False
            block_count = 1
            
            addresses = [i["offset"] for i in func_instrs]
            
            for instr in func_instrs:
                m = instr["mnemonic"].lower()
                ops = instr["operands"]
                
                if m.startswith("j") and m != "jmp":
                    has_conditional = True
                    block_count += 1
                elif m == "jmp":
                    # Check if backward jump (loop)
                    target_match = re.search(r'([0-9a-f]+)', ops)
                    if target_match:
                        target = int(target_match.group(1), 16)
                        if target < instr["offset"]:
                            has_loop = True
                    block_count += 1
                elif m in ["call", "callq"]:
                    has_call = True
            
            # Format input
            instr_strs = []
            for i in func_instrs[:20]:
                instr_strs.append(f"{i['offset']:#x}:{i['mnemonic']} {i['operands']}")
            
            input_text = f"Function: {func_name}\n" + "\n".join(instr_strs)
            
            # Format output
            output_parts = [f"blocks: ~{block_count}"]
            if has_conditional:
                output_parts.append("has_conditionals: true")
            if has_loop:
                output_parts.append("has_loops: true")
            if has_call:
                output_parts.append("has_calls: true")
            
            samples.append({
                "input": input_text,
                "output": "; ".join(output_parts),
                "metadata": {
                    "syntax": syntax,
                    "compiler": binary.compiler,
                    "opt": binary.opt_level,
                    "function": func_name,
                }
            })
    
    return samples


def generate_dataset():
    """Generate complete dataset from real binaries."""
    print("=" * 70)
    print("REAL BINARY DATASET GENERATOR")
    print("=" * 70)
    
    # Check available compilers
    compilers = []
    for c in ["gcc", "clang"]:
        if check_compiler(c):
            compilers.append(c)
            print(f"  ✓ {c} available")
        else:
            print(f"  ✗ {c} not found")
    
    if not compilers:
        print("ERROR: No compilers available!")
        return
    
    opt_levels = ["-O0", "-O1", "-O2", "-O3"]
    
    level0_samples = []
    level1_samples = []
    level2_samples = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        total = len(ALL_PROGRAMS) * len(compilers) * len(opt_levels)
        count = 0
        
        print(f"\nCompiling {total} binary variants...")
        
        for name, code in ALL_PROGRAMS.items():
            for compiler in compilers:
                for opt in opt_levels:
                    count += 1
                    binary = compile_program(name, code, compiler, opt, tmpdir)
                    
                    if binary:
                        l0 = generate_level0_samples(binary)
                        l1 = generate_level1_samples(binary)
                        l2 = generate_level2_samples(binary)
                        
                        level0_samples.extend(l0)
                        level1_samples.extend(l1)
                        level2_samples.extend(l2)
                        
                        if count % 20 == 0:
                            print(f"  Progress: {count}/{total} ({len(level0_samples)} L0, {len(level1_samples)} L1, {len(level2_samples)} L2)")
    
    # Deduplicate by hash
    def dedupe(samples):
        seen = set()
        unique = []
        for s in samples:
            h = hashlib.md5((s["input"] + s["output"]).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(s)
        return unique
    
    level0_samples = dedupe(level0_samples)
    level1_samples = dedupe(level1_samples)
    level2_samples = dedupe(level2_samples)
    
    print(f"\nUnique samples:")
    print(f"  Level 0: {len(level0_samples)}")
    print(f"  Level 1: {len(level1_samples)}")
    print(f"  Level 2: {len(level2_samples)}")
    
    # Save datasets
    base = Path(__file__).parent.parent
    
    for level, samples in [("level0", level0_samples), ("level1", level1_samples), ("level2", level2_samples)]:
        path = base / f"{level}_real" / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        random.shuffle(samples)
        
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        print(f"  Saved {path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    # Syntax distribution
    for level, samples in [("Level 0", level0_samples), ("Level 1", level1_samples), ("Level 2", level2_samples)]:
        intel = sum(1 for s in samples if s.get("metadata", {}).get("syntax") == "intel")
        att = sum(1 for s in samples if s.get("metadata", {}).get("syntax") == "att")
        print(f"\n{level}:")
        print(f"  Intel syntax: {intel}")
        print(f"  AT&T syntax: {att}")
        
        # Compiler distribution
        gcc = sum(1 for s in samples if s.get("metadata", {}).get("compiler") == "gcc")
        clang = sum(1 for s in samples if s.get("metadata", {}).get("compiler") == "clang")
        print(f"  GCC: {gcc}")
        print(f"  Clang: {clang}")
        
        # Optimization distribution
        for opt in opt_levels:
            n = sum(1 for s in samples if s.get("metadata", {}).get("opt") == opt)
            print(f"  {opt}: {n}")


if __name__ == "__main__":
    generate_dataset()
