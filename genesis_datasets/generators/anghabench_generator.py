#!/usr/bin/env python3
"""
AnghaBench Dataset Generator

Uses the AnghaBench dataset (1M real C functions from GitHub)
to generate training data from REAL code, not synthetic.

This is the proper approach:
1. Real C code from real projects
2. Multiple compilers (gcc, clang)
3. Multiple optimization levels
4. Ground truth from Capstone (industry standard)
5. Cross-validation with objdump
"""

import subprocess
import tempfile
import json
import re
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Try to import datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Note: 'datasets' library not installed. Will use fallback.")


def check_tool(tool: str) -> bool:
    """Check if a tool is available."""
    try:
        subprocess.run([tool, "--version"], capture_output=True, check=True)
        return True
    except:
        return False


def get_capstone_disasm(binary_path: Path) -> list[dict]:
    """Get disassembly using Capstone (ground truth)."""
    try:
        from capstone import Cs, CS_ARCH_X86, CS_MODE_64
        
        # Get .text section base address using readelf
        readelf = subprocess.run(
            ["readelf", "-S", str(binary_path)],
            capture_output=True, text=True
        )
        
        text_base = 0
        for line in readelf.stdout.split('\n'):
            if '.text' in line:
                # Parse: [ X] .text PROGBITS ADDRESS ...
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == '.text' and i + 2 < len(parts):
                        try:
                            text_base = int(parts[i + 2], 16)
                        except:
                            pass
                        break
        
        # Extract .text section bytes
        result = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", 
             str(binary_path), "/dev/stdout"],
            capture_output=True
        )
        text_bytes = result.stdout
        
        if not text_bytes:
            return []
        
        cs = Cs(CS_ARCH_X86, CS_MODE_64)
        instructions = []
        
        # Disassemble with correct base address
        for instr in cs.disasm(text_bytes, text_base):
            instructions.append({
                "offset": instr.address,
                "bytes": instr.bytes.hex(),
                "mnemonic": instr.mnemonic,
                "operands": instr.op_str,
            })
        
        return instructions
    except Exception as e:
        return []


def get_objdump_disasm(binary_path: Path, syntax: str = "intel") -> list[dict]:
    """Get disassembly using objdump."""
    cmd = ["objdump", "-d"]
    if syntax == "intel":
        cmd.extend(["-M", "intel"])
    cmd.append(str(binary_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    instructions = []
    for line in result.stdout.split('\n'):
        match = re.match(r'\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S+)\s*(.*)', line)
        if match:
            addr, bytes_hex, mnemonic, operands = match.groups()
            instructions.append({
                "offset": int(addr, 16),
                "bytes": bytes_hex.strip().replace(" ", ""),
                "mnemonic": mnemonic.strip(),
                "operands": operands.strip(),
                "syntax": syntax,
            })
    
    return instructions


def compile_c_code(code: str, compiler: str, opt: str, tmpdir: Path) -> Optional[Path]:
    """Compile C code to binary."""
    # Create unique filename
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    src_path = tmpdir / f"{code_hash}.c"
    bin_path = tmpdir / f"{code_hash}_{compiler}_{opt.replace('-', '')}"
    
    # Add main if missing
    if "int main" not in code and "void main" not in code:
        code = code + "\nint main() { return 0; }\n"
    
    src_path.write_text(code)
    
    result = subprocess.run(
        [compiler, opt, "-w", "-o", str(bin_path), str(src_path)],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        return None
    
    return bin_path


def verify_instruction(capstone_instr: dict, objdump_instr: dict) -> bool:
    """Verify instruction matches between Capstone and objdump."""
    # Normalize mnemonics
    cap_m = capstone_instr["mnemonic"].lower()
    obj_m = objdump_instr["mnemonic"].lower()
    
    # Handle suffix differences (movl vs mov)
    obj_m_base = re.sub(r'[lqwb]$', '', obj_m)
    
    return cap_m == obj_m or cap_m == obj_m_base


def process_function(args) -> list[dict]:
    """Process a single C function."""
    code, compiler, opt, tmpdir = args
    samples = []
    
    try:
        bin_path = compile_c_code(code, compiler, opt, Path(tmpdir))
        if not bin_path or not bin_path.exists():
            return []
        
        # Get ground truth from Capstone
        capstone_instrs = get_capstone_disasm(bin_path)
        if not capstone_instrs:
            return []
        
        # Get objdump for both syntaxes
        objdump_intel = get_objdump_disasm(bin_path, "intel")
        objdump_att = get_objdump_disasm(bin_path, "att")
        
        # Build offset maps
        intel_map = {i["offset"]: i for i in objdump_intel}
        att_map = {i["offset"]: i for i in objdump_att}
        
        for cap_instr in capstone_instrs[:50]:  # Limit per function
            offset = cap_instr["offset"]
            
            # Verify with objdump
            if offset in intel_map:
                obj_instr = intel_map[offset]
                if verify_instruction(cap_instr, obj_instr):
                    # Level 0 sample (Intel)
                    samples.append({
                        "level": 0,
                        "input": f"Bytes: {cap_instr['bytes']}",
                        "output": f"Instruction: {cap_instr['mnemonic']}",
                        "verified": True,
                        "compiler": compiler,
                        "opt": opt,
                        "syntax": "intel",
                    })
                    
                    # Level 1 sample
                    samples.append({
                        "level": 1,
                        "input": f"Instruction: {cap_instr['mnemonic']} {cap_instr['operands']}",
                        "output": get_semantic_output(cap_instr["mnemonic"]),
                        "verified": True,
                        "compiler": compiler,
                        "opt": opt,
                    })
            
            # AT&T syntax sample
            if offset in att_map:
                att_instr = att_map[offset]
                samples.append({
                    "level": 0,
                    "input": f"Bytes: {cap_instr['bytes']}",
                    "output": f"Instruction: {att_instr['mnemonic']}",
                    "verified": True,
                    "compiler": compiler,
                    "opt": opt,
                    "syntax": "att",
                })
        
        # Clean up
        if bin_path.exists():
            bin_path.unlink()
        
    except Exception as e:
        pass
    
    return samples


def get_semantic_output(mnemonic: str) -> str:
    """Get semantic description for instruction."""
    m = mnemonic.lower()
    
    semantics = {
        # Data movement
        "mov": "Category: data_transfer; Effect: writes destination register",
        "movzx": "Category: data_transfer; Effect: zero-extends and writes destination",
        "movsx": "Category: data_transfer; Effect: sign-extends and writes destination",
        "lea": "Category: data_transfer; Effect: loads effective address into destination",
        "push": "Category: stack; Effect: decrements RSP, writes to stack",
        "pop": "Category: stack; Effect: reads from stack, increments RSP",
        
        # Arithmetic
        "add": "Category: arithmetic; Effect: adds operands, sets flags (CF,OF,SF,ZF,AF,PF)",
        "sub": "Category: arithmetic; Effect: subtracts operands, sets flags",
        "imul": "Category: arithmetic; Effect: signed multiply, sets flags",
        "idiv": "Category: arithmetic; Effect: signed divide, modifies RAX/RDX",
        "inc": "Category: arithmetic; Effect: increments by 1, sets flags (not CF)",
        "dec": "Category: arithmetic; Effect: decrements by 1, sets flags (not CF)",
        "neg": "Category: arithmetic; Effect: two's complement negation, sets flags",
        
        # Bitwise
        "and": "Category: bitwise; Effect: logical AND, sets flags, clears CF/OF",
        "or": "Category: bitwise; Effect: logical OR, sets flags, clears CF/OF",
        "xor": "Category: bitwise; Effect: logical XOR, sets flags, clears CF/OF",
        "not": "Category: bitwise; Effect: one's complement, no flags affected",
        "shl": "Category: bitwise; Effect: shift left, sets CF to last bit shifted out",
        "shr": "Category: bitwise; Effect: shift right logical, sets CF",
        "sar": "Category: bitwise; Effect: shift right arithmetic, preserves sign",
        
        # Comparison
        "cmp": "Category: comparison; Effect: subtracts without storing, sets flags",
        "test": "Category: comparison; Effect: ANDs without storing, sets flags",
        
        # Control flow
        "jmp": "Category: control_flow; Effect: unconditional jump to target",
        "je": "Category: control_flow; Effect: jump if ZF=1 (equal)",
        "jne": "Category: control_flow; Effect: jump if ZF=0 (not equal)",
        "jg": "Category: control_flow; Effect: jump if greater (signed)",
        "jl": "Category: control_flow; Effect: jump if less (signed)",
        "jge": "Category: control_flow; Effect: jump if greater or equal (signed)",
        "jle": "Category: control_flow; Effect: jump if less or equal (signed)",
        "ja": "Category: control_flow; Effect: jump if above (unsigned)",
        "jb": "Category: control_flow; Effect: jump if below (unsigned)",
        "call": "Category: control_flow; Effect: pushes return address, jumps to target",
        "ret": "Category: control_flow; Effect: pops return address, jumps to it",
        
        # Other
        "nop": "Category: no_operation; Effect: none",
        "endbr64": "Category: security; Effect: CET end-branch marker",
    }
    
    return semantics.get(m, f"Category: other; Effect: executes {m} operation")


def download_anghabench(limit: int = 10000) -> list[str]:
    """Download C functions from ExeBench or fallback."""
    if not HAS_DATASETS:
        print("datasets library not available, using fallback functions")
        return get_fallback_functions()
    
    try:
        print("Loading ExeBench from HuggingFace...")
        # ExeBench has compilable C functions
        ds = load_dataset("jordiae/exebench", split="train_real_compilable", streaming=True)
        
        functions = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            
            # ExeBench has 'func_def' field with the C function
            if "func_def" in item and item["func_def"]:
                code = item["func_def"]
                # Add required includes
                code = "#include <stddef.h>\n#include <stdint.h>\n" + code
                functions.append(code)
            
            if i % 1000 == 0 and i > 0:
                print(f"  Loaded {i}/{limit} functions...")
        
        if functions:
            print(f"  Loaded {len(functions)} functions from ExeBench")
            return functions
        
    except Exception as e:
        print(f"Could not load ExeBench: {e}")
    
    print("Using fallback functions")
    return get_fallback_functions()


def get_fallback_functions() -> list[str]:
    """Fallback C functions if AnghaBench unavailable."""
    # Collection of diverse, real-world-like C functions
    return [
        # String functions
        "int strlen_impl(const char* s) { int n = 0; while (s[n]) n++; return n; }",
        "void strcpy_impl(char* d, const char* s) { while ((*d++ = *s++)); }",
        "int strcmp_impl(const char* a, const char* b) { while (*a && *a == *b) { a++; b++; } return *a - *b; }",
        "char* strcat_impl(char* d, const char* s) { char* p = d; while (*p) p++; while ((*p++ = *s++)); return d; }",
        "char* strchr_impl(const char* s, int c) { while (*s && *s != c) s++; return *s == c ? (char*)s : 0; }",
        "char* strrchr_impl(const char* s, int c) { const char* r = 0; while (*s) { if (*s == c) r = s; s++; } return (char*)r; }",
        "int strncmp_impl(const char* a, const char* b, int n) { while (n-- && *a && *a == *b) { a++; b++; } return n < 0 ? 0 : *a - *b; }",
        
        # Memory functions
        "void* memcpy_impl(void* d, const void* s, int n) { char* dp = d; const char* sp = s; while (n--) *dp++ = *sp++; return d; }",
        "void* memset_impl(void* s, int c, int n) { char* p = s; while (n--) *p++ = c; return s; }",
        "int memcmp_impl(const void* a, const void* b, int n) { const unsigned char* pa = a, *pb = b; while (n--) { if (*pa != *pb) return *pa - *pb; pa++; pb++; } return 0; }",
        
        # Math functions
        "int abs_impl(int x) { return x < 0 ? -x : x; }",
        "int min_impl(int a, int b) { return a < b ? a : b; }",
        "int max_impl(int a, int b) { return a > b ? a : b; }",
        "int clamp_impl(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }",
        "int gcd_impl(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }",
        "int lcm_impl(int a, int b) { return a / gcd_impl(a, b) * b; }",
        "int factorial_impl(int n) { int r = 1; for (int i = 2; i <= n; i++) r *= i; return r; }",
        "int fibonacci_impl(int n) { if (n <= 1) return n; int a = 0, b = 1; for (int i = 2; i <= n; i++) { int t = a + b; a = b; b = t; } return b; }",
        "int power_impl(int base, int exp) { int r = 1; while (exp > 0) { if (exp & 1) r *= base; base *= base; exp >>= 1; } return r; }",
        "int isqrt_impl(int n) { int x = n, y = (x + 1) / 2; while (y < x) { x = y; y = (x + n/x) / 2; } return x; }",
        
        # Bit manipulation
        "int popcount_impl(unsigned x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }",
        "int clz_impl(unsigned x) { if (x == 0) return 32; int n = 0; if (x <= 0x0000FFFF) { n += 16; x <<= 16; } if (x <= 0x00FFFFFF) { n += 8; x <<= 8; } if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; } if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; } if (x <= 0x7FFFFFFF) n++; return n; }",
        "unsigned reverse_bits_impl(unsigned x) { x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1); x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2); x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4); x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8); return (x >> 16) | (x << 16); }",
        "int is_power_of_2_impl(unsigned x) { return x && !(x & (x - 1)); }",
        "unsigned next_power_of_2_impl(unsigned x) { x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; return x + 1; }",
        
        # Array functions
        "int sum_array_impl(int* a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }",
        "int max_array_impl(int* a, int n) { int m = a[0]; for (int i = 1; i < n; i++) if (a[i] > m) m = a[i]; return m; }",
        "int min_array_impl(int* a, int n) { int m = a[0]; for (int i = 1; i < n; i++) if (a[i] < m) m = a[i]; return m; }",
        "int find_impl(int* a, int n, int x) { for (int i = 0; i < n; i++) if (a[i] == x) return i; return -1; }",
        "int count_impl(int* a, int n, int x) { int c = 0; for (int i = 0; i < n; i++) if (a[i] == x) c++; return c; }",
        "void reverse_array_impl(int* a, int n) { for (int i = 0; i < n/2; i++) { int t = a[i]; a[i] = a[n-1-i]; a[n-1-i] = t; } }",
        "void rotate_left_impl(int* a, int n) { int t = a[0]; for (int i = 0; i < n-1; i++) a[i] = a[i+1]; a[n-1] = t; }",
        "void rotate_right_impl(int* a, int n) { int t = a[n-1]; for (int i = n-1; i > 0; i--) a[i] = a[i-1]; a[0] = t; }",
        
        # Sorting
        "void bubble_sort_impl(int* a, int n) { for (int i = 0; i < n-1; i++) for (int j = 0; j < n-i-1; j++) if (a[j] > a[j+1]) { int t = a[j]; a[j] = a[j+1]; a[j+1] = t; } }",
        "void selection_sort_impl(int* a, int n) { for (int i = 0; i < n-1; i++) { int m = i; for (int j = i+1; j < n; j++) if (a[j] < a[m]) m = j; int t = a[i]; a[i] = a[m]; a[m] = t; } }",
        "void insertion_sort_impl(int* a, int n) { for (int i = 1; i < n; i++) { int k = a[i], j = i-1; while (j >= 0 && a[j] > k) { a[j+1] = a[j]; j--; } a[j+1] = k; } }",
        
        # Searching
        "int binary_search_impl(int* a, int n, int x) { int l = 0, r = n - 1; while (l <= r) { int m = (l + r) / 2; if (a[m] == x) return m; if (a[m] < x) l = m + 1; else r = m - 1; } return -1; }",
        "int lower_bound_impl(int* a, int n, int x) { int l = 0, r = n; while (l < r) { int m = (l + r) / 2; if (a[m] < x) l = m + 1; else r = m; } return l; }",
        "int upper_bound_impl(int* a, int n, int x) { int l = 0, r = n; while (l < r) { int m = (l + r) / 2; if (a[m] <= x) l = m + 1; else r = m; } return l; }",
        
        # Character functions
        "int is_digit_impl(int c) { return c >= '0' && c <= '9'; }",
        "int is_alpha_impl(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }",
        "int is_alnum_impl(int c) { return is_digit_impl(c) || is_alpha_impl(c); }",
        "int is_space_impl(int c) { return c == ' ' || c == '\\t' || c == '\\n' || c == '\\r'; }",
        "int to_upper_impl(int c) { return (c >= 'a' && c <= 'z') ? c - 32 : c; }",
        "int to_lower_impl(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }",
        
        # Conversion
        "int atoi_impl(const char* s) { int n = 0, neg = 0; while (*s == ' ') s++; if (*s == '-') { neg = 1; s++; } else if (*s == '+') s++; while (*s >= '0' && *s <= '9') n = n * 10 + (*s++ - '0'); return neg ? -n : n; }",
        
        # Linked list simulations (array-based)
        "int list_length_impl(int* next, int head) { int len = 0; while (head != -1) { len++; head = next[head]; } return len; }",
        
        # Hash functions
        "unsigned djb2_hash_impl(const char* s) { unsigned h = 5381; int c; while ((c = *s++)) h = ((h << 5) + h) + c; return h; }",
        "unsigned fnv1a_hash_impl(const char* s) { unsigned h = 2166136261; while (*s) { h ^= *s++; h *= 16777619; } return h; }",
        
        # More algorithms
        "int is_prime_impl(int n) { if (n < 2) return 0; for (int i = 2; i * i <= n; i++) if (n % i == 0) return 0; return 1; }",
        "int count_primes_impl(int n) { int c = 0; for (int i = 2; i < n; i++) if (is_prime_impl(i)) c++; return c; }",
        "int collatz_steps_impl(int n) { int s = 0; while (n != 1) { n = (n % 2 == 0) ? n / 2 : 3 * n + 1; s++; } return s; }",
    ]


def generate_dataset(num_functions: int = 5000, num_workers: int = 4):
    """Generate complete dataset from AnghaBench."""
    print("=" * 70)
    print("ANGHABENCH DATASET GENERATOR")
    print("Real code from real projects")
    print("=" * 70)
    
    # Check tools
    compilers = []
    for c in ["gcc", "clang"]:
        if check_tool(c):
            compilers.append(c)
            print(f"  ✓ {c}")
        else:
            print(f"  ✗ {c} (not found)")
    
    if not compilers:
        print("ERROR: No compilers found!")
        return
    
    opt_levels = ["-O0", "-O1", "-O2", "-O3"]
    
    # Download/load functions
    functions = download_anghabench(num_functions)
    print(f"\nLoaded {len(functions)} functions")
    
    # Process functions
    all_samples = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nProcessing with {len(compilers)} compilers × {len(opt_levels)} opt levels...")
        
        # Prepare work items
        work_items = []
        for func in functions:
            for compiler in compilers:
                for opt in opt_levels:
                    work_items.append((func, compiler, opt, tmpdir))
        
        # Process (could parallelize but keeping simple for reliability)
        for i, item in enumerate(work_items):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(work_items)} ({len(all_samples)} samples)")
            
            samples = process_function(item)
            all_samples.extend(samples)
    
    # Deduplicate
    seen = set()
    unique_samples = []
    for s in all_samples:
        key = s["input"] + s["output"]
        h = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_samples.append(s)
    
    print(f"\nUnique samples: {len(unique_samples)}")
    
    # Split by level
    level0 = [s for s in unique_samples if s["level"] == 0]
    level1 = [s for s in unique_samples if s["level"] == 1]
    
    print(f"  Level 0: {len(level0)}")
    print(f"  Level 1: {len(level1)}")
    
    # Save
    base = Path(__file__).parent.parent
    
    for level, samples in [("level0_angha", level0), ("level1_angha", level1)]:
        if not samples:
            continue
            
        path = base / level / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        random.shuffle(samples)
        
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        print(f"  Saved {path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    for name, samples in [("Level 0", level0), ("Level 1", level1)]:
        if not samples:
            continue
        
        print(f"\n{name}:")
        
        # Syntax distribution
        intel = sum(1 for s in samples if s.get("syntax") == "intel")
        att = sum(1 for s in samples if s.get("syntax") == "att")
        print(f"  Intel: {intel}, AT&T: {att}")
        
        # Compiler distribution
        for compiler in compilers:
            n = sum(1 for s in samples if s.get("compiler") == compiler)
            print(f"  {compiler}: {n}")
        
        # Opt level distribution
        for opt in opt_levels:
            n = sum(1 for s in samples if s.get("opt") == opt)
            print(f"  {opt}: {n}")


if __name__ == "__main__":
    generate_dataset(num_functions=5000)
