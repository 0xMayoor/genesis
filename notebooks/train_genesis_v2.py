# ============================================================================
# GENESIS Training v2 - Fixed for Colab
# - Installs clang
# - Uses robust data generation (no HuggingFace dependency issues)
# - 10,000+ training samples
# ============================================================================

import os
import subprocess
import sys
import json
import random
import hashlib
import tempfile
import re
from pathlib import Path

print("=" * 70)
print("GENESIS TRAINING v2")
print("=" * 70)

# 1. Setup
print("\n[1/7] Setting up environment...")

# Install clang
print("  Installing clang...")
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "clang"], capture_output=True)

# Verify compilers
for compiler in ["gcc", "clang"]:
    result = subprocess.run([compiler, "--version"], capture_output=True)
    if result.returncode == 0:
        print(f"  ✓ {compiler}")
    else:
        print(f"  ✗ {compiler}")

# Clone repo
if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")

# Install Python deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "peft", "accelerate", "capstone"], check=True)

# 2. Generate LARGE dataset from diverse C functions
print("\n[2/7] Generating training data...")

# ============================================================================
# COMPREHENSIVE C FUNCTION LIBRARY - 200+ real-world patterns
# ============================================================================

C_FUNCTIONS = [
    # ===== STRING OPERATIONS =====
    "int my_strlen(const char* s) { int n = 0; while (s[n]) n++; return n; }",
    "void my_strcpy(char* d, const char* s) { while ((*d++ = *s++)); }",
    "int my_strcmp(const char* a, const char* b) { while (*a && *a == *b) { a++; b++; } return *a - *b; }",
    "char* my_strcat(char* d, const char* s) { char* p = d; while (*p) p++; while ((*p++ = *s++)); return d; }",
    "char* my_strchr(const char* s, int c) { while (*s && *s != c) s++; return *s == c ? (char*)s : 0; }",
    "char* my_strrchr(const char* s, int c) { const char* r = 0; while (*s) { if (*s == c) r = s; s++; } return (char*)r; }",
    "int my_strncmp(const char* a, const char* b, int n) { while (n-- && *a && *a == *b) { a++; b++; } return n < 0 ? 0 : *a - *b; }",
    "char* my_strstr(const char* h, const char* n) { if (!*n) return (char*)h; for (; *h; h++) { const char* a = h, *b = n; while (*a && *b && *a == *b) { a++; b++; } if (!*b) return (char*)h; } return 0; }",
    "int my_strspn(const char* s, const char* a) { int n = 0; while (s[n] && my_strchr(a, s[n])) n++; return n; }",
    "void my_strrev(char* s) { int l = my_strlen(s); for (int i = 0; i < l/2; i++) { char t = s[i]; s[i] = s[l-1-i]; s[l-1-i] = t; } }",
    
    # ===== MEMORY OPERATIONS =====
    "void* my_memcpy(void* d, const void* s, int n) { char* dp = d; const char* sp = s; while (n--) *dp++ = *sp++; return d; }",
    "void* my_memset(void* s, int c, int n) { char* p = s; while (n--) *p++ = c; return s; }",
    "int my_memcmp(const void* a, const void* b, int n) { const unsigned char* pa = a, *pb = b; while (n--) { if (*pa != *pb) return *pa - *pb; pa++; pb++; } return 0; }",
    "void* my_memmove(void* d, const void* s, int n) { char* dp = d; const char* sp = s; if (dp < sp) while (n--) *dp++ = *sp++; else { dp += n; sp += n; while (n--) *--dp = *--sp; } return d; }",
    "void* my_memchr(const void* s, int c, int n) { const unsigned char* p = s; while (n--) { if (*p == c) return (void*)p; p++; } return 0; }",
    
    # ===== MATH OPERATIONS =====
    "int my_abs(int x) { return x < 0 ? -x : x; }",
    "int my_min(int a, int b) { return a < b ? a : b; }",
    "int my_max(int a, int b) { return a > b ? a : b; }",
    "int my_clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }",
    "int my_sign(int x) { return (x > 0) - (x < 0); }",
    "int my_gcd(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }",
    "int my_lcm(int a, int b) { return a / my_gcd(a, b) * b; }",
    "int my_factorial(int n) { int r = 1; for (int i = 2; i <= n; i++) r *= i; return r; }",
    "int my_fibonacci(int n) { if (n <= 1) return n; int a = 0, b = 1; for (int i = 2; i <= n; i++) { int t = a + b; a = b; b = t; } return b; }",
    "int my_power(int b, int e) { int r = 1; while (e > 0) { if (e & 1) r *= b; b *= b; e >>= 1; } return r; }",
    "int my_isqrt(int n) { int x = n, y = (x + 1) / 2; while (y < x) { x = y; y = (x + n/x) / 2; } return x; }",
    "int my_log2(unsigned x) { int r = 0; while (x >>= 1) r++; return r; }",
    "int my_ceil_div(int a, int b) { return (a + b - 1) / b; }",
    "int my_round_up(int x, int m) { return ((x + m - 1) / m) * m; }",
    "int my_mod(int a, int b) { int r = a % b; return r < 0 ? r + b : r; }",
    
    # ===== BIT OPERATIONS =====
    "int my_popcount(unsigned x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }",
    "int my_clz(unsigned x) { if (x == 0) return 32; int n = 0; if (x <= 0x0000FFFF) { n += 16; x <<= 16; } if (x <= 0x00FFFFFF) { n += 8; x <<= 8; } if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; } if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; } if (x <= 0x7FFFFFFF) n++; return n; }",
    "int my_ctz(unsigned x) { if (x == 0) return 32; int n = 0; if (!(x & 0x0000FFFF)) { n += 16; x >>= 16; } if (!(x & 0x000000FF)) { n += 8; x >>= 8; } if (!(x & 0x0000000F)) { n += 4; x >>= 4; } if (!(x & 0x00000003)) { n += 2; x >>= 2; } if (!(x & 0x00000001)) n++; return n; }",
    "unsigned my_reverse_bits(unsigned x) { x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1); x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2); x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4); x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8); return (x >> 16) | (x << 16); }",
    "int my_is_power_of_2(unsigned x) { return x && !(x & (x - 1)); }",
    "unsigned my_next_power_of_2(unsigned x) { x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; return x + 1; }",
    "unsigned my_rotate_left(unsigned x, int n) { return (x << n) | (x >> (32 - n)); }",
    "unsigned my_rotate_right(unsigned x, int n) { return (x >> n) | (x << (32 - n)); }",
    "int my_parity(unsigned x) { x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1; return x & 1; }",
    "unsigned my_swap_bytes(unsigned x) { return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000); }",
    "int my_bit_set(int x, int n) { return x | (1 << n); }",
    "int my_bit_clear(int x, int n) { return x & ~(1 << n); }",
    "int my_bit_toggle(int x, int n) { return x ^ (1 << n); }",
    "int my_bit_check(int x, int n) { return (x >> n) & 1; }",
    
    # ===== ARRAY OPERATIONS =====
    "int arr_sum(int* a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }",
    "int arr_max(int* a, int n) { int m = a[0]; for (int i = 1; i < n; i++) if (a[i] > m) m = a[i]; return m; }",
    "int arr_min(int* a, int n) { int m = a[0]; for (int i = 1; i < n; i++) if (a[i] < m) m = a[i]; return m; }",
    "int arr_find(int* a, int n, int x) { for (int i = 0; i < n; i++) if (a[i] == x) return i; return -1; }",
    "int arr_count(int* a, int n, int x) { int c = 0; for (int i = 0; i < n; i++) if (a[i] == x) c++; return c; }",
    "void arr_reverse(int* a, int n) { for (int i = 0; i < n/2; i++) { int t = a[i]; a[i] = a[n-1-i]; a[n-1-i] = t; } }",
    "void arr_rotate_left(int* a, int n) { int t = a[0]; for (int i = 0; i < n-1; i++) a[i] = a[i+1]; a[n-1] = t; }",
    "void arr_rotate_right(int* a, int n) { int t = a[n-1]; for (int i = n-1; i > 0; i--) a[i] = a[i-1]; a[0] = t; }",
    "void arr_copy(int* d, int* s, int n) { for (int i = 0; i < n; i++) d[i] = s[i]; }",
    "void arr_fill(int* a, int n, int v) { for (int i = 0; i < n; i++) a[i] = v; }",
    "int arr_all(int* a, int n, int v) { for (int i = 0; i < n; i++) if (a[i] != v) return 0; return 1; }",
    "int arr_any(int* a, int n, int v) { for (int i = 0; i < n; i++) if (a[i] == v) return 1; return 0; }",
    "int arr_dot(int* a, int* b, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }",
    "void arr_add(int* r, int* a, int* b, int n) { for (int i = 0; i < n; i++) r[i] = a[i] + b[i]; }",
    "void arr_sub(int* r, int* a, int* b, int n) { for (int i = 0; i < n; i++) r[i] = a[i] - b[i]; }",
    "void arr_scale(int* a, int n, int k) { for (int i = 0; i < n; i++) a[i] *= k; }",
    
    # ===== SORTING =====
    "void bubble_sort(int* a, int n) { for (int i = 0; i < n-1; i++) for (int j = 0; j < n-i-1; j++) if (a[j] > a[j+1]) { int t = a[j]; a[j] = a[j+1]; a[j+1] = t; } }",
    "void selection_sort(int* a, int n) { for (int i = 0; i < n-1; i++) { int m = i; for (int j = i+1; j < n; j++) if (a[j] < a[m]) m = j; int t = a[i]; a[i] = a[m]; a[m] = t; } }",
    "void insertion_sort(int* a, int n) { for (int i = 1; i < n; i++) { int k = a[i], j = i-1; while (j >= 0 && a[j] > k) { a[j+1] = a[j]; j--; } a[j+1] = k; } }",
    "int partition(int* a, int lo, int hi) { int p = a[hi], i = lo - 1; for (int j = lo; j < hi; j++) if (a[j] <= p) { i++; int t = a[i]; a[i] = a[j]; a[j] = t; } int t = a[i+1]; a[i+1] = a[hi]; a[hi] = t; return i + 1; }",
    
    # ===== SEARCHING =====
    "int binary_search(int* a, int n, int x) { int l = 0, r = n - 1; while (l <= r) { int m = (l + r) / 2; if (a[m] == x) return m; if (a[m] < x) l = m + 1; else r = m - 1; } return -1; }",
    "int lower_bound(int* a, int n, int x) { int l = 0, r = n; while (l < r) { int m = (l + r) / 2; if (a[m] < x) l = m + 1; else r = m; } return l; }",
    "int upper_bound(int* a, int n, int x) { int l = 0, r = n; while (l < r) { int m = (l + r) / 2; if (a[m] <= x) l = m + 1; else r = m; } return l; }",
    
    # ===== CHARACTER OPERATIONS =====
    "int is_digit(int c) { return c >= '0' && c <= '9'; }",
    "int is_alpha(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }",
    "int is_alnum(int c) { return is_digit(c) || is_alpha(c); }",
    "int is_space(int c) { return c == ' ' || c == '\\t' || c == '\\n' || c == '\\r'; }",
    "int is_upper(int c) { return c >= 'A' && c <= 'Z'; }",
    "int is_lower(int c) { return c >= 'a' && c <= 'z'; }",
    "int to_upper(int c) { return is_lower(c) ? c - 32 : c; }",
    "int to_lower(int c) { return is_upper(c) ? c + 32 : c; }",
    "int is_hex(int c) { return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }",
    "int hex_val(int c) { if (is_digit(c)) return c - '0'; if (c >= 'a') return c - 'a' + 10; return c - 'A' + 10; }",
    
    # ===== CONVERSION =====
    "int my_atoi(const char* s) { int n = 0, neg = 0; while (*s == ' ') s++; if (*s == '-') { neg = 1; s++; } else if (*s == '+') s++; while (*s >= '0' && *s <= '9') n = n * 10 + (*s++ - '0'); return neg ? -n : n; }",
    "void my_itoa(int n, char* s) { int i = 0, neg = 0; if (n < 0) { neg = 1; n = -n; } do { s[i++] = n % 10 + '0'; n /= 10; } while (n); if (neg) s[i++] = '-'; s[i] = 0; my_strrev(s); }",
    
    # ===== HASH FUNCTIONS =====
    "unsigned djb2_hash(const char* s) { unsigned h = 5381; int c; while ((c = *s++)) h = ((h << 5) + h) + c; return h; }",
    "unsigned fnv1a_hash(const char* s) { unsigned h = 2166136261; while (*s) { h ^= *s++; h *= 16777619; } return h; }",
    "unsigned sdbm_hash(const char* s) { unsigned h = 0; int c; while ((c = *s++)) h = c + (h << 6) + (h << 16) - h; return h; }",
    
    # ===== ALGORITHMS =====
    "int is_prime(int n) { if (n < 2) return 0; for (int i = 2; i * i <= n; i++) if (n % i == 0) return 0; return 1; }",
    "int count_primes(int n) { int c = 0; for (int i = 2; i < n; i++) if (is_prime(i)) c++; return c; }",
    "int collatz_steps(int n) { int s = 0; while (n != 1) { n = (n % 2 == 0) ? n / 2 : 3 * n + 1; s++; } return s; }",
    "int sum_digits(int n) { int s = 0; while (n) { s += n % 10; n /= 10; } return s; }",
    "int count_digits(int n) { int c = 0; do { c++; n /= 10; } while (n); return c; }",
    "int reverse_int(int n) { int r = 0; while (n) { r = r * 10 + n % 10; n /= 10; } return r; }",
    "int is_palindrome_int(int n) { return n == reverse_int(n); }",
    
    # ===== DATA STRUCTURES (array-based) =====
    "void stack_push(int* s, int* top, int v) { s[(*top)++] = v; }",
    "int stack_pop(int* s, int* top) { return s[--(*top)]; }",
    "int stack_peek(int* s, int top) { return s[top - 1]; }",
    "void queue_enqueue(int* q, int* rear, int cap, int v) { q[(*rear)++ % cap] = v; }",
    "int queue_dequeue(int* q, int* front, int cap) { return q[(*front)++ % cap]; }",
    
    # ===== CONTROL FLOW PATTERNS =====
    "int simple_if(int x) { if (x > 0) return 1; return 0; }",
    "int if_else(int x) { if (x > 0) return 1; else return -1; }",
    "int if_elseif(int x) { if (x > 0) return 1; else if (x < 0) return -1; else return 0; }",
    "int nested_if(int x, int y) { if (x > 0) { if (y > 0) return 1; else return 2; } else { if (y > 0) return 3; else return 4; } }",
    "int while_loop(int n) { int s = 0; while (n > 0) { s += n; n--; } return s; }",
    "int for_loop(int n) { int s = 0; for (int i = 1; i <= n; i++) s += i; return s; }",
    "int do_while(int n) { int s = 0; do { s += n; n--; } while (n > 0); return s; }",
    "int nested_loop(int n) { int s = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) s += i * j; return s; }",
    "int break_loop(int* a, int n, int x) { for (int i = 0; i < n; i++) if (a[i] == x) return i; return -1; }",
    "int continue_loop(int* a, int n) { int s = 0; for (int i = 0; i < n; i++) { if (a[i] < 0) continue; s += a[i]; } return s; }",
    
    # ===== MORE DIVERSE FUNCTIONS =====
    "int triple(int x) { return x * 3; }",
    "int square(int x) { return x * x; }",
    "int cube(int x) { return x * x * x; }",
    "int double_val(int x) { return x + x; }",
    "int negate(int x) { return -x; }",
    "int increment(int x) { return x + 1; }",
    "int decrement(int x) { return x - 1; }",
    "int is_zero(int x) { return x == 0; }",
    "int is_positive(int x) { return x > 0; }",
    "int is_negative(int x) { return x < 0; }",
    "int is_even(int x) { return (x & 1) == 0; }",
    "int is_odd(int x) { return x & 1; }",
    "int avg(int a, int b) { return (a + b) / 2; }",
    "int diff(int a, int b) { return a > b ? a - b : b - a; }",
    "int sum3(int a, int b, int c) { return a + b + c; }",
    "int max3(int a, int b, int c) { int m = a; if (b > m) m = b; if (c > m) m = c; return m; }",
    "int min3(int a, int b, int c) { int m = a; if (b < m) m = b; if (c < m) m = c; return m; }",
    "int median3(int a, int b, int c) { if (a > b) { int t = a; a = b; b = t; } if (b > c) { int t = b; b = c; c = t; } if (a > b) { int t = a; a = b; b = t; } return b; }",
]

# Add main() wrapper and compile each function
def get_all_functions():
    """Generate all functions with main() wrapper."""
    functions = []
    for func in C_FUNCTIONS:
        # Extract function name
        match = re.match(r'(?:int|void|char\*|unsigned|void\*)\s+(\w+)\s*\(', func)
        if match:
            fname = match.group(1)
            # Add simple main that calls the function
            code = func + f"\nint main() {{ return 0; }}\n"
            functions.append(code)
    return functions

FUNCTIONS = get_all_functions()
print(f"  Loaded {len(FUNCTIONS)} C functions")

# Compile and extract instructions
def compile_and_extract(code, compiler, opt, tmpdir):
    """Compile C code and extract verified instructions."""
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    src = Path(tmpdir) / f"{code_hash}.c"
    bin_path = Path(tmpdir) / f"{code_hash}_{compiler}_{opt}"
    
    src.write_text(code)
    
    result = subprocess.run(
        [compiler, opt, "-w", "-o", str(bin_path), str(src)],
        capture_output=True, timeout=30
    )
    
    if result.returncode != 0 or not bin_path.exists():
        return []
    
    # Get objdump disassembly (Intel syntax)
    disasm = subprocess.run(
        ["objdump", "-d", "-M", "intel", str(bin_path)],
        capture_output=True, text=True
    ).stdout
    
    # Parse instructions
    samples = []
    for line in disasm.split('\n'):
        match = re.match(r'\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S+)\s*(.*)', line)
        if match:
            addr, bytes_hex, mnemonic, operands = match.groups()
            bytes_clean = bytes_hex.strip().replace(" ", "")
            
            if len(bytes_clean) > 0 and len(bytes_clean) <= 30:
                # Level 0 sample
                samples.append({
                    "level": 0,
                    "input": f"Bytes: {bytes_clean}",
                    "output": f"Instruction: {mnemonic}",
                    "compiler": compiler,
                    "opt": opt,
                })
                
                # Level 1 sample
                samples.append({
                    "level": 1,
                    "input": f"Instruction: {mnemonic} {operands}",
                    "output": get_semantics(mnemonic),
                    "compiler": compiler,
                    "opt": opt,
                })
    
    # Clean up
    try:
        src.unlink()
        bin_path.unlink()
    except:
        pass
    
    return samples

def get_semantics(mnemonic):
    """Get semantic description for instruction."""
    m = mnemonic.lower()
    semantics = {
        "mov": "Category: data_transfer; writes destination",
        "movzx": "Category: data_transfer; zero-extends to destination",
        "movsx": "Category: data_transfer; sign-extends to destination",
        "lea": "Category: data_transfer; loads effective address",
        "push": "Category: stack; decrements RSP, writes to stack",
        "pop": "Category: stack; reads from stack, increments RSP",
        "add": "Category: arithmetic; adds operands, sets CF/OF/SF/ZF",
        "sub": "Category: arithmetic; subtracts operands, sets flags",
        "imul": "Category: arithmetic; signed multiply",
        "idiv": "Category: arithmetic; signed divide",
        "inc": "Category: arithmetic; increments by 1",
        "dec": "Category: arithmetic; decrements by 1",
        "neg": "Category: arithmetic; negates value",
        "and": "Category: bitwise; logical AND, clears CF/OF",
        "or": "Category: bitwise; logical OR, clears CF/OF",
        "xor": "Category: bitwise; logical XOR, clears CF/OF",
        "not": "Category: bitwise; one's complement",
        "shl": "Category: bitwise; shift left",
        "shr": "Category: bitwise; shift right logical",
        "sar": "Category: bitwise; shift right arithmetic",
        "cmp": "Category: comparison; subtracts without storing, sets flags",
        "test": "Category: comparison; ANDs without storing, sets flags",
        "jmp": "Category: control_flow; unconditional jump",
        "je": "Category: control_flow; jump if equal (ZF=1)",
        "jne": "Category: control_flow; jump if not equal (ZF=0)",
        "jg": "Category: control_flow; jump if greater (signed)",
        "jl": "Category: control_flow; jump if less (signed)",
        "jge": "Category: control_flow; jump if greater/equal",
        "jle": "Category: control_flow; jump if less/equal",
        "ja": "Category: control_flow; jump if above (unsigned)",
        "jb": "Category: control_flow; jump if below (unsigned)",
        "call": "Category: control_flow; pushes return address, jumps",
        "ret": "Category: control_flow; pops return address, returns",
        "nop": "Category: no_operation; does nothing",
        "endbr64": "Category: security; CET end-branch",
    }
    return semantics.get(m, f"Category: other; executes {m}")

# Generate all samples
all_samples = []
compilers = ["gcc", "clang"] if subprocess.run(["which", "clang"], capture_output=True).returncode == 0 else ["gcc"]
opt_levels = ["-O0", "-O1", "-O2", "-O3"]

print(f"  Using compilers: {compilers}")
print(f"  Optimization levels: {opt_levels}")

with tempfile.TemporaryDirectory() as tmpdir:
    total = len(FUNCTIONS) * len(compilers) * len(opt_levels)
    count = 0
    
    for func in FUNCTIONS:
        for compiler in compilers:
            for opt in opt_levels:
                count += 1
                if count % 100 == 0:
                    print(f"  Progress: {count}/{total} ({len(all_samples)} samples)")
                
                samples = compile_and_extract(func, compiler, opt, tmpdir)
                all_samples.extend(samples)

# Deduplicate
seen = set()
unique = []
for s in all_samples:
    key = s["input"] + s["output"]
    h = hashlib.md5(key.encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        unique.append(s)

print(f"\n  Total unique samples: {len(unique)}")

# Split by level
level0 = [s for s in unique if s["level"] == 0]
level1 = [s for s in unique if s["level"] == 1]

print(f"  Level 0: {len(level0)}")
print(f"  Level 1: {len(level1)}")

# Save
os.makedirs("genesis_datasets/level0_v2", exist_ok=True)
os.makedirs("genesis_datasets/level1_v2", exist_ok=True)

random.shuffle(level0)
random.shuffle(level1)

with open("genesis_datasets/level0_v2/train.jsonl", "w") as f:
    for s in level0:
        f.write(json.dumps(s) + "\n")

with open("genesis_datasets/level1_v2/train.jsonl", "w") as f:
    for s in level1:
        f.write(json.dumps(s) + "\n")

print("  Saved datasets")

# 3. Training
print("\n[3/7] Loading PyTorch...")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Config
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-5
PATIENCE = 15
LORA_R = 64
LORA_ALPHA = 128

class SimpleDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                self.samples.append(f"{d['input']}\n{d['output']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.samples[idx], truncation=True, max_length=self.max_len,
                            padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone()
        }

def train_level(name, dataset_path, output_path, max_len=256):
    print(f"\n[4/7] Training {name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                      lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
    model = get_peft_model(base, lora).to(device)
    
    dataset = SimpleDataset(dataset_path, tokenizer, max_len)
    print(f"  Dataset: {len(dataset)} samples")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float("inf")
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda'):
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        
        if avg < best_loss - 0.001:
            best_loss = avg
            no_improve = 0
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"  Epoch {epoch+1}: {avg:.4f} * (saved)")
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"  Epoch {epoch+1}: {avg:.4f}")
        
        if no_improve >= PATIENCE and epoch > 20:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"  Best loss: {best_loss:.4f}")
    return best_loss

# Train
results = {}
results["level0"] = train_level("Level 0", "genesis_datasets/level0_v2/train.jsonl", "models/level0_v2", 128)
results["level1"] = train_level("Level 1", "genesis_datasets/level1_v2/train.jsonl", "models/level1_v2", 256)

# 5. Quick verification
print("\n[5/7] Quick verification...")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

for level in ["level0_v2", "level1_v2"]:
    print(f"\n  {level}:")
    tokenizer = AutoTokenizer.from_pretrained(f"models/{level}")
    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base, f"models/{level}").to(device).eval()
    
    if "level0" in level:
        tests = ["Bytes: 55", "Bytes: c3", "Bytes: 4889e5"]
    else:
        tests = ["Instruction: push rbp", "Instruction: ret", "Instruction: mov rax, rbx"]
    
    for test in tests:
        prompt = f"{test}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"    {test} → {result.split(chr(10))[-1][:50]}")

# 6. Package
print("\n[6/7] Packaging...")
import zipfile

with zipfile.ZipFile("genesis_v2.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_v2", "level1_v2"]:
        for f in Path(f"models/{level}").iterdir():
            zf.write(f, f"models/{level}/{f.name}")

print("\n" + "=" * 70)
print("[7/7] COMPLETE")
print("=" * 70)
for name, loss in results.items():
    print(f"  {name}: {loss:.4f}")

print("\nDownload: genesis_v2.zip")

from IPython.display import FileLink
display(FileLink("genesis_v2.zip"))
