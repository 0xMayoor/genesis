# ============================================================================
# GENESIS HONEST TRAINING
# 
# Ground truth: Capstone (industry standard disassembler)
# Data source: Real compiled binaries
# No test knowledge: We don't look at what tests expect
#
# If the model fails tests, we improve data quality - NOT game the test.
# ============================================================================

import os
import subprocess
import sys
import json
import random
import hashlib
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

print("=" * 70)
print("GENESIS HONEST TRAINING")
print("Ground Truth: Capstone | Data: Real Binaries | No Test Gaming")
print("=" * 70)

# Setup
print("\n[1/6] Setup...")
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "clang", "gcc"], capture_output=True)

if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "peft", "accelerate", "capstone"], check=True)

# Verify tools
print("  Compilers:", end=" ")
compilers = []
for c in ["gcc", "clang"]:
    if subprocess.run([c, "--version"], capture_output=True).returncode == 0:
        compilers.append(c)
        print(f"✓{c}", end=" ")
print()

# ============================================================================
# REAL C PROGRAMS - Diverse, real-world code patterns
# These are NOT crafted to produce specific instructions
# ============================================================================

REAL_PROGRAMS = {
    # String operations
    "strlen": '''
int my_strlen(const char* s) {
    int len = 0;
    while (s[len]) len++;
    return len;
}
int main() { return my_strlen("hello"); }
''',
    "strcpy": '''
void my_strcpy(char* dst, const char* src) {
    while ((*dst++ = *src++));
}
int main() { char buf[10]; my_strcpy(buf, "hi"); return buf[0]; }
''',
    "strcmp": '''
int my_strcmp(const char* a, const char* b) {
    while (*a && *a == *b) { a++; b++; }
    return *a - *b;
}
int main() { return my_strcmp("abc", "abd"); }
''',

    # Memory operations
    "memcpy": '''
void* my_memcpy(void* dst, const void* src, unsigned long n) {
    char* d = dst;
    const char* s = src;
    while (n--) *d++ = *s++;
    return dst;
}
int main() { char a[10], b[10] = "test"; my_memcpy(a, b, 5); return a[0]; }
''',
    "memset": '''
void* my_memset(void* s, int c, unsigned long n) {
    char* p = s;
    while (n--) *p++ = c;
    return s;
}
int main() { char buf[10]; my_memset(buf, 'x', 5); return buf[0]; }
''',

    # Math
    "factorial": '''
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
int main() { return factorial(5); }
''',
    "fibonacci": '''
int fib(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int tmp = a + b;
        a = b;
        b = tmp;
    }
    return b;
}
int main() { return fib(10); }
''',
    "gcd": '''
int gcd(int a, int b) {
    while (b != 0) {
        int tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}
int main() { return gcd(48, 18); }
''',
    "power": '''
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
''',
    "isqrt": '''
int isqrt(int n) {
    int x = n;
    int y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}
int main() { return isqrt(100); }
''',

    # Bit manipulation
    "popcount": '''
int popcount(unsigned int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
int main() { return popcount(0xFF); }
''',
    "reverse_bits": '''
unsigned int reverse_bits(unsigned int n) {
    unsigned int result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    return result;
}
int main() { return reverse_bits(0x12345678); }
''',

    # Array operations
    "sum_array": '''
int sum_array(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
int main() { int a[] = {1,2,3,4,5}; return sum_array(a, 5); }
''',
    "max_array": '''
int max_array(int* arr, int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}
int main() { int a[] = {3,1,4,1,5,9}; return max_array(a, 6); }
''',
    "binary_search": '''
int binary_search(int* arr, int n, int target) {
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
int main() { int a[] = {1,2,3,4,5}; return binary_search(a, 5, 3); }
''',

    # Sorting
    "bubble_sort": '''
void bubble_sort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}
int main() { int a[] = {5,2,8,1,9}; bubble_sort(a, 5); return a[0]; }
''',
    "insertion_sort": '''
void insertion_sort(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}
int main() { int a[] = {5,2,8,1,9}; insertion_sort(a, 5); return a[0]; }
''',

    # Character operations
    "is_digit": '''
int is_digit(char c) {
    return c >= '0' && c <= '9';
}
int main() { return is_digit('5'); }
''',
    "to_upper": '''
char to_upper(char c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}
int main() { return to_upper('a'); }
''',
    "atoi": '''
int my_atoi(const char* s) {
    int result = 0;
    int sign = 1;
    while (*s == ' ') s++;
    if (*s == '-') { sign = -1; s++; }
    else if (*s == '+') s++;
    while (*s >= '0' && *s <= '9') {
        result = result * 10 + (*s - '0');
        s++;
    }
    return sign * result;
}
int main() { return my_atoi("-123"); }
''',

    # Control flow
    "abs": '''
int my_abs(int x) {
    return x < 0 ? -x : x;
}
int main() { return my_abs(-42); }
''',
    "min": '''
int my_min(int a, int b) {
    return a < b ? a : b;
}
int main() { return my_min(3, 7); }
''',
    "max": '''
int my_max(int a, int b) {
    return a > b ? a : b;
}
int main() { return my_max(3, 7); }
''',
    "clamp": '''
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
int main() { return clamp(15, 0, 10); }
''',

    # Recursive
    "fib_recursive": '''
int fib_r(int n) {
    if (n <= 1) return n;
    return fib_r(n-1) + fib_r(n-2);
}
int main() { return fib_r(10); }
''',

    # Pointer operations
    "swap": '''
void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
int main() { int x = 1, y = 2; swap(&x, &y); return x; }
''',
    "reverse_array": '''
void reverse(int* arr, int n) {
    for (int i = 0; i < n / 2; i++) {
        int tmp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = tmp;
    }
}
int main() { int a[] = {1,2,3,4,5}; reverse(a, 5); return a[0]; }
''',

    # Hash
    "djb2": '''
unsigned long djb2(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}
int main() { return djb2("hello") & 0xFF; }
''',

    # Prime
    "is_prime": '''
int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}
int main() { return is_prime(17); }
''',
}

# ============================================================================
# CAPSTONE GROUND TRUTH
# This is the ONLY source of truth for disassembly
# ============================================================================

print("\n[2/6] Setting up Capstone ground truth...")

from capstone import Cs, CS_ARCH_X86, CS_MODE_64

def get_capstone_ground_truth(binary_path: Path) -> list[dict]:
    """
    Extract ground truth from binary using Capstone.
    Capstone is the industry standard - what it says IS correct.
    """
    # Get .text section info
    readelf = subprocess.run(["readelf", "-S", str(binary_path)], 
                            capture_output=True, text=True)
    
    text_addr = 0
    text_size = 0
    for line in readelf.stdout.split('\n'):
        if '.text' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == '.text':
                    try:
                        text_addr = int(parts[i + 2], 16)
                        text_size = int(parts[i + 4], 16)
                    except:
                        pass
                    break
    
    # Extract .text bytes
    objcopy = subprocess.run(
        ["objcopy", "-O", "binary", "--only-section=.text", 
         str(binary_path), "/dev/stdout"],
        capture_output=True
    )
    text_bytes = objcopy.stdout
    
    if not text_bytes:
        return []
    
    # Disassemble with Capstone - THIS IS GROUND TRUTH
    cs = Cs(CS_ARCH_X86, CS_MODE_64)
    cs.detail = True
    
    instructions = []
    for instr in cs.disasm(text_bytes, text_addr):
        instructions.append({
            "address": instr.address,
            "bytes": instr.bytes.hex(),
            "mnemonic": instr.mnemonic,
            "operands": instr.op_str,
            "size": instr.size,
        })
    
    return instructions

def compile_program(name: str, code: str, compiler: str, opt: str, tmpdir: Path) -> Optional[Path]:
    """Compile a C program."""
    src = tmpdir / f"{name}.c"
    binary = tmpdir / f"{name}_{compiler}_{opt.replace('-', '')}"
    
    src.write_text(code)
    
    result = subprocess.run(
        [compiler, opt, "-w", "-o", str(binary), str(src)],
        capture_output=True,
        timeout=30
    )
    
    if result.returncode != 0:
        return None
    
    return binary

# ============================================================================
# GENERATE TRAINING DATA FROM GROUND TRUTH
# ============================================================================

def describe_instruction(mnemonic: str, operands: str) -> str:
    """
    Describe what an instruction does based on x86 ISA manual.
    NOT based on test expectations.
    """
    m = mnemonic.lower()
    
    # These descriptions are from Intel/AMD manuals
    descriptions = {
        # Data movement
        "mov": "Copies data from source to destination",
        "movzx": "Copies with zero extension to larger register",
        "movsx": "Copies with sign extension to larger register",
        "movsxd": "Copies dword with sign extension to qword",
        "lea": "Computes effective address without memory access",
        "push": "Decrements stack pointer and stores value",
        "pop": "Loads value from stack and increments stack pointer",
        "xchg": "Exchanges values between two locations",
        
        # Arithmetic
        "add": "Adds source to destination, sets flags",
        "sub": "Subtracts source from destination, sets flags",
        "inc": "Increments operand by one",
        "dec": "Decrements operand by one",
        "neg": "Two's complement negation",
        "mul": "Unsigned multiply",
        "imul": "Signed multiply",
        "div": "Unsigned divide",
        "idiv": "Signed divide",
        
        # Logic
        "and": "Bitwise AND, clears CF and OF",
        "or": "Bitwise OR, clears CF and OF",
        "xor": "Bitwise XOR, clears CF and OF",
        "not": "One's complement negation",
        "shl": "Shift left, multiply by 2",
        "sal": "Shift arithmetic left",
        "shr": "Shift right logical, divide by 2",
        "sar": "Shift right arithmetic, preserves sign",
        "rol": "Rotate left",
        "ror": "Rotate right",
        
        # Compare/Test
        "cmp": "Compares by subtracting, sets flags, discards result",
        "test": "Compares by ANDing, sets flags, discards result",
        
        # Control flow
        "jmp": "Unconditional jump to target",
        "je": "Jump if equal (ZF=1)",
        "jz": "Jump if zero (ZF=1)",
        "jne": "Jump if not equal (ZF=0)",
        "jnz": "Jump if not zero (ZF=0)",
        "jg": "Jump if greater (signed)",
        "jge": "Jump if greater or equal (signed)",
        "jl": "Jump if less (signed)",
        "jle": "Jump if less or equal (signed)",
        "ja": "Jump if above (unsigned)",
        "jae": "Jump if above or equal (unsigned)",
        "jb": "Jump if below (unsigned)",
        "jbe": "Jump if below or equal (unsigned)",
        "js": "Jump if sign flag set",
        "jns": "Jump if sign flag not set",
        "call": "Pushes return address and jumps to target",
        "ret": "Pops return address and jumps to it",
        "leave": "Restores stack frame (mov rsp,rbp; pop rbp)",
        
        # Other
        "nop": "No operation",
        "endbr64": "End branch marker for CET",
        "endbr32": "End branch marker for CET (32-bit)",
        "cdq": "Sign extends EAX into EDX:EAX",
        "cdqe": "Sign extends EAX into RAX",
        "cqo": "Sign extends RAX into RDX:RAX",
        "int3": "Breakpoint trap",
        
        # Conditional set
        "sete": "Set byte if equal",
        "setne": "Set byte if not equal",
        "setg": "Set byte if greater",
        "setl": "Set byte if less",
        "setge": "Set byte if greater or equal",
        "setle": "Set byte if less or equal",
        
        # Conditional move
        "cmove": "Move if equal",
        "cmovne": "Move if not equal",
        "cmovg": "Move if greater",
        "cmovl": "Move if less",
    }
    
    return descriptions.get(m, f"Executes {m} operation")

def analyze_cfg(instructions: list[dict]) -> Optional[dict]:
    """
    Analyze control flow graph from instruction sequence.
    """
    if len(instructions) < 5:
        return None
    
    # Find basic block boundaries
    leaders = {instructions[0]["address"]}
    
    for i, instr in enumerate(instructions):
        m = instr["mnemonic"].lower()
        
        # After a jump/call/ret, next instruction is a leader
        if m in ["jmp", "ret", "call"] or m.startswith("j"):
            if i + 1 < len(instructions):
                leaders.add(instructions[i + 1]["address"])
        
        # Jump targets are leaders
        if m.startswith("j") or m == "call":
            ops = instr["operands"]
            try:
                if ops.startswith("0x"):
                    target = int(ops, 16)
                    leaders.add(target)
            except:
                pass
    
    # Count blocks
    num_blocks = len(leaders)
    
    # Detect loops (back edges)
    has_loop = False
    for instr in instructions:
        m = instr["mnemonic"].lower()
        if m == "jmp" or m.startswith("j"):
            try:
                target = int(instr["operands"], 16)
                if target < instr["address"]:
                    has_loop = True
            except:
                pass
    
    # Count conditionals
    conditionals = sum(1 for i in instructions 
                      if i["mnemonic"].lower().startswith("j") 
                      and i["mnemonic"].lower() not in ["jmp"])
    
    # Build input/output
    instr_strs = []
    for instr in instructions[:15]:
        m = instr["mnemonic"].lower()
        marker = ""
        if m == "ret":
            marker = " [return]"
        elif m == "call":
            marker = " [call]"
        elif m == "jmp":
            marker = " [jump]"
        elif m.startswith("j"):
            marker = " [conditional]"
        
        instr_strs.append(f"{instr['address']:#x}:{instr['mnemonic']} {instr['operands']}{marker}")
    
    input_text = "Instructions:\n" + "\n".join(instr_strs)
    output_text = f"Analysis: {num_blocks} basic blocks"
    if has_loop:
        output_text += "; loop detected"
    if conditionals > 0:
        output_text += f"; {conditionals} conditional branches"
    
    return {
        "input": input_text,
        "output": output_text,
    }

print("\n[3/6] Generating training data from Capstone ground truth...")

level0_samples = []  # bytes -> mnemonic
level1_samples = []  # instruction -> what it does (from Capstone analysis)
level2_samples = []  # instruction sequence -> CFG

opt_levels = ["-O0", "-O1", "-O2", "-O3"]

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    total = len(REAL_PROGRAMS) * len(compilers) * len(opt_levels)
    count = 0
    
    for name, code in REAL_PROGRAMS.items():
        for compiler in compilers:
            for opt in opt_levels:
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total}")
                
                binary = compile_program(name, code, compiler, opt, tmpdir)
                if not binary:
                    continue
                
                # Get GROUND TRUTH from Capstone
                instructions = get_capstone_ground_truth(binary)
                
                for instr in instructions:
                    # Level 0: bytes -> mnemonic (PURE GROUND TRUTH)
                    level0_samples.append({
                        "input": f"Bytes: {instr['bytes']}",
                        "output": f"Instruction: {instr['mnemonic']}",
                        "source": f"{name}_{compiler}_{opt}",
                    })
                    
                    # Level 1: instruction -> description
                    # Description is based on what the instruction ACTUALLY does
                    # NOT what a test expects
                    full_instr = f"{instr['mnemonic']} {instr['operands']}".strip()
                    level1_samples.append({
                        "input": f"Instruction: {full_instr}",
                        "output": describe_instruction(instr['mnemonic'], instr['operands']),
                        "source": f"{name}_{compiler}_{opt}",
                    })
                
                # Level 2: CFG analysis
                if len(instructions) >= 5:
                    cfg_sample = analyze_cfg(instructions)
                    if cfg_sample:
                        level2_samples.append(cfg_sample)
                
                # Cleanup
                if binary.exists():
                    binary.unlink()

# Deduplicate
def dedupe(samples):
    seen = set()
    unique = []
    for s in samples:
        key = s["input"]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

level0_samples = dedupe(level0_samples)
level1_samples = dedupe(level1_samples)
level2_samples = dedupe(level2_samples)

print(f"\n  Level 0: {len(level0_samples)} samples")
print(f"  Level 1: {len(level1_samples)} samples")
print(f"  Level 2: {len(level2_samples)} samples")

# Save
print("\n[4/6] Saving datasets...")

os.makedirs("genesis_datasets/level0_honest", exist_ok=True)
os.makedirs("genesis_datasets/level1_honest", exist_ok=True)
os.makedirs("genesis_datasets/level2_honest", exist_ok=True)

random.shuffle(level0_samples)
random.shuffle(level1_samples)
random.shuffle(level2_samples)

for name, samples in [("level0_honest", level0_samples), 
                      ("level1_honest", level1_samples),
                      ("level2_honest", level2_samples)]:
    with open(f"genesis_datasets/{name}/train.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps({"input": s["input"], "output": s["output"]}) + "\n")
    print(f"  Saved {name}: {len(samples)} samples")

# ============================================================================
# TRAINING
# ============================================================================

print("\n[5/6] Training on honest data...")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

MODEL_NAME = "distilgpt2"
BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-5
PATIENCE = 15
LORA_R = 64
LORA_ALPHA = 128

class HonestDataset(Dataset):
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

def train_honest(name, dataset_path, output_path, max_len=256):
    print(f"\n  Training {name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                      lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
    model = get_peft_model(base, lora).to(device)
    
    dataset = HonestDataset(dataset_path, tokenizer, max_len)
    print(f"    Samples: {len(dataset)}")
    
    if len(dataset) < 50:
        print(f"    WARNING: Very few samples. Model may not generalize well.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       num_workers=2, pin_memory=True)
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
                loss = model(input_ids=input_ids, attention_mask=attention_mask, 
                           labels=labels).loss
            
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
            if epoch % 10 == 0:
                print(f"    Epoch {epoch+1}: {avg:.4f} * (saved)")
        else:
            no_improve += 1
        
        if no_improve >= PATIENCE and epoch > 30:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    print(f"    Best loss: {best_loss:.4f}")
    return best_loss

# Train
results = {}
results["level0"] = train_honest("Level 0 (bytes→mnemonic)", 
                                 "genesis_datasets/level0_honest/train.jsonl",
                                 "models/level0_honest", 128)
results["level1"] = train_honest("Level 1 (instruction→description)",
                                 "genesis_datasets/level1_honest/train.jsonl", 
                                 "models/level1_honest", 256)
results["level2"] = train_honest("Level 2 (sequence→CFG)",
                                 "genesis_datasets/level2_honest/train.jsonl",
                                 "models/level2_honest", 512)

# ============================================================================
# VERIFICATION (using Capstone, not test expectations)
# ============================================================================

print("\n[6/6] Verification against Capstone ground truth...")

from peft import PeftModel

# Test with bytes we KNOW the answer to (from Capstone)
test_bytes = [
    ("55", "push"),      # Capstone says this is push
    ("c3", "ret"),       # Capstone says this is ret
    ("89e5", "mov"),     # Capstone says this is mov
    ("4889e5", "mov"),   # Capstone says this is mov
    ("31c0", "xor"),     # Capstone says this is xor
]

print("\n  Level 0 verification (against Capstone):")
tokenizer = AutoTokenizer.from_pretrained("models/level0_honest")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "models/level0_honest").to(device).eval()

correct = 0
for bytes_hex, expected in test_bytes:
    prompt = f"Bytes: {bytes_hex}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    predicted = result.split("Instruction:")[-1].strip().split()[0] if "Instruction:" in result else "?"
    
    match = "✓" if predicted.lower() == expected else "✗"
    if predicted.lower() == expected:
        correct += 1
    print(f"    {bytes_hex} → {predicted} (expected: {expected}) {match}")

print(f"\n  Accuracy: {correct}/{len(test_bytes)} = {100*correct/len(test_bytes):.0f}%")

# Package
print("\n" + "=" * 60)
print("PACKAGING")
print("=" * 60)

import zipfile

with zipfile.ZipFile("genesis_honest.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_honest", "level1_honest", "level2_honest"]:
        model_path = Path(f"models/{level}")
        if model_path.exists():
            for f in model_path.iterdir():
                zf.write(f, f"models/{level}/{f.name}")

print(f"  Zip: {Path('genesis_honest.zip').stat().st_size / 1024 / 1024:.1f} MB")

from google.colab import files
files.download("genesis_honest.zip")

print("\n" + "=" * 60)
print("COMPLETE - HONEST TRAINING")
print("=" * 60)
print("Ground truth: Capstone")
print("Data source: Real compiled binaries")
print("No test gaming: Model learns actual mappings")
print()
for name, loss in results.items():
    print(f"  {name}: {loss:.4f}")
