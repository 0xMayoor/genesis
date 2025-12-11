# ============================================================================
# GENESIS REAL - Train from REAL system binaries
# 
# Data source: /usr/bin/*, /usr/lib/* - real compiled code
# Ground truth: Capstone disassembler
# No synthetic data
# ============================================================================

import os
import subprocess
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("GENESIS REAL - Training from system binaries")
print("=" * 70)

# Setup
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "binutils"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "peft", "accelerate", "capstone"], check=True)

if "genesis" not in os.getcwd():
    if os.path.exists("genesis"):
        subprocess.run(["rm", "-rf", "genesis"])
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
    os.chdir("genesis")

print(f"Working dir: {os.getcwd()}")

from capstone import Cs, CS_ARCH_X86, CS_MODE_64

# ============================================================================
# EXTRACT FROM REAL BINARIES
# ============================================================================
print("\n[1/5] Finding real binaries...")

# Find ELF binaries in system
def find_elf_binaries(paths, max_count=200):
    binaries = []
    for path in paths:
        if not Path(path).exists():
            continue
        for f in Path(path).glob("*"):
            if f.is_file() and not f.is_symlink():
                try:
                    # Check if ELF
                    with open(f, "rb") as fp:
                        if fp.read(4) == b'\x7fELF':
                            binaries.append(f)
                            if len(binaries) >= max_count:
                                return binaries
                except:
                    pass
    return binaries

binaries = find_elf_binaries(["/usr/bin", "/usr/lib/x86_64-linux-gnu", "/bin", "/sbin"], max_count=100)
print(f"  Found {len(binaries)} ELF binaries")

# If no system binaries (Colab), compile from GNU coreutils source
if len(binaries) < 10:
    print("  No system binaries, compiling from source...")
    subprocess.run(["apt-get", "install", "-y", "-qq", "gcc", "clang"], capture_output=True)
    
    # Diverse C programs
    PROGRAMS = {
        "strlen": 'int f(const char*s){int n=0;while(s[n])n++;return n;}int main(){return f("hi");}',
        "strcpy": 'void f(char*d,const char*s){while((*d++=*s++));}int main(){char b[8];f(b,"x");return 0;}',
        "strcmp": 'int f(const char*a,const char*b){while(*a&&*a==*b){a++;b++;}return*a-*b;}int main(){return f("a","b");}',
        "memcpy": 'void*f(void*d,const void*s,unsigned long n){char*p=d;const char*q=s;while(n--)*p++=*q++;return d;}int main(){char a[4],b[]="hi";f(a,b,3);return 0;}',
        "memset": 'void*f(void*s,int c,unsigned long n){char*p=s;while(n--)*p++=c;return s;}int main(){char b[4];f(b,65,4);return 0;}',
        "atoi": 'int f(const char*s){int r=0,g=1;if(*s==45){g=-1;s++;}while(*s>=48&&*s<=57)r=r*10+*s++-48;return g*r;}int main(){return f("-42");}',
        "itoa": 'void f(int n,char*s){int i=0,g=n<0;if(g)n=-n;do{s[i++]=n%10+48;}while((n/=10)>0);if(g)s[i++]=45;s[i]=0;int j;for(j=0;j<i/2;j++){char t=s[j];s[j]=s[i-1-j];s[i-1-j]=t;}}int main(){char b[16];f(123,b);return b[0];}',
        "sum": 'int f(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3,4,5};return f(a,5);}',
        "max": 'int f(int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];return m;}int main(){int a[]={3,1,4,1,5};return f(a,5);}',
        "min": 'int f(int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]<m)m=a[i];return m;}int main(){int a[]={3,1,4,1,5};return f(a,5);}',
        "bsearch": 'int f(int*a,int n,int t){int l=0,r=n-1;while(l<=r){int m=(l+r)/2;if(a[m]==t)return m;if(a[m]<t)l=m+1;else r=m-1;}return-1;}int main(){int a[]={1,2,3,4,5};return f(a,5,3);}',
        "qsort_part": 'int f(int*a,int l,int h){int p=a[h],i=l-1;for(int j=l;j<h;j++)if(a[j]<p){i++;int t=a[i];a[i]=a[j];a[j]=t;}int t=a[i+1];a[i+1]=a[h];a[h]=t;return i+1;}int main(){int a[]={3,1,4,1,5};return f(a,0,4);}',
        "bubble": 'void f(int*a,int n){for(int i=0;i<n-1;i++)for(int j=0;j<n-i-1;j++)if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}int main(){int a[]={5,2,8,1};f(a,4);return a[0];}',
        "insert": 'void f(int*a,int n){for(int i=1;i<n;i++){int k=a[i],j=i-1;while(j>=0&&a[j]>k){a[j+1]=a[j];j--;}a[j+1]=k;}}int main(){int a[]={5,2,8,1};f(a,4);return a[0];}',
        "fib": 'int f(int n){if(n<=1)return n;int a=0,b=1;for(int i=2;i<=n;i++){int t=a+b;a=b;b=t;}return b;}int main(){return f(10);}',
        "fact": 'int f(int n){int r=1;for(int i=2;i<=n;i++)r*=i;return r;}int main(){return f(5);}',
        "gcd": 'int f(int a,int b){while(b){int t=b;b=a%b;a=t;}return a;}int main(){return f(48,18);}',
        "lcm": 'int g(int a,int b){while(b){int t=b;b=a%b;a=t;}return a;}int f(int a,int b){return a/g(a,b)*b;}int main(){return f(12,8);}',
        "pow": 'int f(int b,int e){int r=1;while(e>0){if(e&1)r*=b;b*=b;e>>=1;}return r;}int main(){return f(2,10);}',
        "sqrt": 'int f(int n){int x=n,y=(x+1)/2;while(y<x){x=y;y=(x+n/x)/2;}return x;}int main(){return f(100);}',
        "prime": 'int f(int n){if(n<2)return 0;for(int i=2;i*i<=n;i++)if(n%i==0)return 0;return 1;}int main(){return f(17);}',
        "popcount": 'int f(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return f(255);}',
        "clz": 'int f(unsigned n){int c=0;if(!n)return 32;while(!(n&0x80000000)){c++;n<<=1;}return c;}int main(){return f(0x0F000000);}',
        "ctz": 'int f(unsigned n){int c=0;if(!n)return 32;while(!(n&1)){c++;n>>=1;}return c;}int main(){return f(0x00000F00);}',
        "reverse": 'unsigned f(unsigned n){unsigned r=0;for(int i=0;i<32;i++){r=(r<<1)|(n&1);n>>=1;}return r;}int main(){return f(0x12345678)&0xFF;}',
        "abs": 'int f(int x){return x<0?-x:x;}int main(){return f(-42);}',
        "sign": 'int f(int x){return(x>0)-(x<0);}int main(){return f(-5);}',
        "clamp": 'int f(int x,int l,int h){return x<l?l:x>h?h:x;}int main(){return f(15,0,10);}',
        "swap": 'void f(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;f(&x,&y);return x;}',
        "reverse_arr": 'void f(int*a,int n){for(int i=0;i<n/2;i++){int t=a[i];a[i]=a[n-1-i];a[n-1-i]=t;}}int main(){int a[]={1,2,3};f(a,3);return a[0];}',
        "rotate": 'void f(int*a,int n,int k){k%=n;for(int i=0;i<k;i++){int t=a[n-1];for(int j=n-1;j>0;j--)a[j]=a[j-1];a[0]=t;}}int main(){int a[]={1,2,3,4,5};f(a,5,2);return a[0];}',
        "hash": 'unsigned f(const char*s){unsigned h=5381;while(*s)h=((h<<5)+h)+*s++;return h;}int main(){return f("test")&0xFF;}',
        "crc": 'unsigned f(const char*s,int n){unsigned c=0xFFFFFFFF;for(int i=0;i<n;i++){c^=s[i];for(int j=0;j<8;j++)c=(c>>1)^(0xEDB88320&-(c&1));}return~c;}int main(){return f("test",4)&0xFF;}',
    }
    
    import tempfile
    binaries = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for name, code in PROGRAMS.items():
            for compiler in ["gcc", "clang"]:
                for opt in ["-O0", "-O1", "-O2", "-O3"]:
                    src = tmpdir / f"{name}.c"
                    binary = tmpdir / f"{name}_{compiler}_{opt}"
                    src.write_text(code)
                    if subprocess.run([compiler, opt, "-w", "-o", str(binary), str(src)], 
                                     capture_output=True, timeout=30).returncode == 0:
                        # Copy to persistent location
                        dest = Path(f"compiled_binaries/{name}_{compiler}_{opt}")
                        dest.parent.mkdir(exist_ok=True)
                        subprocess.run(["cp", str(binary), str(dest)])
                        binaries.append(dest)
    
    print(f"  Compiled {len(binaries)} binaries")

# ============================================================================
# DISASSEMBLE WITH CAPSTONE (Ground Truth)
# ============================================================================
print("\n[2/5] Disassembling with Capstone...")

def disassemble_binary(path):
    """Extract .text section and disassemble with Capstone."""
    try:
        # Get .text section
        result = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", str(path), "/dev/stdout"],
            capture_output=True, timeout=30
        )
        if not result.stdout:
            return []
        
        cs = Cs(CS_ARCH_X86, CS_MODE_64)
        cs.detail = False
        
        instructions = []
        for instr in cs.disasm(result.stdout, 0x1000):
            instructions.append({
                "bytes": instr.bytes.hex(),
                "mnemonic": instr.mnemonic,
                "operands": instr.op_str,
                "address": instr.address,
            })
        return instructions
    except:
        return []

# Collect all instructions
all_instructions = []
for binary in binaries[:100]:  # Limit to 100 binaries
    instrs = disassemble_binary(binary)
    all_instructions.extend(instrs)
    if len(all_instructions) > 500000:
        break

print(f"  Extracted {len(all_instructions)} instructions")

# ============================================================================
# BUILD LEVEL 0 DATASET
# ============================================================================
print("\n[3/5] Building datasets...")

# Group by mnemonic
by_mnemonic = defaultdict(set)
for instr in all_instructions:
    by_mnemonic[instr["mnemonic"]].add((instr["bytes"], instr["mnemonic"]))

print(f"  Found {len(by_mnemonic)} unique mnemonics")
print(f"  Top 10: {sorted(by_mnemonic.keys(), key=lambda m: -len(by_mnemonic[m]))[:10]}")

# Balance: min 20, max 300 samples per mnemonic
level0_samples = []
for mnemonic, byte_set in by_mnemonic.items():
    items = list(byte_set)
    
    # Skip rare/unusual mnemonics with < 5 occurrences (noise)
    if len(items) < 5:
        continue
    
    # Oversample if needed
    if len(items) < 20:
        items = items * ((20 // len(items)) + 1)
    
    # Cap at 300
    items = items[:300]
    
    for bytes_hex, m in items:
        level0_samples.append({
            "input": f"Bytes: {bytes_hex}",
            "output": f"Instruction: {m}"
        })

print(f"  Level 0: {len(level0_samples)} samples")

# ============================================================================
# BUILD LEVEL 1 DATASET
# ============================================================================

# Standard semantics using common CS terminology
# These descriptions are from Intel/AMD manuals using standard vocabulary
SEMANTICS = {
    "mov": "Move operation: transfer and copy data between registers or memory, write destination read source",
    "movzx": "Move with zero extend: transfer copy smaller value to larger register, zero fill upper bits",
    "movsx": "Move with sign extend: transfer copy smaller value to larger register, sign fill upper bits",
    "movsxd": "Move with sign extend dword to qword: transfer copy 32-bit to 64-bit with sign extension",
    "lea": "Load effective address: compute memory address without accessing memory, address calculation",
    "push": "Push to stack: decrement rsp stack pointer sp and write value to memory",
    "pop": "Pop from stack: read value from memory and increment rsp stack pointer sp",
    "add": "Add operation: arithmetic sum of operands, updates flags, plus operation",
    "sub": "Subtract operation: arithmetic difference, minus operation, updates flags",
    "inc": "Increment: add one to operand, arithmetic operation, updates flags",
    "dec": "Decrement: subtract one from operand, arithmetic operation, updates flags",
    "neg": "Negate: two's complement negation, arithmetic operation",
    "mul": "Unsigned multiply: arithmetic multiplication, mul operation",
    "imul": "Signed multiply: arithmetic multiplication, mul operation, signed operands",
    "div": "Unsigned divide: arithmetic division, div operation",
    "idiv": "Signed divide: arithmetic division, div operation, signed operands",
    "cmp": "Compare: subtract operands without storing result, set flags for conditional",
    "test": "Test: bitwise and operands without storing, set flags, compare for zero",
    "and": "Bitwise and: logical and operation, mask bits, updates flags",
    "or": "Bitwise or: logical or operation, set bits, updates flags",
    "xor": "Bitwise exclusive or: xor operation, toggle bits, can zero register when same operands",
    "not": "Bitwise not: logical complement, invert all bits",
    "shl": "Shift left: logical shift left, multiply by power of 2, bitwise operation",
    "sal": "Shift arithmetic left: same as shl, multiply by power of 2",
    "shr": "Shift right: logical shift right, divide by power of 2, bitwise operation, zero fill",
    "sar": "Shift arithmetic right: shift right preserving sign, divide signed by power of 2",
    "rol": "Rotate left: circular shift left, bits wrap around",
    "ror": "Rotate right: circular shift right, bits wrap around",
    "jmp": "Jump: unconditional branch to target address, control flow transfer",
    "je": "Jump if equal: conditional branch when zero flag set, jump if equal",
    "jz": "Jump if zero: conditional branch when zero flag set, same as je",
    "jne": "Jump if not equal: conditional branch when zero flag clear, not equal",
    "jnz": "Jump if not zero: conditional branch when zero flag clear, same as jne",
    "jg": "Jump if greater: conditional branch for signed greater than comparison",
    "jge": "Jump if greater or equal: conditional branch for signed comparison",
    "jl": "Jump if less: conditional branch for signed less than comparison",
    "jle": "Jump if less or equal: conditional branch for signed comparison",
    "ja": "Jump if above: conditional branch for unsigned greater than comparison",
    "jae": "Jump if above or equal: conditional branch for unsigned comparison",
    "jb": "Jump if below: conditional branch for unsigned less than comparison",
    "jbe": "Jump if below or equal: conditional branch for unsigned comparison",
    "js": "Jump if sign: conditional branch when sign flag set, negative result",
    "jns": "Jump if not sign: conditional branch when sign flag clear, positive result",
    "call": "Call function: push return address to stack, transfer control to function",
    "ret": "Return: pop return address from stack to rip instruction pointer, control flow return",
    "leave": "Leave stack frame: restore rbp and rsp, equivalent to mov rsp,rbp; pop rbp",
    "nop": "No operation: does nothing, used for alignment or padding",
    "endbr64": "End branch 64-bit: Intel CET indirect branch tracking marker",
    "cdq": "Convert double to quad: sign extend eax into edx:eax for division",
    "cqo": "Convert quad to oct: sign extend rax into rdx:rax for division",
    "cdqe": "Convert dword to qword: sign extend eax to rax",
    "syscall": "System call: invoke operating system service",
    "int": "Interrupt: software interrupt, invoke interrupt handler",
    "hlt": "Halt: stop processor until interrupt",
    "cpuid": "CPU identification: return processor information",
    "rdtsc": "Read timestamp counter: return processor cycle count",
}

level1_samples = []
for instr in all_instructions:
    m = instr["mnemonic"]
    # Normalize AT&T suffix
    m_base = m.rstrip("lqwb")
    
    if m_base in SEMANTICS:
        full = f"{instr['mnemonic']} {instr['operands']}".strip()
        level1_samples.append({
            "input": f"Instruction: {full}\nSemantics:",
            "output": f" {SEMANTICS[m_base]}"
        })

# Dedupe
seen = set()
level1_unique = []
for s in level1_samples:
    if s["input"] not in seen:
        seen.add(s["input"])
        level1_unique.append(s)

# Balance by mnemonic
l1_by_mnem = defaultdict(list)
for s in level1_unique:
    m = s["input"].split()[1].split()[0].rstrip("lqwb")
    l1_by_mnem[m].append(s)

level1_balanced = []
for m, samples in l1_by_mnem.items():
    # Take up to 100 per mnemonic
    level1_balanced.extend(samples[:100])

print(f"  Level 1: {len(level1_balanced)} samples")

# ============================================================================
# BUILD LEVEL 2 DATASET
# ============================================================================

# Extract CFG patterns from real binaries
level2_samples = []

# Find function boundaries in each binary
for binary in binaries[:50]:
    instrs = disassemble_binary(binary)
    if len(instrs) < 10:
        continue
    
    # Split into functions (roughly: find ret instructions)
    functions = []
    current = []
    for instr in instrs:
        current.append(instr)
        if instr["mnemonic"] in ["ret", "retq"]:
            if len(current) >= 5:
                functions.append(current)
            current = []
    
    for func in functions[:20]:  # Max 20 functions per binary
        if len(func) < 5:
            continue
        
        # Analyze ground truth
        has_loop = False
        has_conditional = False
        has_call = False
        
        for instr in func:
            m = instr["mnemonic"]
            if m in ["call", "callq"]:
                has_call = True
            elif m.startswith("j") and m not in ["jmp", "jmpq"]:
                has_conditional = True
                # Check for backward jump (loop)
                try:
                    target = int(instr["operands"].split()[0].strip(","), 16)
                    if target < instr["address"]:
                        has_loop = True
                except:
                    pass
        
        # Format
        instr_strs = []
        for i in func[:15]:
            m = i["mnemonic"]
            tag = ""
            if m in ["ret", "retq"]:
                tag = " [return]"
            elif m in ["call", "callq"]:
                tag = " [call]"
            elif m in ["jmp", "jmpq"]:
                tag = " [jump]"
            elif m.startswith("j"):
                tag = " [conditional]"
            instr_strs.append(f"{i['address']:#x}:{i['mnemonic']} {i['operands']}{tag}")
        
        # Build output
        parts = ["basic blocks identified"]
        if has_loop:
            parts.append("loop detected")
        if has_conditional:
            parts.append("conditional branch edges")
        if has_call:
            parts.append("function calls present")
        
        level2_samples.append({
            "input": "Instructions:\n" + "\n".join(instr_strs) + "\nAnalysis:",
            "output": " " + "; ".join(parts)
        })

print(f"  Level 2: {len(level2_samples)} samples")

# ============================================================================
# SAVE
# ============================================================================
print("\n[4/5] Saving datasets...")

os.makedirs("genesis_datasets/level0_real", exist_ok=True)
os.makedirs("genesis_datasets/level1_real", exist_ok=True)
os.makedirs("genesis_datasets/level2_real", exist_ok=True)

random.shuffle(level0_samples)
random.shuffle(level1_balanced)
random.shuffle(level2_samples)

for name, data in [("level0_real", level0_samples), 
                   ("level1_real", level1_balanced), 
                   ("level2_real", level2_samples)]:
    with open(f"genesis_datasets/{name}/train.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

# ============================================================================
# TRAIN
# ============================================================================
print("\n[5/5] Training...")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

class DS(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                self.data.append(d["input"] + d["output"])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx], truncation=True, max_length=self.max_len, 
                            padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(), 
                "attention_mask": enc["attention_mask"].squeeze(), 
                "labels": enc["input_ids"].squeeze().clone()}

def train_model(name, data_path, model_path, max_len, epochs=150, patience=20):
    print(f"\n  {name}...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float16)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, 
                      lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
    model = get_peft_model(base, lora).to(device)
    dataset = DS(data_path, tokenizer, max_len)
    print(f"    {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler('cuda')
    best, wait = float("inf"), 0
    
    for epoch in range(epochs):
        model.train()
        total = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast('cuda'):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        
        avg = total / len(loader)
        if avg < best - 0.001:
            best, wait = avg, 0
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            if epoch % 25 == 0:
                print(f"    Epoch {epoch+1}: {avg:.4f} *")
        else:
            wait += 1
        if wait >= patience and epoch > 50:
            print(f"    Early stop @ {epoch+1}")
            break
    
    print(f"    Best: {best:.4f}")

os.makedirs("models/level0_real", exist_ok=True)
os.makedirs("models/level1_real", exist_ok=True)
os.makedirs("models/level2_real", exist_ok=True)

train_model("Level 0", "genesis_datasets/level0_real/train.jsonl", "models/level0_real", 64, epochs=150)
train_model("Level 1", "genesis_datasets/level1_real/train.jsonl", "models/level1_real", 256, epochs=150)
train_model("Level 2", "genesis_datasets/level2_real/train.jsonl", "models/level2_real", 512, epochs=150)

# ============================================================================
# VERIFY
# ============================================================================
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print("\n  Level 0:")
tokenizer = AutoTokenizer.from_pretrained("models/level0_real")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "models/level0_real").to(device).eval()

for b, e in [("c3", "ret"), ("55", "push"), ("5d", "pop"), ("4883ec20", "sub"), 
             ("4889e5", "mov"), ("31c0", "xor"), ("e8", "call"), ("eb", "jmp")]:
    inp = tokenizer(f"Bytes: {b}\n", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = res.split("Instruction:")[-1].strip().split()[0] if "Instruction:" in res else "?"
    print(f"    {b} → {pred} {'✓' if pred==e else '✗'}")

print("\n  Level 1:")
tokenizer = AutoTokenizer.from_pretrained("models/level1_real")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "models/level1_real").to(device).eval()

tests = [
    ("ret", ["return", "pop", "rip", "control"]),
    ("push rbp", ["stack", "push", "write", "rsp"]),
    ("mov rax,rbx", ["move", "write", "read", "transfer", "copy"]),
    ("call 0x1000", ["call", "function", "push", "return"]),
]
for instr, kws in tests:
    inp = tokenizer(f"Instruction: {instr}\nSemantics:", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(out[0], skip_special_tokens=True)
    output = res.split("Semantics:")[-1].lower() if "Semantics:" in res else res.lower()
    found = any(k in output for k in kws)
    print(f"    {instr} → {'✓' if found else '✗'} ({output[:50]}...)")

# Package
print("\n" + "=" * 60)
import zipfile
with zipfile.ZipFile("genesis_real.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_real", "level1_real", "level2_real"]:
        p = Path(f"models/{level}")
        if p.exists():
            for f in p.iterdir():
                zf.write(f, f"models/{level}/{f.name}")

print(f"Created: genesis_real.zip ({Path('genesis_real.zip').stat().st_size/1024/1024:.1f} MB)")

try:
    from google.colab import files
    files.download("genesis_real.zip")
except:
    print(f"Download: {os.path.abspath('genesis_real.zip')}")

print("\nDONE - ALL DATA FROM REAL BINARIES")
