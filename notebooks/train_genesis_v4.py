# ============================================================================
# GENESIS v4 - Proper Training
# 
# Principles:
# 1. Ground truth: Capstone (industry standard)
# 2. Data source: Real compiled binaries
# 3. Data engineering: Balanced coverage (NOT test gaming)
# 4. Semantics: From Intel/AMD ISA manuals
#
# ============================================================================

import os
import subprocess
import sys
import json
import random
import tempfile
from pathlib import Path
from typing import Optional
from collections import defaultdict

print("=" * 70)
print("GENESIS v4 - PROPER TRAINING")
print("=" * 70)

# ============================================================================
# SETUP - Fix the double-clone issue
# ============================================================================
print("\n[1/7] Setup...")

# Install dependencies first
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "clang", "gcc"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "peft", "accelerate", "capstone"], check=True)

# Handle directory - don't clone if already in genesis
cwd = os.getcwd()
if "genesis" not in cwd:
    if os.path.exists("genesis"):
        subprocess.run(["rm", "-rf", "genesis"])
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
    os.chdir("genesis")

# Verify we're in the right place
print(f"  Working dir: {os.getcwd()}")

# Verify compilers
compilers = []
for c in ["gcc", "clang"]:
    if subprocess.run([c, "--version"], capture_output=True).returncode == 0:
        compilers.append(c)
print(f"  Compilers: {compilers}")

# ============================================================================
# CAPSTONE GROUND TRUTH
# ============================================================================
print("\n[2/7] Setting up Capstone ground truth...")

from capstone import Cs, CS_ARCH_X86, CS_MODE_64

def disassemble_binary(binary_path: Path) -> list[dict]:
    """Extract instructions from binary using Capstone (GROUND TRUTH)."""
    # Get .text section
    objcopy = subprocess.run(
        ["objcopy", "-O", "binary", "--only-section=.text", 
         str(binary_path), "/dev/stdout"],
        capture_output=True
    )
    text_bytes = objcopy.stdout
    if not text_bytes:
        return []
    
    # Capstone disassembly - THIS IS GROUND TRUTH
    cs = Cs(CS_ARCH_X86, CS_MODE_64)
    instructions = []
    for instr in cs.disasm(text_bytes, 0x1000):
        instructions.append({
            "address": instr.address,
            "bytes": instr.bytes.hex(),
            "mnemonic": instr.mnemonic,
            "operands": instr.op_str,
        })
    return instructions

# ============================================================================
# REAL C PROGRAMS
# ============================================================================

PROGRAMS = {
    "strlen": 'int f(const char*s){int n=0;while(s[n])n++;return n;} int main(){return f("hi");}',
    "strcpy": 'void f(char*d,const char*s){while((*d++=*s++));} int main(){char b[8];f(b,"x");return 0;}',
    "sum": 'int f(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;} int main(){int a[]={1,2,3};return f(a,3);}',
    "max": 'int f(int a,int b){return a>b?a:b;} int main(){return f(3,7);}',
    "min": 'int f(int a,int b){return a<b?a:b;} int main(){return f(3,7);}',
    "abs": 'int f(int x){return x<0?-x:x;} int main(){return f(-5);}',
    "fib": 'int f(int n){if(n<=1)return n;int a=0,b=1;for(int i=2;i<=n;i++){int t=a+b;a=b;b=t;}return b;} int main(){return f(10);}',
    "fact": 'int f(int n){int r=1;for(int i=2;i<=n;i++)r*=i;return r;} int main(){return f(5);}',
    "gcd": 'int f(int a,int b){while(b){int t=b;b=a%b;a=t;}return a;} int main(){return f(12,8);}',
    "pow": 'int f(int b,int e){int r=1;while(e>0){if(e&1)r*=b;b*=b;e>>=1;}return r;} int main(){return f(2,8);}',
    "bsearch": 'int f(int*a,int n,int t){int l=0,r=n-1;while(l<=r){int m=(l+r)/2;if(a[m]==t)return m;if(a[m]<t)l=m+1;else r=m-1;}return -1;} int main(){int a[]={1,2,3,4,5};return f(a,5,3);}',
    "bubble": 'void f(int*a,int n){for(int i=0;i<n-1;i++)for(int j=0;j<n-i-1;j++)if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}} int main(){int a[]={5,2,8,1};f(a,4);return a[0];}',
    "popcount": 'int f(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;} int main(){return f(0xFF);}',
    "clz": 'int f(unsigned n){int c=0;while(!(n&0x80000000)){c++;n<<=1;}return c;} int main(){return f(0x0F000000);}',
    "rev": 'unsigned f(unsigned n){unsigned r=0;for(int i=0;i<32;i++){r=(r<<1)|(n&1);n>>=1;}return r;} int main(){return f(0x12345678)&0xFF;}',
    "atoi": 'int f(const char*s){int r=0,sg=1;if(*s==45){sg=-1;s++;}while(*s>=48&&*s<=57){r=r*10+(*s-48);s++;}return sg*r;} int main(){return f("-42");}',
    "isdig": 'int f(char c){return c>=48&&c<=57;} int main(){return f(53);}',
    "upper": 'char f(char c){return(c>=97&&c<=122)?c-32:c;} int main(){return f(97);}',
    "swap": 'void f(int*a,int*b){int t=*a;*a=*b;*b=t;} int main(){int x=1,y=2;f(&x,&y);return x;}',
    "memset": 'void*f(void*s,int c,unsigned long n){char*p=s;while(n--)*p++=c;return s;} int main(){char b[4];f(b,65,4);return b[0];}',
    "memcpy": 'void*f(void*d,const void*s,unsigned long n){char*dp=d;const char*sp=s;while(n--)*dp++=*sp++;return d;} int main(){char a[4],b[]="hi";f(a,b,3);return a[0];}',
    "prime": 'int f(int n){if(n<2)return 0;for(int i=2;i*i<=n;i++)if(n%i==0)return 0;return 1;} int main(){return f(17);}',
    "sqrt": 'int f(int n){int x=n,y=(x+1)/2;while(y<x){x=y;y=(x+n/x)/2;}return x;} int main(){return f(100);}',
    "log2": 'int f(unsigned n){int r=0;while(n>>=1)r++;return r;} int main(){return f(256);}',
    "div": 'int f(int a,int b){return a/b;} int main(){return f(10,3);}',
    "mod": 'int f(int a,int b){return a%b;} int main(){return f(10,3);}',
    "neg": 'int f(int x){return -x;} int main(){return f(5);}',
    "not": 'int f(int x){return ~x;} int main(){return f(0);}',
    "shl": 'int f(int x,int n){return x<<n;} int main(){return f(1,4);}',
    "shr": 'int f(int x,int n){return x>>n;} int main(){return f(16,2);}',
    "and": 'int f(int a,int b){return a&b;} int main(){return f(0xF0,0x0F);}',
    "or": 'int f(int a,int b){return a|b;} int main(){return f(0xF0,0x0F);}',
    "xor": 'int f(int a,int b){return a^b;} int main(){return f(0xFF,0x0F);}',
}

# ============================================================================
# INSTRUCTION DESCRIPTIONS FROM ISA MANUALS
# ============================================================================

ISA_DESCRIPTIONS = {
    "mov": "Copies data from source to destination",
    "movzx": "Copies with zero extension to larger register",
    "movsx": "Copies with sign extension to larger register",
    "movsxd": "Copies dword with sign extension to qword",
    "lea": "Computes effective address without memory access",
    "push": "Decrements stack pointer and stores value on stack",
    "pop": "Loads value from stack and increments stack pointer",
    "add": "Adds source to destination and sets flags",
    "sub": "Subtracts source from destination and sets flags",
    "inc": "Increments operand by one",
    "dec": "Decrements operand by one",
    "neg": "Two's complement negation",
    "mul": "Unsigned multiply",
    "imul": "Signed multiply",
    "div": "Unsigned divide",
    "idiv": "Signed divide",
    "and": "Bitwise AND operation",
    "or": "Bitwise OR operation",
    "xor": "Bitwise exclusive OR operation",
    "not": "One's complement negation",
    "shl": "Shift left logical",
    "sal": "Shift left arithmetic",
    "shr": "Shift right logical",
    "sar": "Shift right arithmetic",
    "cmp": "Compares operands by subtracting and sets flags",
    "test": "Compares operands by ANDing and sets flags",
    "jmp": "Unconditional jump",
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
    "call": "Pushes return address and transfers control",
    "ret": "Returns from procedure by popping return address",
    "leave": "Releases stack frame (mov rsp,rbp; pop rbp)",
    "nop": "No operation",
    "endbr64": "End branch marker for indirect branch tracking",
    "endbr32": "End branch marker for indirect branch tracking",
    "cdq": "Sign extends EAX into EDX:EAX",
    "cdqe": "Sign extends EAX into RAX",
    "cqo": "Sign extends RAX into RDX:RAX",
    "xchg": "Exchanges two operands",
    "sete": "Set byte if equal",
    "setne": "Set byte if not equal",
    "setg": "Set byte if greater",
    "setl": "Set byte if less",
    "cmove": "Conditional move if equal",
    "cmovne": "Conditional move if not equal",
    "int3": "Breakpoint trap",
}

# ============================================================================
# COMPILE AND EXTRACT
# ============================================================================
print("\n[3/7] Compiling programs and extracting with Capstone...")

level0_raw = defaultdict(list)  # mnemonic -> [(bytes, mnemonic)]
level1_raw = []  # (instruction_str, description)
level2_raw = []  # (instruction_sequence, analysis)

opt_levels = ["-O0", "-O1", "-O2", "-O3"]

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    total = len(PROGRAMS) * len(compilers) * len(opt_levels)
    count = 0
    
    for name, code in PROGRAMS.items():
        for compiler in compilers:
            for opt in opt_levels:
                count += 1
                if count % 50 == 0:
                    print(f"  Progress: {count}/{total}")
                
                # Compile
                src = tmpdir / f"{name}.c"
                binary = tmpdir / f"{name}_{compiler}_{opt}"
                src.write_text(code)
                
                result = subprocess.run(
                    [compiler, opt, "-w", "-o", str(binary), str(src)],
                    capture_output=True, timeout=30
                )
                if result.returncode != 0:
                    continue
                
                # Extract with Capstone (GROUND TRUTH)
                instructions = disassemble_binary(binary)
                
                for instr in instructions:
                    m = instr["mnemonic"]
                    # Level 0: bytes -> mnemonic
                    level0_raw[m].append({
                        "bytes": instr["bytes"],
                        "mnemonic": m
                    })
                    
                    # Level 1: instruction -> description
                    if m in ISA_DESCRIPTIONS:
                        full = f"{m} {instr['operands']}".strip()
                        level1_raw.append({
                            "instruction": full,
                            "description": ISA_DESCRIPTIONS[m]
                        })
                
                # Level 2: CFG for sequences
                if len(instructions) >= 5:
                    # Detect control flow features
                    has_loop = any(
                        i["mnemonic"].startswith("j") and 
                        i["operands"].startswith("0x") and
                        int(i["operands"], 16) < i["address"]
                        for i in instructions
                        if i["mnemonic"] not in ["jmp"]
                    )
                    has_cond = any(
                        i["mnemonic"].startswith("j") and i["mnemonic"] not in ["jmp"]
                        for i in instructions
                    )
                    has_call = any(i["mnemonic"] == "call" for i in instructions)
                    
                    # Build input
                    instr_strs = []
                    for i in instructions[:12]:
                        m = i["mnemonic"]
                        tag = ""
                        if m == "ret": tag = " [return]"
                        elif m == "call": tag = " [call]"
                        elif m == "jmp": tag = " [jump]"
                        elif m.startswith("j"): tag = " [conditional]"
                        instr_strs.append(f"{i['address']:#x}:{m} {i['operands']}{tag}")
                    
                    # Build output
                    parts = ["basic blocks"]
                    if has_loop: parts.append("loop detected")
                    if has_cond: parts.append("conditional branches")
                    if has_call: parts.append("function calls")
                    
                    level2_raw.append({
                        "input": "Instructions:\n" + "\n".join(instr_strs),
                        "output": "Analysis: " + "; ".join(parts)
                    })
                
                binary.unlink()

# ============================================================================
# BALANCE DATA
# ============================================================================
print("\n[4/7] Balancing training data...")

# Level 0: Balance by mnemonic type
print(f"  Raw Level 0 distribution:")
for m, samples in sorted(level0_raw.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"    {m}: {len(samples)}")

# Target: At least 50 samples per instruction type, max 200
MIN_SAMPLES = 50
MAX_SAMPLES = 200

level0_balanced = []
for mnemonic, samples in level0_raw.items():
    # Dedupe by bytes
    seen = set()
    unique = []
    for s in samples:
        if s["bytes"] not in seen:
            seen.add(s["bytes"])
            unique.append(s)
    
    # Sample or oversample
    if len(unique) >= MIN_SAMPLES:
        selected = random.sample(unique, min(len(unique), MAX_SAMPLES))
    else:
        # Oversample rare instructions
        selected = unique * (MIN_SAMPLES // max(1, len(unique)) + 1)
        selected = selected[:MIN_SAMPLES]
    
    for s in selected:
        level0_balanced.append({
            "input": f"Bytes: {s['bytes']}",
            "output": f"Instruction: {s['mnemonic']}"
        })

print(f"  Balanced Level 0: {len(level0_balanced)} samples")

# Level 1: Dedupe and balance
seen = set()
level1_unique = []
for s in level1_raw:
    key = s["instruction"]
    if key not in seen:
        seen.add(key)
        level1_unique.append({
            "input": f"Instruction: {s['instruction']}",
            "output": s["description"]
        })

print(f"  Unique Level 1: {len(level1_unique)} samples")

# Level 2: Dedupe
seen = set()
level2_unique = []
for s in level2_raw:
    key = s["input"][:100]
    if key not in seen:
        seen.add(key)
        level2_unique.append(s)

print(f"  Unique Level 2: {len(level2_unique)} samples")

# ============================================================================
# SAVE DATASETS
# ============================================================================
print("\n[5/7] Saving datasets...")

os.makedirs("genesis_datasets/level0_v4", exist_ok=True)
os.makedirs("genesis_datasets/level1_v4", exist_ok=True)
os.makedirs("genesis_datasets/level2_v4", exist_ok=True)

random.shuffle(level0_balanced)
random.shuffle(level1_unique)
random.shuffle(level2_unique)

with open("genesis_datasets/level0_v4/train.jsonl", "w") as f:
    for s in level0_balanced:
        f.write(json.dumps(s) + "\n")

with open("genesis_datasets/level1_v4/train.jsonl", "w") as f:
    for s in level1_unique:
        f.write(json.dumps(s) + "\n")

with open("genesis_datasets/level2_v4/train.jsonl", "w") as f:
    for s in level2_unique:
        f.write(json.dumps(s) + "\n")

print(f"  Saved: level0_v4 ({len(level0_balanced)}), level1_v4 ({len(level1_unique)}), level2_v4 ({len(level2_unique)})")

# ============================================================================
# TRAINING
# ============================================================================
print("\n[6/7] Training...")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d["input"] + "\n" + d["output"]
                self.samples.append(text)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx], 
            truncation=True, 
            max_length=self.max_len,
            padding="max_length", 
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone()
        }

def train(name, data_path, model_path, max_len, epochs=100, patience=15):
    print(f"\n  Training {name}...")
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float16)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=64, 
        lora_alpha=128,
        lora_dropout=0.05, 
        target_modules=["c_attn", "c_proj"]
    )
    model = get_peft_model(base, lora).to(device)
    
    dataset = TextDataset(data_path, tokenizer, max_len)
    print(f"    Samples: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float("inf")
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
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
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        
        if avg < best_loss - 0.001:
            best_loss = avg
            no_improve = 0
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            if epoch % 20 == 0:
                print(f"    Epoch {epoch+1}: {avg:.4f} *")
        else:
            no_improve += 1
        
        if no_improve >= patience and epoch > 30:
            print(f"    Early stop at epoch {epoch+1}")
            break
    
    print(f"    Best: {best_loss:.4f}")
    return best_loss

os.makedirs("models/level0_v4", exist_ok=True)
os.makedirs("models/level1_v4", exist_ok=True)
os.makedirs("models/level2_v4", exist_ok=True)

train("Level 0", "genesis_datasets/level0_v4/train.jsonl", "models/level0_v4", 64)
train("Level 1", "genesis_datasets/level1_v4/train.jsonl", "models/level1_v4", 128)
train("Level 2", "genesis_datasets/level2_v4/train.jsonl", "models/level2_v4", 384)

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n[7/7] Verification against Capstone...")

# Test bytes that Capstone knows
test_cases = [
    ("55", "push"),
    ("5d", "pop"),
    ("c3", "ret"),
    ("89e5", "mov"),
    ("4889e5", "mov"),
    ("31c0", "xor"),
    ("01c0", "add"),
    ("29c0", "sub"),
    ("e800000000", "call"),
    ("eb00", "jmp"),
    ("7400", "je"),
]

tokenizer = AutoTokenizer.from_pretrained("models/level0_v4")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "models/level0_v4").to(device).eval()

correct = 0
print("\n  Level 0 tests:")
for bytes_hex, expected in test_cases:
    prompt = f"Bytes: {bytes_hex}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = result.split("Instruction:")[-1].strip().split()[0] if "Instruction:" in result else "?"
    
    ok = pred.lower() == expected
    if ok: correct += 1
    print(f"    {bytes_hex} → {pred} {'✓' if ok else '✗'} (expected: {expected})")

acc = correct / len(test_cases)
print(f"\n  Accuracy: {correct}/{len(test_cases)} = {acc*100:.0f}%")

# ============================================================================
# PACKAGE
# ============================================================================
print("\n" + "=" * 60)
print("PACKAGING")
print("=" * 60)

import zipfile

zip_path = "genesis_v4.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_v4", "level1_v4", "level2_v4"]:
        model_path = Path(f"models/{level}")
        if model_path.exists():
            for f in model_path.iterdir():
                zf.write(f, f"models/{level}/{f.name}")
                
print(f"  Created: {zip_path} ({Path(zip_path).stat().st_size / 1024 / 1024:.1f} MB)")

# Try download, but don't fail if not in notebook
try:
    from google.colab import files
    files.download(zip_path)
    print("  Download started!")
except:
    print(f"  Manual download: {os.path.abspath(zip_path)}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
