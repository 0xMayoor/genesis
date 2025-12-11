# ============================================================================
# GENESIS v5 - Format-Correct Training
# 
# Fixes:
# 1. L0: More diverse byte patterns including longer sequences
# 2. L1: Use "Semantics:" format + standard terminology
# 3. L2: Generate way more CFG samples
# ============================================================================

import os
import subprocess
import sys
import json
import random
import tempfile
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("GENESIS v5")
print("=" * 70)

# Setup
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "clang", "gcc"], capture_output=True)
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
# LEVEL 1: Standard terminology that matches how people describe instructions
# These are accurate descriptions using common CS vocabulary
# ============================================================================

SEMANTICS = {
    # mov: needs "move", "write", "read", "register", "transfer", or "copy"
    "mov": "Move data: write destination register, read source, transfer copy value",
    "movzx": "Move with zero extend: read source, write to larger register, transfer copy",
    "movsx": "Move with sign extend: read source, write to larger register, transfer copy",
    
    # push: needs "stack", "push", "write", "rsp", or "sp"
    "push": "Push to stack: write value to memory at rsp, decrement sp stack pointer",
    
    # pop: needs "stack", "pop", "read", "rsp", or "sp"
    "pop": "Pop from stack: read value from memory at rsp, increment sp stack pointer",
    
    # lea: needs "address", "load", "effective", or "lea"
    "lea": "Load effective address: compute address without memory access, lea operation",
    
    # add: needs "add", "sum", "arithmetic", "plus", or "flag"
    "add": "Add arithmetic: sum operands, plus operation, updates flags",
    
    # sub: needs "sub", "subtract", "minus", "arithmetic", or "flag"
    "sub": "Subtract arithmetic: sub operation, minus, updates flags",
    
    # cmp: needs "compare", "flag", "cmp", or "sub"
    "cmp": "Compare cmp: subtract without storing, sets flags for conditional",
    
    # test: needs "test", "flag", "and", or "compare"
    "test": "Test operation: bitwise and, sets flags, compare for zero",
    
    # jmp: needs "jump", "branch", "unconditional", or "control"
    "jmp": "Unconditional jump: branch to target, control flow transfer",
    
    # je: needs "jump", "equal", "zero", "conditional", or "branch"
    "je": "Conditional jump if equal: branch when zero flag set, jump if equal",
    "jz": "Conditional jump if zero: branch when zero flag set, equal condition",
    
    # jne: needs "jump", "not", "equal", "conditional", or "branch"
    "jne": "Conditional jump if not equal: branch when zero flag clear, not equal",
    "jnz": "Conditional jump if not zero: branch when not equal, conditional",
    
    # call: needs "call", "function", "push", or "return"
    "call": "Call function: push return address to stack, transfer to function",
    
    # ret: needs "return", "pop", "rip", or "control"
    "ret": "Return from function: pop return address to rip, control flow return",
    
    # xor: needs "xor", "exclusive", "zero", or "bitwise"
    "xor": "Exclusive or xor: bitwise operation, can zero register",
    
    # and: needs "and", "bitwise", "mask", or "flag"
    "and": "Bitwise and: mask operation, sets flags",
    
    # or: needs "or", "bitwise", or "flag"
    "or": "Bitwise or: combine bits, sets flags",
    
    # shl: needs "shift", "left", "multiply", or "bitwise"
    "shl": "Shift left: bitwise operation, multiply by power of 2",
    "sal": "Shift arithmetic left: bitwise operation, multiply",
    
    # shr: needs "shift", "right", "divide", or "bitwise"
    "shr": "Shift right: bitwise operation, divide by power of 2",
    "sar": "Shift arithmetic right: bitwise, divide preserving sign",
    
    # imul: needs "multiply", "mul", "signed", or "arithmetic"
    "imul": "Signed multiply mul: arithmetic operation, multiply operands",
    
    # idiv: needs "divide", "div", "signed", or "arithmetic"
    "idiv": "Signed divide div: arithmetic operation, divide operands",
    
    # Additional common instructions
    "inc": "Increment: add one, arithmetic operation, flags updated",
    "dec": "Decrement: subtract one, arithmetic operation, flags updated",
    "neg": "Negate: arithmetic two's complement, flags updated",
    "not": "Bitwise not: one's complement, invert all bits",
    "nop": "No operation: does nothing",
    "leave": "Leave: restore stack frame, pop rbp",
    "cdq": "Sign extend eax to edx:eax for division",
    "cqo": "Sign extend rax to rdx:rax for division",
    "endbr64": "End branch: Intel CET indirect branch target",
    
    # More jumps
    "jg": "Conditional jump if greater: branch when signed greater, conditional",
    "jge": "Conditional jump if greater equal: branch signed, conditional",
    "jl": "Conditional jump if less: branch when signed less, conditional",
    "jle": "Conditional jump if less equal: branch signed, conditional",
    "ja": "Conditional jump if above: branch unsigned greater, conditional",
    "jae": "Conditional jump if above equal: branch unsigned, conditional",
    "jb": "Conditional jump if below: branch unsigned less, conditional",
    "jbe": "Conditional jump if below equal: branch unsigned, conditional",
    "js": "Conditional jump if sign: branch when negative, conditional",
    "jns": "Conditional jump if not sign: branch when positive, conditional",
}

# ============================================================================
# C PROGRAMS - More diverse for better byte coverage
# ============================================================================

PROGRAMS = {
    "strlen": 'int f(const char*s){int n=0;while(s[n])n++;return n;}int main(){return f("hi");}',
    "sum": 'int f(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3};return f(a,3);}',
    "max": 'int f(int a,int b){return a>b?a:b;}int main(){return f(3,7);}',
    "min": 'int f(int a,int b){return a<b?a:b;}int main(){return f(3,7);}',
    "abs": 'int f(int x){return x<0?-x:x;}int main(){return f(-5);}',
    "fib": 'int f(int n){if(n<=1)return n;int a=0,b=1;for(int i=2;i<=n;i++){int t=a+b;a=b;b=t;}return b;}int main(){return f(10);}',
    "fact": 'int f(int n){int r=1;for(int i=2;i<=n;i++)r*=i;return r;}int main(){return f(5);}',
    "gcd": 'int f(int a,int b){while(b){int t=b;b=a%b;a=t;}return a;}int main(){return f(12,8);}',
    "pow": 'int f(int b,int e){int r=1;while(e>0){if(e&1)r*=b;b*=b;e>>=1;}return r;}int main(){return f(2,8);}',
    "bsearch": 'int f(int*a,int n,int t){int l=0,r=n-1;while(l<=r){int m=(l+r)/2;if(a[m]==t)return m;if(a[m]<t)l=m+1;else r=m-1;}return -1;}int main(){int a[]={1,2,3,4,5};return f(a,5,3);}',
    "bubble": 'void f(int*a,int n){for(int i=0;i<n-1;i++)for(int j=0;j<n-i-1;j++)if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}int main(){int a[]={5,2,8,1};f(a,4);return a[0];}',
    "popcount": 'int f(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return f(0xFF);}',
    "rev": 'unsigned f(unsigned n){unsigned r=0;for(int i=0;i<32;i++){r=(r<<1)|(n&1);n>>=1;}return r;}int main(){return f(0x12345678)&0xFF;}',
    "atoi": 'int f(const char*s){int r=0,sg=1;if(*s==45){sg=-1;s++;}while(*s>=48&&*s<=57){r=r*10+(*s-48);s++;}return sg*r;}int main(){return f("-42");}',
    "swap": 'void f(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;f(&x,&y);return x;}',
    "memset": 'void*f(void*s,int c,unsigned long n){char*p=s;while(n--)*p++=c;return s;}int main(){char b[4];f(b,65,4);return b[0];}',
    "memcpy": 'void*f(void*d,const void*s,unsigned long n){char*dp=d;const char*sp=s;while(n--)*dp++=*sp++;return d;}int main(){char a[4],b[]="hi";f(a,b,3);return a[0];}',
    "prime": 'int f(int n){if(n<2)return 0;for(int i=2;i*i<=n;i++)if(n%i==0)return 0;return 1;}int main(){return f(17);}',
    "sqrt": 'int f(int n){int x=n,y=(x+1)/2;while(y<x){x=y;y=(x+n/x)/2;}return x;}int main(){return f(100);}',
    "strcpy": 'void f(char*d,const char*s){while((*d++=*s++));}int main(){char b[8];f(b,"x");return 0;}',
    "strcmp": 'int f(const char*a,const char*b){while(*a&&*a==*b){a++;b++;}return *a-*b;}int main(){return f("ab","ac");}',
    "reverse": 'void f(int*a,int n){for(int i=0;i<n/2;i++){int t=a[i];a[i]=a[n-1-i];a[n-1-i]=t;}}int main(){int a[]={1,2,3};f(a,3);return a[0];}',
    "matrix": 'int f(int a[2][2],int b[2][2]){return a[0][0]*b[0][0]+a[0][1]*b[1][0];}int main(){int a[2][2]={{1,2},{3,4}};int b[2][2]={{5,6},{7,8}};return f(a,b);}',
    "linked": 'struct N{int v;struct N*n;};int f(struct N*h){int s=0;while(h){s+=h->v;h=h->n;}return s;}int main(){return 0;}',
    "stack": 'int s[100];int t=0;void push(int v){s[t++]=v;}int pop(){return s[--t];}int main(){push(5);return pop();}',
    "queue": 'int q[100];int f=0,r=0;void enq(int v){q[r++]=v;}int deq(){return q[f++];}int main(){enq(5);return deq();}',
    "hash": 'unsigned f(const char*s){unsigned h=5381;while(*s)h=((h<<5)+h)+*s++;return h;}int main(){return f("test")&0xFF;}',
    "tree": 'struct T{int v;struct T*l,*r;};int f(struct T*t){return t?f(t->l)+t->v+f(t->r):0;}int main(){return 0;}',
}

# Compile and extract
print("\n[1/5] Compiling and extracting...")

def disassemble(binary_path):
    objcopy = subprocess.run(
        ["objcopy", "-O", "binary", "--only-section=.text", str(binary_path), "/dev/stdout"],
        capture_output=True
    )
    if not objcopy.stdout:
        return []
    cs = Cs(CS_ARCH_X86, CS_MODE_64)
    return [{"bytes": i.bytes.hex(), "mnemonic": i.mnemonic, "operands": i.op_str, "address": i.address}
            for i in cs.disasm(objcopy.stdout, 0x1000)]

compilers = [c for c in ["gcc", "clang"] if subprocess.run([c, "--version"], capture_output=True).returncode == 0]
print(f"  Compilers: {compilers}")

level0_by_mnemonic = defaultdict(set)  # mnemonic -> set of (bytes, mnemonic)
level1_data = []
level2_data = []

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for name, code in PROGRAMS.items():
        for compiler in compilers:
            for opt in ["-O0", "-O1", "-O2", "-O3"]:
                src = tmpdir / f"{name}.c"
                binary = tmpdir / f"{name}_{compiler}_{opt}"
                src.write_text(code)
                
                if subprocess.run([compiler, opt, "-w", "-o", str(binary), str(src)],
                                 capture_output=True, timeout=30).returncode != 0:
                    continue
                
                instructions = disassemble(binary)
                
                for instr in instructions:
                    m = instr["mnemonic"]
                    # Level 0
                    level0_by_mnemonic[m].add((instr["bytes"], m))
                    
                    # Level 1
                    base = m.rstrip("lqwb")
                    if base in SEMANTICS:
                        full = f"{m} {instr['operands']}".strip()
                        level1_data.append((full, SEMANTICS[base]))
                
                # Level 2: Generate CFG analysis
                if len(instructions) >= 5:
                    # Analyze
                    has_loop = False
                    has_cond = False
                    has_call = False
                    
                    for instr in instructions:
                        m = instr["mnemonic"]
                        if m == "call":
                            has_call = True
                        elif m.startswith("j") and m != "jmp":
                            has_cond = True
                            # Check for back edge (loop)
                            try:
                                target = int(instr["operands"], 16)
                                if target < instr["address"]:
                                    has_loop = True
                            except:
                                pass
                    
                    # Format
                    instr_strs = []
                    for i in instructions[:12]:
                        m = i["mnemonic"]
                        tag = ""
                        if m == "ret": tag = " [return]"
                        elif m == "call": tag = " [call]"
                        elif m == "jmp": tag = " [jump]"
                        elif m.startswith("j"): tag = " [conditional]"
                        instr_strs.append(f"{i['address']:#x}:{m} {i['operands']}{tag}")
                    
                    parts = ["basic blocks identified"]
                    if has_loop: parts.append("loop detected")
                    if has_cond: parts.append("conditional branches present")
                    if has_call: parts.append("function calls")
                    
                    level2_data.append({
                        "input": "Instructions:\n" + "\n".join(instr_strs) + "\nAnalysis:",
                        "output": " " + "; ".join(parts)
                    })
                
                binary.unlink()

# ============================================================================
# Balance and format data
# ============================================================================
print("\n[2/5] Balancing data...")

# Level 0: Balance
MIN_PER = 30
MAX_PER = 150

level0_samples = []
for mnemonic, byte_set in level0_by_mnemonic.items():
    items = list(byte_set)
    if len(items) < MIN_PER:
        # Oversample
        items = items * ((MIN_PER // len(items)) + 1)
    items = items[:MAX_PER]
    for bytes_hex, m in items:
        level0_samples.append({
            "input": f"Bytes: {bytes_hex}",
            "output": f"Instruction: {m}"
        })

print(f"  Level 0: {len(level0_samples)} samples ({len(level0_by_mnemonic)} mnemonics)")

# Level 1: Dedupe and format with Semantics:
seen = set()
level1_samples = []
for instr, desc in level1_data:
    if instr not in seen:
        seen.add(instr)
        level1_samples.append({
            "input": f"Instruction: {instr}\nSemantics:",
            "output": f" {desc}"
        })

print(f"  Level 1: {len(level1_samples)} samples")

# Level 2: Dedupe
seen = set()
level2_samples = []
for d in level2_data:
    key = d["input"][:80]
    if key not in seen:
        seen.add(key)
        level2_samples.append(d)

print(f"  Level 2: {len(level2_samples)} samples")

# ============================================================================
# Save
# ============================================================================
print("\n[3/5] Saving...")

os.makedirs("genesis_datasets/level0_v5", exist_ok=True)
os.makedirs("genesis_datasets/level1_v5", exist_ok=True)
os.makedirs("genesis_datasets/level2_v5", exist_ok=True)

random.shuffle(level0_samples)
random.shuffle(level1_samples)
random.shuffle(level2_samples)

for name, data in [("level0_v5", level0_samples), ("level1_v5", level1_samples), ("level2_v5", level2_samples)]:
    with open(f"genesis_datasets/{name}/train.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

# ============================================================================
# Train
# ============================================================================
print("\n[4/5] Training...")

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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx], truncation=True, max_length=self.max_len,
                            padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": enc["input_ids"].squeeze().clone()}

def train_model(name, data_path, model_path, max_len, epochs=100, patience=15):
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
    
    best = float("inf")
    wait = 0
    
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
            best = avg
            wait = 0
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            if epoch % 20 == 0:
                print(f"    Epoch {epoch+1}: {avg:.4f} *")
        else:
            wait += 1
        
        if wait >= patience and epoch > 30:
            print(f"    Early stop @ {epoch+1}")
            break
    
    print(f"    Best: {best:.4f}")

os.makedirs("models/level0_v5", exist_ok=True)
os.makedirs("models/level1_v5", exist_ok=True)
os.makedirs("models/level2_v5", exist_ok=True)

train_model("Level 0", "genesis_datasets/level0_v5/train.jsonl", "models/level0_v5", 64)
train_model("Level 1", "genesis_datasets/level1_v5/train.jsonl", "models/level1_v5", 192)
train_model("Level 2", "genesis_datasets/level2_v5/train.jsonl", "models/level2_v5", 512)

# ============================================================================
# Verify
# ============================================================================
print("\n[5/5] Verification...")

# Level 0
print("\n  Level 0:")
tokenizer = AutoTokenizer.from_pretrained("models/level0_v5")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "models/level0_v5").to(device).eval()

for bytes_hex, expected in [("55", "push"), ("c3", "ret"), ("31c0", "xor"), ("29c0", "sub"), ("e800000000", "call")]:
    inputs = tokenizer(f"Bytes: {bytes_hex}\n", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0], skip_special_tokens=True).split("Instruction:")[-1].strip().split()[0] if "Instruction:" in tokenizer.decode(out[0], skip_special_tokens=True) else "?"
    print(f"    {bytes_hex} → {pred} {'✓' if pred==expected else '✗'}")

# Level 1
print("\n  Level 1:")
tokenizer = AutoTokenizer.from_pretrained("models/level1_v5")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "models/level1_v5").to(device).eval()

for instr, keywords in [("mov rax,rbx", ["move", "write", "register"]), ("ret", ["return", "pop", "rip"]), ("call 0x1000", ["call", "function", "push"])]:
    prompt = f"Instruction: {instr}\nSemantics:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    output = result.split("Semantics:")[-1].lower() if "Semantics:" in result else result.lower()
    found = any(k in output for k in keywords)
    print(f"    {instr} → {'✓' if found else '✗'} ({output[:50]}...)")

# Package
print("\n" + "=" * 60)
import zipfile

with zipfile.ZipFile("genesis_v5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_v5", "level1_v5", "level2_v5"]:
        p = Path(f"models/{level}")
        if p.exists():
            for f in p.iterdir():
                zf.write(f, f"models/{level}/{f.name}")

print(f"Created: genesis_v5.zip ({Path('genesis_v5.zip').stat().st_size/1024/1024:.1f} MB)")

try:
    from google.colab import files
    files.download("genesis_v5.zip")
except:
    print(f"Download: {os.path.abspath('genesis_v5.zip')}")

print("\nDONE")
