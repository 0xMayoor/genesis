#!/usr/bin/env python3
"""
LEVEL 0 ULTIMATE - Train until 100%

Goal: bytes → mnemonic with 100% accuracy
Method: Train until loss stops improving, no epoch limit
Data: Real binaries from system, maximum coverage
"""

import os
import subprocess
import sys
import json
import random
from pathlib import Path
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("=" * 60)
print("LEVEL 0 ULTIMATE")
print("=" * 60)

# ============================================================================
# SETUP
# ============================================================================
print("\n[SETUP]")

# Install quietly
subprocess.run(["apt-get", "update"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "binutils", "gcc", "clang"], capture_output=True)
result = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch", "transformers", "peft", "accelerate", "capstone"],
                       capture_output=True)

# Handle genesis directory
if "genesis" not in os.getcwd():
    subprocess.run(["rm", "-rf", "genesis"], capture_output=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/0xMayoor/genesis.git"])
    os.chdir("genesis")

print(f"  Dir: {os.getcwd()}")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ============================================================================
# DATA COLLECTION - Maximum coverage
# ============================================================================
print("\n[DATA COLLECTION]")

def disassemble(binary_path):
    """Disassemble .text section with Capstone."""
    try:
        result = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", str(binary_path), "/dev/stdout"],
            capture_output=True, timeout=30
        )
        if not result.stdout:
            return []
        cs = Cs(CS_ARCH_X86, CS_MODE_64)
        return [(i.bytes.hex(), i.mnemonic) for i in cs.disasm(result.stdout, 0)]
    except:
        return []

# Source 1: System binaries
print("  Finding system binaries...")
system_binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/lib/x86_64-linux-gnu"]:
    if Path(path).exists():
        for f in Path(path).iterdir():
            if f.is_file() and not f.is_symlink():
                try:
                    with open(f, "rb") as fp:
                        if fp.read(4) == b'\x7fELF':
                            system_binaries.append(f)
                except:
                    pass
            if len(system_binaries) >= 200:
                break
    if len(system_binaries) >= 200:
        break

print(f"    Found {len(system_binaries)} system ELFs")

# Source 2: Compile diverse programs
print("  Compiling programs...")

PROGRAMS = {
    # String operations
    "strlen": 'int f(const char*s){int n=0;while(s[n])n++;return n;}int main(){return f("x");}',
    "strcpy": 'void f(char*d,const char*s){while((*d++=*s++));}int main(){char b[4];f(b,"x");return 0;}',
    "strcmp": 'int f(const char*a,const char*b){while(*a&&*a==*b){a++;b++;}return*a-*b;}int main(){return f("a","b");}',
    "strcat": 'char*f(char*d,const char*s){char*r=d;while(*d)d++;while((*d++=*s++));return r;}int main(){char b[8]="a";f(b,"b");return 0;}',
    # Memory operations
    "memcpy": 'void*f(void*d,const void*s,int n){char*p=d;const char*q=s;while(n--)*p++=*q++;return d;}int main(){char a[4],b[]="x";f(a,b,2);return 0;}',
    "memset": 'void*f(void*s,int c,int n){char*p=s;while(n--)*p++=c;return s;}int main(){char b[4];f(b,0,4);return 0;}',
    "memmove": 'void*f(void*d,const void*s,int n){char*dp=d;const char*sp=s;if(dp<sp)while(n--)*dp++=*sp++;else{dp+=n;sp+=n;while(n--)*--dp=*--sp;}return d;}int main(){char b[]="abc";f(b+1,b,2);return 0;}',
    # Math
    "abs": 'int f(int x){return x<0?-x:x;}int main(){return f(-5);}',
    "min": 'int f(int a,int b){return a<b?a:b;}int main(){return f(3,7);}',
    "max": 'int f(int a,int b){return a>b?a:b;}int main(){return f(3,7);}',
    "clamp": 'int f(int x,int l,int h){return x<l?l:x>h?h:x;}int main(){return f(5,0,10);}',
    "gcd": 'int f(int a,int b){while(b){int t=b;b=a%b;a=t;}return a;}int main(){return f(12,8);}',
    "pow": 'int f(int b,int e){int r=1;while(e>0){if(e&1)r*=b;b*=b;e>>=1;}return r;}int main(){return f(2,8);}',
    "sqrt": 'int f(int n){int x=n,y=(x+1)/2;while(y<x){x=y;y=(x+n/x)/2;}return x;}int main(){return f(100);}',
    "fib": 'int f(int n){if(n<=1)return n;int a=0,b=1;for(int i=2;i<=n;i++){int t=a+b;a=b;b=t;}return b;}int main(){return f(10);}',
    "fact": 'int f(int n){int r=1;for(int i=2;i<=n;i++)r*=i;return r;}int main(){return f(5);}',
    "prime": 'int f(int n){if(n<2)return 0;for(int i=2;i*i<=n;i++)if(n%i==0)return 0;return 1;}int main(){return f(17);}',
    # Bit operations
    "popcount": 'int f(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return f(255);}',
    "clz": 'int f(unsigned n){int c=0;if(!n)return 32;while(!(n&0x80000000)){c++;n<<=1;}return c;}int main(){return f(0xF);}',
    "ctz": 'int f(unsigned n){int c=0;if(!n)return 32;while(!(n&1)){c++;n>>=1;}return c;}int main(){return f(0xF0);}',
    "reverse": 'unsigned f(unsigned n){unsigned r=0;for(int i=0;i<32;i++){r=(r<<1)|(n&1);n>>=1;}return r;}int main(){return f(1)&0xFF;}',
    "rotl": 'unsigned f(unsigned n,int k){return(n<<k)|(n>>(32-k));}int main(){return f(1,4);}',
    "rotr": 'unsigned f(unsigned n,int k){return(n>>k)|(n<<(32-k));}int main(){return f(16,4);}',
    # Array operations
    "sum": 'int f(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3};return f(a,3);}',
    "find": 'int f(int*a,int n,int t){for(int i=0;i<n;i++)if(a[i]==t)return i;return-1;}int main(){int a[]={1,2,3};return f(a,3,2);}',
    "count": 'int f(int*a,int n,int t){int c=0;for(int i=0;i<n;i++)if(a[i]==t)c++;return c;}int main(){int a[]={1,2,1};return f(a,3,1);}',
    "bsearch": 'int f(int*a,int n,int t){int l=0,r=n-1;while(l<=r){int m=(l+r)/2;if(a[m]==t)return m;if(a[m]<t)l=m+1;else r=m-1;}return-1;}int main(){int a[]={1,2,3,4,5};return f(a,5,3);}',
    "bubble": 'void f(int*a,int n){for(int i=0;i<n-1;i++)for(int j=0;j<n-i-1;j++)if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}int main(){int a[]={3,1,2};f(a,3);return a[0];}',
    "insert": 'void f(int*a,int n){for(int i=1;i<n;i++){int k=a[i],j=i-1;while(j>=0&&a[j]>k){a[j+1]=a[j];j--;}a[j+1]=k;}}int main(){int a[]={3,1,2};f(a,3);return a[0];}',
    "select": 'void f(int*a,int n){for(int i=0;i<n-1;i++){int m=i;for(int j=i+1;j<n;j++)if(a[j]<a[m])m=j;int t=a[i];a[i]=a[m];a[m]=t;}}int main(){int a[]={3,1,2};f(a,3);return a[0];}',
    "reverse_arr": 'void f(int*a,int n){for(int i=0;i<n/2;i++){int t=a[i];a[i]=a[n-1-i];a[n-1-i]=t;}}int main(){int a[]={1,2,3};f(a,3);return a[0];}',
    # Control flow
    "switch": 'int f(int x){switch(x){case 1:return 10;case 2:return 20;case 3:return 30;default:return 0;}}int main(){return f(2);}',
    "nested": 'int f(int n){int s=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++)s+=i*j;return s;}int main(){return f(3);}',
    "recur": 'int f(int n){return n<=1?n:f(n-1)+f(n-2);}int main(){return f(10);}',
    # Pointer operations
    "swap": 'void f(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;f(&x,&y);return x;}',
    "indirect": 'int f(int**pp){return**pp;}int main(){int x=42,*p=&x;return f(&p);}',
}

import tempfile
compiled = []
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
                    compiled.append(binary)

print(f"    Compiled {len(compiled)} binaries")

# Collect all instructions
print("  Extracting instructions...")
all_pairs = []  # (bytes_hex, mnemonic)

for binary in system_binaries + compiled:
    pairs = disassemble(binary)
    all_pairs.extend(pairs)

print(f"    Raw: {len(all_pairs)} instruction instances")

# Group by mnemonic
by_mnemonic = defaultdict(set)
for bytes_hex, mnemonic in all_pairs:
    by_mnemonic[mnemonic].add(bytes_hex)

print(f"    Unique mnemonics: {len(by_mnemonic)}")

# Show distribution
mnem_counts = {m: len(v) for m, v in by_mnemonic.items()}
top = sorted(mnem_counts.items(), key=lambda x: -x[1])[:15]
print(f"    Top 15: {[f'{m}:{c}' for m,c in top]}")

# ============================================================================
# BUILD BALANCED DATASET
# ============================================================================
print("\n[DATASET]")

# For each mnemonic, collect unique byte patterns
# Balance: ensure minimum representation
MIN_SAMPLES = 30
MAX_SAMPLES = 500

samples = []
for mnemonic, byte_set in by_mnemonic.items():
    unique_bytes = list(byte_set)
    
    # Skip very rare mnemonics (likely noise)
    if len(unique_bytes) < 3:
        continue
    
    # Oversample if needed
    if len(unique_bytes) < MIN_SAMPLES:
        factor = (MIN_SAMPLES // len(unique_bytes)) + 1
        unique_bytes = unique_bytes * factor
    
    # Cap
    unique_bytes = unique_bytes[:MAX_SAMPLES]
    
    for b in unique_bytes:
        samples.append({"bytes": b, "mnemonic": mnemonic})

print(f"  Total samples: {len(samples)}")

# Shuffle
random.shuffle(samples)

# Format for training
# Input: "Bytes: {hex}\n"
# Output: "Instruction: {mnemonic}"
train_data = []
for s in samples:
    train_data.append({
        "input": f"Bytes: {s['bytes']}\n",
        "output": f"Instruction: {s['mnemonic']}"
    })

# Save
os.makedirs("genesis_datasets/level0_ultimate", exist_ok=True)
with open("genesis_datasets/level0_ultimate/train.jsonl", "w") as f:
    for d in train_data:
        f.write(json.dumps(d) + "\n")

print(f"  Saved to genesis_datasets/level0_ultimate/train.jsonl")

# ============================================================================
# TRAINING - No epoch limit, stop when converged
# ============================================================================
print("\n[TRAINING]")

class ByteMnemonicDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        for d in data:
            text = d["input"] + d["output"]
            self.samples.append(text)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

# Initialize
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"]
)

model = get_peft_model(base_model, lora_config)
model = model.to(device)

dataset = ByteMnemonicDataset(train_data, tokenizer)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"  Samples: {len(dataset)}")
print(f"  Batches: {len(loader)}")
print(f"  Training until convergence...")

# Training loop - no max epochs
best_loss = float("inf")
patience = 30  # Stop after 30 epochs without improvement
wait = 0
epoch = 0
min_epochs = 100  # Train at least this many

os.makedirs("models/level0_ultimate", exist_ok=True)

while True:
    epoch += 1
    model.train()
    
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    
    # Check improvement
    if avg_loss < best_loss - 0.0001:
        best_loss = avg_loss
        wait = 0
        model.save_pretrained("models/level0_ultimate")
        tokenizer.save_pretrained("models/level0_ultimate")
        marker = "*"
    else:
        wait += 1
        marker = ""
    
    # Log every 10 epochs
    if epoch % 10 == 0 or marker == "*":
        print(f"    Epoch {epoch}: loss={avg_loss:.4f} best={best_loss:.4f} {marker}")
    
    # Check stopping condition
    if epoch >= min_epochs and wait >= patience:
        print(f"    Converged at epoch {epoch}")
        break
    
    # Safety limit
    if epoch >= 1000:
        print(f"    Safety limit reached")
        break

print(f"  Final: loss={best_loss:.4f}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n[VERIFICATION]")

# Reload best model
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("distilgpt2"),
    "models/level0_ultimate"
).to(device)
model.eval()

def predict(bytes_hex):
    """Predict mnemonic from bytes."""
    prompt = f"Bytes: {bytes_hex}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract mnemonic
    if "Instruction:" in result:
        after = result.split("Instruction:")[-1].strip()
        if after:
            return after.split()[0]
    return ""

# Test on known patterns
TEST_CASES = [
    # Basic
    ("c3", "ret"),
    ("55", "push"),
    ("5d", "pop"),
    ("90", "nop"),
    ("c9", "leave"),
    # REX.W + common
    ("4889e5", "mov"),
    ("4883ec20", "sub"),
    ("4883c420", "add"),
    ("4839c0", "cmp"),
    ("4885c0", "test"),
    ("4831c0", "xor"),
    # No REX
    ("89e5", "mov"),
    ("83ec20", "sub"),
    ("83c420", "add"),
    ("39c0", "cmp"),
    ("85c0", "test"),
    ("31c0", "xor"),
    ("29c0", "sub"),
    ("01c0", "add"),
    # Jumps
    ("eb00", "jmp"),
    ("e900000000", "jmp"),
    ("7400", "je"),
    ("7500", "jne"),
    ("7c00", "jl"),
    ("7f00", "jg"),
    # Call
    ("e800000000", "call"),
    ("ffd0", "call"),
    # LEA
    ("488d05", "lea"),
    ("8d45fc", "lea"),
    # MOVZX
    ("0fb6c0", "movzx"),
    ("0fb645fc", "movzx"),
    # Special
    ("f30f1efa", "endbr64"),
]

correct = 0
total = len(TEST_CASES)

for bytes_hex, expected in TEST_CASES:
    pred = predict(bytes_hex)
    # Normalize (strip l/q/w/b suffix)
    pred_norm = pred.rstrip("lqwb") if pred else ""
    expected_norm = expected.rstrip("lqwb")
    
    match = pred_norm == expected_norm
    if match:
        correct += 1
    
    status = "✓" if match else "✗"
    print(f"  {bytes_hex:15} → {pred:10} (expected: {expected}) {status}")

accuracy = correct / total * 100
print(f"\n  Verification: {correct}/{total} = {accuracy:.1f}%")

# ============================================================================
# PACKAGE
# ============================================================================
print("\n[PACKAGE]")

import zipfile
with zipfile.ZipFile("level0_ultimate.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    model_dir = Path("models/level0_ultimate")
    for f in model_dir.iterdir():
        zf.write(f, f"models/level0_ultimate/{f.name}")

size_mb = Path("level0_ultimate.zip").stat().st_size / 1024 / 1024
print(f"  Created: level0_ultimate.zip ({size_mb:.1f} MB)")

try:
    from google.colab import files
    files.download("level0_ultimate.zip")
except:
    print(f"  Download: {os.path.abspath('level0_ultimate.zip')}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
