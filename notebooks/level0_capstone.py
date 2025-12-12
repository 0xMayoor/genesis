#!/usr/bin/env python3
"""
LEVEL 0: CAPSTONE-ONLY

Use Capstone as ground truth (not objdump which has bugs on Colab).
Read ELF .text section directly and disassemble with Capstone.
"""

import os
import subprocess
import sys
import json
import random
import re
import struct
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("LEVEL 0: CAPSTONE GROUND TRUTH")
print("=" * 60)

# Setup
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "binutils", "gcc", "clang"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "capstone", "pyelftools", "numpy"], capture_output=True)

if "genesis" not in os.getcwd():
    subprocess.run(["rm", "-rf", "genesis"], capture_output=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/0xMayoor/genesis.git"])
    os.chdir("genesis")

print(f"\n[SETUP]")
print(f"  Dir: {os.getcwd()}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ============================================================================
# CAPSTONE-BASED DISASSEMBLY
# ============================================================================

cs = Cs(CS_ARCH_X86, CS_MODE_64)

def get_text_section(binary_path):
    """Extract .text section from ELF binary using pyelftools."""
    try:
        from elftools.elf.elffile import ELFFile
        with open(binary_path, 'rb') as f:
            elf = ELFFile(f)
            for section in elf.iter_sections():
                if section.name == '.text':
                    return section.data()
    except:
        pass
    
    # Fallback: use objcopy
    try:
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            result = subprocess.run(
                ["objcopy", "-O", "binary", "--only-section=.text", 
                 str(binary_path), tmp.name],
                capture_output=True, timeout=30
            )
            if result.returncode == 0:
                with open(tmp.name, 'rb') as f:
                    return f.read()
    except:
        pass
    
    return None

def disassemble_with_capstone(binary_path):
    """Disassemble using Capstone (correct ground truth)."""
    text = get_text_section(binary_path)
    if not text:
        return []
    
    samples = []
    for instr in cs.disasm(text, 0):
        mnemonic = instr.mnemonic.lower()
        bytes_list = list(instr.bytes)
        if bytes_list and mnemonic:
            samples.append((bytes_list, mnemonic))
    return samples

# Valid x86-64 mnemonics (for filtering)
VALID_PREFIXES = {'push', 'pop', 'mov', 'lea', 'add', 'sub', 'xor', 'and', 'or', 'not',
                  'cmp', 'test', 'jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jae', 'jb', 'jbe',
                  'jg', 'jge', 'jl', 'jle', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp',
                  'call', 'ret', 'nop', 'int', 'syscall', 'endbr64', 'endbr32',
                  'imul', 'mul', 'idiv', 'div', 'neg', 'inc', 'dec',
                  'shl', 'shr', 'sar', 'sal', 'rol', 'ror', 'rcl', 'rcr',
                  'bt', 'bts', 'btr', 'btc', 'bsf', 'bsr', 'bswap',
                  'cmov', 'set', 'xchg', 'cmpxchg', 'xadd',
                  'cdq', 'cdqe', 'cqo', 'cbw', 'cwde', 'cwd', 'cltq',
                  'leave', 'enter', 'hlt', 'wait', 'lock', 'rep', 'repn',
                  'lods', 'stos', 'movs', 'cmps', 'scas',
                  'in', 'out', 'cpuid', 'rdtsc',
                  'fld', 'fst', 'fadd', 'fsub', 'fmul', 'fdiv',
                  'movss', 'movsd', 'movaps', 'movups', 'movapd', 'movupd',
                  'addss', 'addsd', 'addps', 'addpd', 'subss', 'subsd',
                  'mulss', 'mulsd', 'divss', 'divsd', 'sqrtss', 'sqrtsd',
                  'xorps', 'xorpd', 'andps', 'andpd', 'orps', 'orpd',
                  'cmpss', 'cmpsd', 'cmpps', 'cmppd', 'comiss', 'comisd',
                  'cvt', 'pxor', 'por', 'pand', 'pandn', 'padd', 'psub', 'pmul',
                  'punpck', 'pack', 'pshuf', 'shuf', 'blend', 'ptest',
                  'aes', 'pclmul', 'crc32', 'popcnt', 'lzcnt', 'tzcnt',
                  'prefetch', 'lfence', 'sfence', 'mfence', 'clflush',
                  'ud2', 'data16'}

def is_valid_mnemonic(m):
    m = m.lower()
    if not m or len(m) > 15:
        return False
    for prefix in VALID_PREFIXES:
        if m.startswith(prefix):
            return True
    return m in VALID_PREFIXES

# ============================================================================
# DATA COLLECTION
# ============================================================================
print("\n[DATA COLLECTION]")

# Find system binaries
print("  Finding system binaries...")
binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin", "/usr/lib/x86_64-linux-gnu"]:
    if Path(path).exists():
        try:
            for f in Path(path).iterdir():
                if f.is_file() and not f.is_symlink():
                    try:
                        with open(f, "rb") as fp:
                            if fp.read(4) == b'\x7fELF':
                                binaries.append(f)
                    except:
                        pass
                if len(binaries) >= 200:
                    break
        except:
            pass
    if len(binaries) >= 200:
        break

print(f"    Found {len(binaries)} system binaries")

# Compile programs
print("  Compiling programs...")
PROGRAMS = {
    "basic": "int main(){return 0;}",
    "loop": "int main(){int s=0;for(int i=0;i<100;i++)s+=i;return s;}",
    "func": "int f(int x){return x*2+1;}int main(){return f(10);}",
    "fptr": "int add(int a,int b){return a+b;}int main(){int(*f)(int,int)=add;return f(3,2);}",
    "indirect": "int f1(){return 1;}int f2(){return 2;}int main(){int(*t[])()={f1,f2};return t[0]()+t[1]();}",
    "tls": "__thread int tls_var=42;int main(){return tls_var;}",
    "ptr": "void swap(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;swap(&x,&y);return x;}",
    "arr": "int sum(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3,4,5};return sum(a,5);}",
    "str": "int len(char*s){int n=0;while(s[n])n++;return n;}int main(){return len(\"hello\");}",
    "cond": "int max(int a,int b){return a>b?a:b;}int main(){return max(5,3);}",
    "switch": "int f(int x){switch(x){case 0:return 10;case 1:return 20;case 2:return 30;default:return 0;}}int main(){return f(1);}",
    "rec": "int fib(int n){return n<=1?n:fib(n-1)+fib(n-2);}int main(){return fib(10);}",
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
                    binaries.append(binary)
                    compiled.append(binary)

print(f"    Compiled {len(compiled)} additional binaries")
print(f"    Total: {len(binaries)}")

# Extract using Capstone
print("  Extracting with Capstone...")
all_samples = []
for i, binary in enumerate(binaries):
    samples = disassemble_with_capstone(binary)
    all_samples.extend(samples)
    if (i + 1) % 50 == 0:
        print(f"    Processed {i+1}/{len(binaries)} binaries, {len(all_samples)} samples")

print(f"    Total samples: {len(all_samples)}")

# Filter to valid mnemonics
mnemonic_counts = defaultdict(int)
for _, mnemonic in all_samples:
    if is_valid_mnemonic(mnemonic):
        mnemonic_counts[mnemonic] += 1

valid_mnemonics = {m for m, c in mnemonic_counts.items() if c >= 10}
print(f"    Valid mnemonics: {len(valid_mnemonics)}")

mnemonic_to_id = {m: i for i, m in enumerate(sorted(valid_mnemonics))}
id_to_mnemonic = {i: m for m, i in mnemonic_to_id.items()}
num_classes = len(mnemonic_to_id)

samples = [(b, m) for b, m in all_samples if m in valid_mnemonics]
print(f"    Filtered samples: {len(samples)}")

# Show sample mnemonics
print(f"    Mnemonics: {list(mnemonic_to_id.keys())[:30]}...")

# Verify critical patterns are correct
print("\n  Verifying critical patterns in training data:")
critical = {'push': [], 'pop': [], 'jmp': [], 'mov': [], 'ret': []}
for bytes_list, mnemonic in samples[:100000]:
    if mnemonic in critical and len(critical[mnemonic]) < 5:
        critical[mnemonic].append(bytes(bytes_list).hex())
for m, examples in critical.items():
    print(f"    {m}: {examples[:3]}")

# ADD CRITICAL PATTERNS EXPLICITLY (patterns that are often missing)
print("\n  Adding critical patterns...")
CRITICAL_PATTERNS = [
    # Indirect jumps (ff /4 = ffe0-ffe7)
    ([0xff, 0xe0], 'jmp'),  # jmp rax
    ([0xff, 0xe1], 'jmp'),  # jmp rcx
    ([0xff, 0xe2], 'jmp'),  # jmp rdx
    ([0xff, 0xe3], 'jmp'),  # jmp rbx
    ([0xff, 0xe4], 'jmp'),  # jmp rsp
    ([0xff, 0xe5], 'jmp'),  # jmp rbp
    ([0xff, 0xe6], 'jmp'),  # jmp rsi
    ([0xff, 0xe7], 'jmp'),  # jmp rdi
    ([0x41, 0xff, 0xe0], 'jmp'),  # jmp r8
    ([0x41, 0xff, 0xe1], 'jmp'),  # jmp r9
    # Indirect calls (ff /2 = ffd0-ffd7)
    ([0xff, 0xd0], 'call'),  # call rax
    ([0xff, 0xd1], 'call'),  # call rcx
    ([0xff, 0xd2], 'call'),  # call rdx
    # FS segment prefix instructions (64)
    ([0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00], 'mov'),  # mov rax, fs:[0x28]
    ([0x64, 0x48, 0x2b, 0x14, 0x25, 0x28, 0x00, 0x00, 0x00], 'sub'),  # sub rdx, fs:[0x28]
    ([0x64, 0x48, 0x8b, 0x0c, 0x25, 0x00, 0x00, 0x00, 0x00], 'mov'),  # mov rcx, fs:[0]
    ([0x64, 0x48, 0x89, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00], 'mov'),  # mov fs:[0x28], rax
    # GS segment prefix (65)
    ([0x65, 0x48, 0x8b, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00], 'mov'),  # mov rax, gs:[0x28]
    # Push variants (make sure all are covered)
    ([0x50], 'push'), ([0x51], 'push'), ([0x52], 'push'), ([0x53], 'push'),
    ([0x54], 'push'), ([0x55], 'push'), ([0x56], 'push'), ([0x57], 'push'),
    # Pop variants
    ([0x58], 'pop'), ([0x59], 'pop'), ([0x5a], 'pop'), ([0x5b], 'pop'),
    ([0x5c], 'pop'), ([0x5d], 'pop'), ([0x5e], 'pop'), ([0x5f], 'pop'),
]

# Add each critical pattern 100 times to ensure it's learned
for bytes_list, mnemonic in CRITICAL_PATTERNS:
    if mnemonic in valid_mnemonics:  # Only if mnemonic is in vocabulary
        for _ in range(100):
            samples.append((bytes_list, mnemonic))

print(f"    Added {len(CRITICAL_PATTERNS) * 100} critical samples")
print(f"    Total samples now: {len(samples)}")

# ============================================================================
# BALANCE & DATASET
# ============================================================================
print("\n[BALANCING]")

by_mnemonic = defaultdict(list)
for bytes_list, mnemonic in samples:
    by_mnemonic[mnemonic].append(bytes_list)

MIN_SAMPLES = 50
MAX_SAMPLES = 500

balanced = []
for mnemonic, byte_lists in by_mnemonic.items():
    unique = list(set(tuple(b) for b in byte_lists))
    if len(unique) < MIN_SAMPLES:
        unique = unique * ((MIN_SAMPLES // len(unique)) + 1)
    unique = unique[:MAX_SAMPLES]
    for b in unique:
        balanced.append((list(b), mnemonic))

random.shuffle(balanced)
print(f"  Balanced: {len(balanced)}")

split = int(0.9 * len(balanced))
train_samples = balanced[:split]
val_samples = balanced[split:]
print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# Dataset
MAX_LEN = 15

class ByteDataset(Dataset):
    def __init__(self, samples, m2id):
        self.samples = samples
        self.m2id = m2id
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        b, m = self.samples[idx]
        b = b + [0] * (MAX_LEN - len(b)) if len(b) < MAX_LEN else b[:MAX_LEN]
        return torch.tensor(b, dtype=torch.long), torch.tensor(self.m2id[m], dtype=torch.long)

train_loader = DataLoader(ByteDataset(train_samples, mnemonic_to_id), batch_size=128, shuffle=True)
val_loader = DataLoader(ByteDataset(val_samples, mnemonic_to_id), batch_size=128)

# ============================================================================
# MODEL
# ============================================================================
print("\n[MODEL]")

class ByteClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(MAX_LEN, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(embed_dim, num_classes))
    
    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        emb = self.byte_embed(x) + self.pos_embed(pos)
        return self.classifier(self.transformer(emb).mean(dim=1))

model = ByteClassifier(num_classes).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n[TRAINING]")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_val_acc = 0
patience, wait = 30, 0
best_state = None

for epoch in range(300):
    model.train()
    train_correct, train_total = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_correct += (model(x).argmax(1) == y).sum().item()
        train_total += y.size(0)
    
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_correct += (model(x).argmax(1) == y).sum().item()
            val_total += y.size(0)
    
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
        if (epoch + 1) % 10 == 0 or val_acc > 0.995:
            print(f"    Epoch {epoch+1}: train={train_acc:.4f} val={val_acc:.4f} *")
    else:
        wait += 1
    
    if wait >= patience and epoch >= 30:
        print(f"    Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
model.to(device)
print(f"\n  Best: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# ============================================================================
# GATE TEST (also using Capstone)
# ============================================================================
print("\n[GATE TEST]")

GATE_PROGRAMS = {
    "reverse": 'void reverse(char*s,int len){for(int i=0;i<len/2;i++){char t=s[i];s[i]=s[len-1-i];s[len-1-i]=t;}}int main(){char s[]="hello";reverse(s,5);return s[0];}',
    "popcount": 'int popcount(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return popcount(0xFF);}',
    "max": 'int max(int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];return m;}int main(){int a[]={3,1,4,1,5,9};return max(a,6);}',
}

def predict(bytes_list):
    b = bytes_list + [0] * (MAX_LEN - len(bytes_list)) if len(bytes_list) < MAX_LEN else bytes_list[:MAX_LEN]
    x = torch.tensor([b], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        return id_to_mnemonic[model(x).argmax(1).item()]

correct, total, unknown = 0, 0, 0
errors = []

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for name, code in GATE_PROGRAMS.items():
        for compiler in ["gcc", "clang"]:
            for opt in ["-O0", "-O1", "-O2", "-O3"]:
                src = tmpdir / f"{name}.c"
                binary = tmpdir / f"{name}_{compiler}_{opt}"
                src.write_text(code)
                if subprocess.run([compiler, opt, "-w", "-o", str(binary), str(src)],
                                 capture_output=True).returncode != 0:
                    continue
                
                # Use Capstone for test too
                text = get_text_section(binary)
                if not text:
                    continue
                
                for instr in cs.disasm(text, 0):
                    expected = instr.mnemonic.lower()
                    bytes_list = list(instr.bytes)
                    
                    if expected not in mnemonic_to_id:
                        unknown += 1
                        continue
                    
                    pred = predict(bytes_list)
                    total += 1
                    
                    if pred == expected:
                        correct += 1
                    elif len(errors) < 20:
                        errors.append(f"{instr.bytes.hex()}: {pred} vs {expected}")

accuracy = 100 * correct / total if total > 0 else 0
print(f"\n  Gate Test: {correct}/{total} = {accuracy:.1f}%")
print(f"  Unknown: {unknown}")

if errors:
    print(f"\n  Errors:")
    for e in errors[:10]:
        print(f"    {e}")

# ============================================================================
# KNOWN PATTERNS TEST
# ============================================================================
print("\n[KNOWN PATTERNS TEST]")
known = [
    ('c3', 'ret'), ('55', 'push'), ('5d', 'pop'), ('50', 'push'), ('54', 'push'),
    ('4889e5', 'mov'), ('4883ec20', 'sub'), ('4883c420', 'add'),
    ('31c0', 'xor'), ('e800000000', 'call'), ('ffe0', 'jmp'),
    ('0fb6c0', 'movzx'), ('f30f1efa', 'endbr64'), ('90', 'nop'),
]

kp_correct = 0
for hex_bytes, expected in known:
    bytes_list = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
    pred = predict(bytes_list)
    ok = pred == expected
    kp_correct += int(ok)
    status = '✓' if ok else '✗'
    print(f"  {hex_bytes:15} -> {pred:10} (exp: {expected:10}) {status}")

print(f"\n  Known patterns: {kp_correct}/{len(known)} = {100*kp_correct/len(known):.0f}%")

# ============================================================================
# SAVE
# ============================================================================
print("\n[SAVE]")

MODEL_DIR = "models/level0"
os.makedirs(MODEL_DIR, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'mnemonic_to_id': mnemonic_to_id,
    'id_to_mnemonic': id_to_mnemonic,
    'num_classes': num_classes,
    'max_len': MAX_LEN,
}, f"{MODEL_DIR}/model.pt")

with open(f"{MODEL_DIR}/config.json", "w") as f:
    json.dump({'num_classes': num_classes, 'max_len': MAX_LEN,
               'mnemonic_to_id': mnemonic_to_id,
               'id_to_mnemonic': {str(k): v for k, v in id_to_mnemonic.items()}}, f, indent=2)

import zipfile
with zipfile.ZipFile("level0.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path(MODEL_DIR).iterdir():
        zf.write(f, f"{MODEL_DIR}/{f.name}")

print(f"  Created: level0.zip")

try:
    import shutil
    os.makedirs("/content/drive/MyDrive/genesis_models", exist_ok=True)
    shutil.copy("level0.zip", "/content/drive/MyDrive/genesis_models/")
    print("  Saved to Google Drive")
except:
    pass

try:
    from google.colab import files
    files.download("level0.zip")
except:
    print(f"  Download: {os.path.abspath('level0.zip')}")

print("\n" + "=" * 60)
print(f"GATE TEST: {accuracy:.1f}%")
print(f"KNOWN PATTERNS: {100*kp_correct/len(known):.0f}%")
print(f"Classes: {num_classes}")
print("=" * 60)
