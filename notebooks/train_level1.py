#!/usr/bin/env python3
"""
Level 1: Instruction Semantics Training

Input: instruction bytes
Output: Multi-label classification for register reads/writes + binary flags

Architecture:
- Byte-level transformer encoder (reuse Level 0 architecture)
- Multiple classification heads:
  - reads: multi-label over 32 registers
  - writes: multi-label over 32 registers
  - mem_read: binary
  - mem_write: binary
  - flags_written: binary

Ground truth: Capstone detail mode
"""

import os
import subprocess
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("LEVEL 1: INSTRUCTION SEMANTICS")
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
from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86 import X86_OP_MEM, X86_OP_REG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# =============================================================================
# REGISTER VOCABULARY
# =============================================================================

# All x86-64 registers we care about
REGISTERS = [
    # 64-bit general purpose
    'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    # 32-bit (lower halves)
    'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
    'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
    # 16-bit
    'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
    # 8-bit
    'al', 'bl', 'cl', 'dl', 'sil', 'dil', 'bpl', 'spl',
    'ah', 'bh', 'ch', 'dh',
    'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b',
    # Special
    'rip', 'rflags',
    # XMM (SSE)
    'xmm0', 'xmm1', 'xmm2', 'xmm3', 'xmm4', 'xmm5', 'xmm6', 'xmm7',
    'xmm8', 'xmm9', 'xmm10', 'xmm11', 'xmm12', 'xmm13', 'xmm14', 'xmm15',
]

REG_TO_IDX = {reg: i for i, reg in enumerate(REGISTERS)}
NUM_REGISTERS = len(REGISTERS)
print(f"  Registers: {NUM_REGISTERS}")

# =============================================================================
# GROUND TRUTH EXTRACTION
# =============================================================================

def get_instruction_semantics(insn) -> dict:
    """Extract ground truth semantics from Capstone instruction.
    
    Uses op.access field for accurate read/write detection:
        1 = CS_AC_READ
        2 = CS_AC_WRITE
        3 = CS_AC_READ | CS_AC_WRITE
    """
    # Start with implicit registers from Capstone
    reads = set(insn.reg_name(r) for r in insn.regs_read)
    writes = set(insn.reg_name(r) for r in insn.regs_write)
    mem_read = False
    mem_write = False
    
    CS_AC_READ = 1
    CS_AC_WRITE = 2
    
    for op in insn.operands:
        if op.type == X86_OP_REG:
            reg = insn.reg_name(op.reg)
            # Use access field instead of position guessing
            if op.access & CS_AC_READ:
                reads.add(reg)
            if op.access & CS_AC_WRITE:
                writes.add(reg)
        elif op.type == X86_OP_MEM:
            # Base and index registers are always read for address calculation
            if op.mem.base:
                reads.add(insn.reg_name(op.mem.base))
            if op.mem.index:
                reads.add(insn.reg_name(op.mem.index))
            # Memory access direction
            if op.access & CS_AC_READ:
                mem_read = True
            if op.access & CS_AC_WRITE:
                mem_write = True
    
    # Special cases for implicit memory access
    if insn.mnemonic == 'push':
        mem_write = True
    elif insn.mnemonic in ['pop', 'ret']:
        mem_read = True
    elif insn.mnemonic == 'call':
        mem_write = True
    
    flags_written = insn.eflags != 0
    writes.discard('rflags')
    
    return {
        'reads': frozenset(r for r in reads if r in REG_TO_IDX),
        'writes': frozenset(w for w in writes if w in REG_TO_IDX),
        'mem_read': mem_read,
        'mem_write': mem_write,
        'flags_written': flags_written,
    }


def get_text_section(binary_path):
    """Extract .text section from ELF binary."""
    try:
        from elftools.elf.elffile import ELFFile
        with open(binary_path, 'rb') as f:
            elf = ELFFile(f)
            for section in elf.iter_sections():
                if section.name == '.text':
                    return section.data()
    except:
        pass
    
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


def disassemble_with_semantics(binary_path):
    """Disassemble binary and extract instruction semantics."""
    text = get_text_section(binary_path)
    if not text:
        return []
    
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    
    samples = []
    for insn in md.disasm(text, 0):
        sem = get_instruction_semantics(insn)
        samples.append((list(insn.bytes), insn.mnemonic, sem))
    
    return samples


# =============================================================================
# DATA COLLECTION
# =============================================================================
print("\n[DATA COLLECTION]")

# Training programs - DIFFERENT from gate test programs
TRAIN_PROGRAMS = {
    "basic": "int main(){return 0;}",
    "loop": "int main(){int s=0;for(int i=0;i<100;i++)s+=i;return s;}",
    "func": "int f(int x){return x*2+1;}int main(){return f(10);}",
    "ptr": "void swap(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;swap(&x,&y);return x;}",
    "arr": "int sum(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3,4,5};return sum(a,5);}",
    "str": "int len(char*s){int n=0;while(s[n])n++;return n;}int main(){return len(\"hello\");}",
    "cond": "int max(int a,int b){return a>b?a:b;}int main(){return max(5,3);}",
    "switch": "int f(int x){switch(x){case 0:return 10;case 1:return 20;default:return 0;}}int main(){return f(1);}",
    "rec": "int fact(int n){return n<=1?1:n*fact(n-1);}int main(){return fact(5);}",
    "bitops": "int pop(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return pop(255);}",
    "fptr": "int add(int a,int b){return a+b;}int main(){int(*f)(int,int)=add;return f(3,2);}",
    "struct": "struct P{int x,y;};int dist(struct P*p){return p->x*p->x+p->y*p->y;}int main(){struct P p={3,4};return dist(&p);}",
}

print("  Finding system binaries...")
binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin"]:
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
                if len(binaries) >= 100:
                    break
        except:
            pass
    if len(binaries) >= 100:
        break

print(f"    Found {len(binaries)} system binaries")

print("  Compiling training programs...")
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    for name, code in TRAIN_PROGRAMS.items():
        for compiler in ["gcc", "clang"]:
            for opt in ["-O0", "-O1", "-O2", "-O3"]:
                src = tmpdir / f"{name}.c"
                binary = tmpdir / f"{name}_{compiler}_{opt}"
                src.write_text(code)
                if subprocess.run([compiler, opt, "-w", "-o", str(binary), str(src)],
                                 capture_output=True, timeout=30).returncode == 0:
                    binaries.append(binary)

print(f"    Total: {len(binaries)} binaries")

print("  Extracting instruction semantics...")
all_samples = []
for i, binary in enumerate(binaries):
    samples = disassemble_with_semantics(binary)
    all_samples.extend(samples)
    if (i + 1) % 30 == 0:
        print(f"    Processed {i+1}/{len(binaries)}, {len(all_samples)} samples")

print(f"  Total samples: {len(all_samples)}")

# Deduplicate by (bytes, mnemonic) -> keep first semantics
unique_samples = {}
for bytes_list, mnemonic, sem in all_samples:
    key = (tuple(bytes_list), mnemonic)
    if key not in unique_samples:
        unique_samples[key] = (bytes_list, mnemonic, sem)

samples = list(unique_samples.values())
print(f"  Unique samples: {len(samples)}")

# =============================================================================
# DATASET
# =============================================================================
print("\n[DATASET]")

MAX_LEN = 15

class SemanticsDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bytes_list, mnemonic, sem = self.samples[idx]
        
        # Pad bytes
        if len(bytes_list) < MAX_LEN:
            bytes_list = bytes_list + [0] * (MAX_LEN - len(bytes_list))
        bytes_tensor = torch.tensor(bytes_list[:MAX_LEN], dtype=torch.long)
        
        # Multi-label for reads
        reads_vec = torch.zeros(NUM_REGISTERS, dtype=torch.float)
        for reg in sem['reads']:
            if reg in REG_TO_IDX:
                reads_vec[REG_TO_IDX[reg]] = 1.0
        
        # Multi-label for writes
        writes_vec = torch.zeros(NUM_REGISTERS, dtype=torch.float)
        for reg in sem['writes']:
            if reg in REG_TO_IDX:
                writes_vec[REG_TO_IDX[reg]] = 1.0
        
        # Binary flags
        mem_read = torch.tensor([1.0 if sem['mem_read'] else 0.0])
        mem_write = torch.tensor([1.0 if sem['mem_write'] else 0.0])
        flags = torch.tensor([1.0 if sem['flags_written'] else 0.0])
        
        return bytes_tensor, reads_vec, writes_vec, mem_read, mem_write, flags


# Split
random.shuffle(samples)
split = int(0.9 * len(samples))
train_samples = samples[:split]
val_samples = samples[split:]

train_dataset = SemanticsDataset(train_samples)
val_dataset = SemanticsDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# =============================================================================
# MODEL
# =============================================================================
print("\n[MODEL]")

class Level1Classifier(nn.Module):
    """Multi-head classifier for instruction semantics."""
    
    def __init__(self, num_registers, max_len=MAX_LEN, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        
        # Shared encoder (same as Level 0)
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification heads
        self.reads_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_registers),
        )
        self.writes_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_registers),
        )
        self.mem_read_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.mem_write_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.flags_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.byte_embed(x) + self.pos_embed(positions)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        
        return {
            'reads': self.reads_head(pooled),
            'writes': self.writes_head(pooled),
            'mem_read': self.mem_read_head(pooled),
            'mem_write': self.mem_write_head(pooled),
            'flags': self.flags_head(pooled),
        }


model = Level1Classifier(NUM_REGISTERS).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

# =============================================================================
# TRAINING
# =============================================================================
print("\n[TRAINING]")

# Loss functions
bce_logits = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    
    for batch in loader:
        bytes_t, reads_t, writes_t, mem_r, mem_w, flags = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        out = model(bytes_t)
        
        loss = (
            bce_logits(out['reads'], reads_t) +
            bce_logits(out['writes'], writes_t) +
            bce_logits(out['mem_read'], mem_r) +
            bce_logits(out['mem_write'], mem_w) +
            bce_logits(out['flags'], flags)
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    
    metrics = {
        'reads_correct': 0, 'reads_total': 0,
        'writes_correct': 0, 'writes_total': 0,
        'mem_read_correct': 0, 'mem_write_correct': 0,
        'flags_correct': 0, 'total': 0,
    }
    
    with torch.no_grad():
        for batch in loader:
            bytes_t, reads_t, writes_t, mem_r, mem_w, flags = [b.to(device) for b in batch]
            out = model(bytes_t)
            
            # Multi-label accuracy (exact match)
            reads_pred = (torch.sigmoid(out['reads']) > 0.5).float()
            writes_pred = (torch.sigmoid(out['writes']) > 0.5).float()
            
            metrics['reads_correct'] += (reads_pred == reads_t).all(dim=1).sum().item()
            metrics['writes_correct'] += (writes_pred == writes_t).all(dim=1).sum().item()
            
            # Binary accuracy
            metrics['mem_read_correct'] += ((torch.sigmoid(out['mem_read']) > 0.5).float() == mem_r).sum().item()
            metrics['mem_write_correct'] += ((torch.sigmoid(out['mem_write']) > 0.5).float() == mem_w).sum().item()
            metrics['flags_correct'] += ((torch.sigmoid(out['flags']) > 0.5).float() == flags).sum().item()
            
            metrics['total'] += bytes_t.size(0)
    
    total = metrics['total']
    return {
        'reads': metrics['reads_correct'] / total,
        'writes': metrics['writes_correct'] / total,
        'mem_read': metrics['mem_read_correct'] / total,
        'mem_write': metrics['mem_write_correct'] / total,
        'flags': metrics['flags_correct'] / total,
    }


best_val_acc = 0
patience, wait = 30, 0
best_state = None

print("  Training...")
for epoch in range(200):
    train_loss = train_epoch(model, train_loader)
    val_acc = eval_epoch(model, val_loader)
    scheduler.step()
    
    avg_acc = sum(val_acc.values()) / len(val_acc)
    
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={train_loss:.4f} val_avg={avg_acc:.4f} *")
    else:
        wait += 1
    
    if wait >= patience and epoch >= 30:
        print(f"    Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
model.to(device)

print(f"\n  Best validation accuracy (avg): {best_val_acc:.4f}")

# Final validation breakdown
final_acc = eval_epoch(model, val_loader)
print("  Per-field validation:")
for field, acc in final_acc.items():
    print(f"    {field:12}: {acc*100:.2f}%")

# =============================================================================
# SAVE
# =============================================================================
print("\n[SAVE]")

os.makedirs("models/level1", exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'registers': REGISTERS,
    'reg_to_idx': REG_TO_IDX,
    'num_registers': NUM_REGISTERS,
    'max_len': MAX_LEN,
}, "models/level1/model.pt")

with open("models/level1/config.json", "w") as f:
    json.dump({
        'num_registers': NUM_REGISTERS,
        'max_len': MAX_LEN,
        'registers': REGISTERS,
    }, f, indent=2)

import zipfile
with zipfile.ZipFile("level1.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level1").iterdir():
        zf.write(f, f"models/level1/{f.name}")

print(f"  Created: level1.zip")

try:
    import shutil
    os.makedirs("/content/drive/MyDrive/genesis_models", exist_ok=True)
    shutil.copy("level1.zip", "/content/drive/MyDrive/genesis_models/")
    print("  Saved to Google Drive")
except:
    pass

try:
    from google.colab import files
    files.download("level1.zip")
except:
    print(f"  Download: {os.path.abspath('level1.zip')}")

print("\n" + "=" * 60)
print("LEVEL 1 TRAINING COMPLETE")
print("=" * 60)
