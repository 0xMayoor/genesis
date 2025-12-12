#!/usr/bin/env python3
"""
LEVEL 0: 100% ACCURACY

Same as level0_classifier.py but with:
1. More diverse C programs covering edge cases
2. Explicit rare patterns (segment prefixes, indirect jumps)
3. Larger training set
"""

import os
import subprocess
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("LEVEL 0: TARGET 100%")
print("=" * 60)

# Setup
subprocess.run(["apt-get", "update"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "binutils", "gcc", "clang"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "capstone", "numpy"], capture_output=True)

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
# DATA COLLECTION - EXPANDED
# ============================================================================
print("\n[DATA COLLECTION]")

def disassemble(binary_path):
    """Disassemble using objdump (more reliable on Colab)."""
    import re
    try:
        result = subprocess.run(
            ["objdump", "-d", "-M", "intel", str(binary_path)],
            capture_output=True, text=True, timeout=60
        )
        if not result.stdout:
            return []
        
        samples = []
        for line in result.stdout.split('\n'):
            # Match: "  addr:    bytes      mnemonic operands"
            match = re.match(r'\s+[0-9a-f]+:\s+([0-9a-f ]+?)\s{2,}(\S+)', line)
            if match:
                bytes_hex = match.group(1).strip().replace(' ', '')
                mnemonic = match.group(2).lower()
                # Convert hex string to byte list
                bytes_list = [int(bytes_hex[i:i+2], 16) for i in range(0, len(bytes_hex), 2)]
                if bytes_list and mnemonic and not mnemonic.startswith('.'):
                    samples.append((bytes_list, mnemonic))
        return samples
    except Exception as e:
        return []

# Find MORE system binaries
print("  Finding system binaries...")
binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/lib/x86_64-linux-gnu",
             "/usr/lib", "/lib/x86_64-linux-gnu"]:
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
                if len(binaries) >= 500:  # More binaries
                    break
        except:
            pass
    if len(binaries) >= 500:
        break

print(f"    Found {len(binaries)} system binaries")

# EXPANDED C programs covering edge cases
print("  Compiling programs...")
PROGRAMS = {
    # Basic
    "basic": "int main(){return 0;}",
    "loop": "int main(){int s=0;for(int i=0;i<100;i++)s+=i;return s;}",
    "func": "int f(int x){return x*2+1;}int g(int x){return f(x)+f(x-1);}int main(){return g(10);}",
    
    # Pointers and indirect calls/jumps (for ffe0 etc)
    "fptr": "int add(int a,int b){return a+b;}int sub(int a,int b){return a-b;}int main(){int(*f)(int,int)=add;return f(3,2);}",
    "vtable": "typedef struct{int(*op)(int);}VT;int dbl(int x){return x*2;}int main(){VT v={dbl};return v.op(5);}",
    "indirect": "int f1(){return 1;}int f2(){return 2;}int main(){int(*t[])()={f1,f2};return t[0]()+t[1]();}",
    "callback": "int apply(int(*f)(int),int x){return f(x);}int inc(int x){return x+1;}int main(){return apply(inc,5);}",
    "jumptbl": "int f(int x){switch(x){case 0:return 10;case 1:return 20;case 2:return 30;case 3:return 40;case 4:return 50;default:return 0;}}int main(){return f(2);}",
    
    # Thread-local storage (for segment prefixes 64/65)
    "tls": "__thread int tls_var=42;int main(){return tls_var;}",
    "tls2": "__thread int a=1,b=2;int main(){return a+b;}",
    "tls3": "__thread char buf[64];int main(){buf[0]='A';return buf[0];}",
    
    # More diverse operations
    "ptr": "void swap(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;swap(&x,&y);return x;}",
    "arr": "int sum(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3,4,5};return sum(a,5);}",
    "str": "int len(char*s){int n=0;while(s[n])n++;return n;}int main(){return len(\"hello\");}",
    "cond": "int max(int a,int b){return a>b?a:b;}int min(int a,int b){return a<b?a:b;}int main(){return max(5,3)+min(5,3);}",
    "rec": "int fib(int n){return n<=1?n:fib(n-1)+fib(n-2);}int main(){return fib(10);}",
    "math": "int main(){int a=100,b=37;return ((a+b)*(a-b))/(a%b+1);}",
    "bit": "unsigned rev(unsigned n){unsigned r=0;for(int i=0;i<32;i++){r=(r<<1)|(n&1);n>>=1;}return r;}int main(){return rev(0x12345678)&0xFF;}",
    "mem": "void cpy(char*d,char*s,int n){while(n--)*d++=*s++;}int main(){char a[8],b[]=\"test\";cpy(a,b,5);return a[0];}",
    "struct": "struct P{int x,y;};int dist(struct P*p){return p->x*p->x+p->y*p->y;}int main(){struct P p={3,4};return dist(&p);}",
    "float": "int main(){float x=3.14f;double y=2.71;return (int)(x*y);}",
    "simd": "int main(){int a[4]={1,2,3,4},s=0;for(int i=0;i<4;i++)s+=a[i];return s;}",
    
    # String and memory operations (various prefixes)
    "memcpy": "void*mcpy(void*d,void*s,int n){char*dp=d,*sp=s;while(n--)*dp++=*sp++;return d;}int main(){char a[8],b[]=\"hi\";mcpy(a,b,3);return a[0];}",
    "memset": "void*mset(void*s,int c,int n){char*p=s;while(n--)*p++=c;return s;}int main(){char b[4];mset(b,'X',4);return b[0];}",
    "strcmp": "int scmp(char*a,char*b){while(*a&&*a==*b){a++;b++;}return *a-*b;}int main(){return scmp(\"abc\",\"abd\");}",
    
    # Signed/unsigned operations
    "signed": "int main(){signed char a=-1;signed int b=-100;return a+b;}",
    "unsigned": "int main(){unsigned char a=255;unsigned int b=0xFFFFFFFF;return (a^(b&0xFF));}",
    "mixed": "int main(){signed int s=-5;unsigned int u=10;return s+u;}",
    
    # More control flow
    "multiret": "int f(int x){if(x<0)return -1;if(x==0)return 0;if(x<10)return 1;return 2;}int main(){return f(5)+f(-1)+f(0)+f(100);}",
    "nested": "int f(int a,int b,int c){if(a>0){if(b>0){if(c>0)return 1;return 2;}return 3;}return 4;}int main(){return f(1,1,1);}",
    "dowhile": "int main(){int i=0,s=0;do{s+=i;i++;}while(i<5);return s;}",
    
    # Additional edge cases
    "lea": "int main(){int a[10];int*p=&a[5];return p-a;}",
    "movsx": "int main(){signed char c=-1;return (int)c;}",
    "movzx": "int main(){unsigned char c=255;return (int)c;}",
    "bswap": "int main(){unsigned x=0x12345678;return __builtin_bswap32(x)&0xFF;}",
    "popcnt": "int main(){return __builtin_popcount(0xFF);}",
    "clz": "int main(){return __builtin_clz(16);}",
    "ctz": "int main(){return __builtin_ctz(16);}",
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
                result = subprocess.run(
                    [compiler, opt, "-w", "-o", str(binary), str(src)],
                    capture_output=True, timeout=30
                )
                if result.returncode == 0:
                    compiled.append(binary)
                    binaries.append(binary)

print(f"    Compiled {len(compiled)} additional binaries")
print(f"    Total binaries: {len(binaries)}")

# Extract all patterns
print("  Extracting instructions...")
all_samples = []

for binary in binaries:
    samples = disassemble(binary)
    all_samples.extend(samples)

print(f"    Raw samples: {len(all_samples)}")

# Build vocabulary
mnemonic_counts = defaultdict(int)
for _, mnemonic in all_samples:
    mnemonic_counts[mnemonic] += 1

# Keep all mnemonics with >= 5 samples (lower threshold)
valid_mnemonics = {m for m, c in mnemonic_counts.items() if c >= 5}
print(f"    Valid mnemonics: {len(valid_mnemonics)}")

mnemonic_to_id = {m: i for i, m in enumerate(sorted(valid_mnemonics))}
id_to_mnemonic = {i: m for m, i in mnemonic_to_id.items()}
num_classes = len(mnemonic_to_id)

samples = [(b, m) for b, m in all_samples if m in valid_mnemonics]
print(f"    Filtered samples: {len(samples)}")

# ============================================================================
# BALANCE DATA
# ============================================================================
print("\n[BALANCING]")

by_mnemonic = defaultdict(list)
for bytes_list, mnemonic in samples:
    by_mnemonic[mnemonic].append(bytes_list)

# More aggressive balancing
MIN_SAMPLES = 100  # Higher minimum
MAX_SAMPLES = 1000  # Higher maximum

balanced = []
for mnemonic, byte_lists in by_mnemonic.items():
    unique = list(set(tuple(b) for b in byte_lists))
    
    if len(unique) < MIN_SAMPLES:
        factor = (MIN_SAMPLES // len(unique)) + 1
        unique = unique * factor
    
    unique = unique[:MAX_SAMPLES]
    
    for b in unique:
        balanced.append((list(b), mnemonic))

random.shuffle(balanced)
print(f"  Balanced samples: {len(balanced)}")

split = int(0.9 * len(balanced))
train_samples = balanced[:split]
val_samples = balanced[split:]

print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# ============================================================================
# DATASET & MODEL (same as before)
# ============================================================================
print("\n[DATASET]")

MAX_LEN = 15
PAD_BYTE = 0

class ByteDataset(Dataset):
    def __init__(self, samples, mnemonic_to_id, max_len=MAX_LEN):
        self.samples = samples
        self.mnemonic_to_id = mnemonic_to_id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bytes_list, mnemonic = self.samples[idx]
        if len(bytes_list) < self.max_len:
            bytes_list = bytes_list + [PAD_BYTE] * (self.max_len - len(bytes_list))
        else:
            bytes_list = bytes_list[:self.max_len]
        x = torch.tensor(bytes_list, dtype=torch.long)
        y = torch.tensor(self.mnemonic_to_id[mnemonic], dtype=torch.long)
        return x, y

train_dataset = ByteDataset(train_samples, mnemonic_to_id)
val_dataset = ByteDataset(val_samples, mnemonic_to_id)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

print("\n[MODEL]")

class ByteClassifier(nn.Module):
    def __init__(self, num_classes, max_len=MAX_LEN, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=PAD_BYTE)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(embed_dim, num_classes)
        )
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.byte_embed(x) + self.pos_embed(positions)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

model = ByteClassifier(num_classes=num_classes).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

# ============================================================================
# TRAINING
# ============================================================================
print("\n[TRAINING]")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

best_val_acc = 0
patience = 30  # More patience
wait = 0
best_state = None

print("  Training until convergence...")

for epoch in range(500):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion)
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict().copy()
        wait = 0
        marker = "*"
    else:
        wait += 1
        marker = ""
    
    if (epoch + 1) % 10 == 0 or marker == "*":
        print(f"    Epoch {epoch+1}: train={train_acc:.4f} val={val_acc:.4f} {marker}")
    
    if wait >= patience and epoch >= 50:
        print(f"    Early stopping at epoch {epoch+1}")
        break
    
    # Stop if we hit 100%
    if val_acc >= 0.9999:
        print(f"    Perfect validation at epoch {epoch+1}")
        best_state = model.state_dict().copy()
        break

model.load_state_dict(best_state)
print(f"\n  Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# ============================================================================
# GATE TEST
# ============================================================================
print("\n[GATE TEST]")

GATE_PROGRAMS = {
    "reverse_string": 'void reverse(char*s,int len){for(int i=0;i<len/2;i++){char t=s[i];s[i]=s[len-1-i];s[len-1-i]=t;}}int main(){char s[]="hello";reverse(s,5);return s[0];}',
    "count_ones": 'int popcount(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return popcount(0xFF);}',
    "array_max": 'int find_max(int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];return m;}int main(){int a[]={3,1,4,1,5,9,2,6};return find_max(a,8);}',
}

def predict(bytes_list):
    if len(bytes_list) < MAX_LEN:
        bytes_list = bytes_list + [0] * (MAX_LEN - len(bytes_list))
    else:
        bytes_list = bytes_list[:MAX_LEN]
    x = torch.tensor([bytes_list], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        pred_id = model(x).argmax(dim=1).item()
    return id_to_mnemonic[pred_id]

correct, total, unknown = 0, 0, 0
errors = []
import re as re_module

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
                
                # Use objdump (same as training)
                result = subprocess.run(
                    ["objdump", "-d", "-M", "intel", str(binary)],
                    capture_output=True, text=True
                )
                if not result.stdout:
                    continue
                
                for line in result.stdout.split('\n'):
                    match = re_module.match(r'\s+[0-9a-f]+:\s+([0-9a-f ]+?)\s{2,}(\S+)', line)
                    if not match:
                        continue
                    bytes_hex = match.group(1).strip().replace(' ', '')
                    expected = match.group(2).lower()
                    
                    if not bytes_hex or expected.startswith('.'):
                        continue
                    
                    bytes_list = [int(bytes_hex[i:i+2], 16) for i in range(0, len(bytes_hex), 2)]
                    
                    if expected not in mnemonic_to_id:
                        unknown += 1
                        continue
                    
                    pred = predict(bytes_list)
                    total += 1
                    
                    if pred == expected:
                        correct += 1
                    elif len(errors) < 20:
                        errors.append(f"{bytes_hex}: {pred} vs {expected}")

accuracy = 100 * correct / total if total > 0 else 0
print(f"\n  Gate Test: {correct}/{total} = {accuracy:.1f}%")
print(f"  Unknown mnemonics: {unknown}")

if errors:
    print(f"\n  Errors:")
    for e in errors[:10]:
        print(f"    {e}")

# ============================================================================
# SAVE
# ============================================================================
print("\n[SAVE]")

os.makedirs("models/level0_100", exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'mnemonic_to_id': mnemonic_to_id,
    'id_to_mnemonic': id_to_mnemonic,
    'num_classes': num_classes,
    'max_len': MAX_LEN,
}, "models/level0_100/model.pt")

config = {
    'num_classes': num_classes,
    'max_len': MAX_LEN,
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 4,
    'mnemonic_to_id': mnemonic_to_id,
    'id_to_mnemonic': {str(k): v for k, v in id_to_mnemonic.items()},
}
with open("models/level0_100/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Inference script
inference_code = '''
import torch
import torch.nn as nn

MAX_LEN = 15

class ByteClassifier(nn.Module):
    def __init__(self, num_classes, max_len=15, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(embed_dim, num_classes))
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.byte_embed(x) + self.pos_embed(positions)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

class Level0:
    def __init__(self, model_dir="models/level0_100"):
        checkpoint = torch.load(f"{model_dir}/model.pt", map_location="cpu")
        self.mnemonic_to_id = checkpoint["mnemonic_to_id"]
        self.id_to_mnemonic = checkpoint["id_to_mnemonic"]
        self.model = ByteClassifier(checkpoint["num_classes"], checkpoint["max_len"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_len = checkpoint["max_len"]
    
    def predict(self, bytes_input):
        if isinstance(bytes_input, str):
            bytes_input = bytes_input.replace(" ", "")
            bytes_list = [int(bytes_input[i:i+2], 16) for i in range(0, len(bytes_input), 2)]
        else:
            bytes_list = list(bytes_input)
        if len(bytes_list) < self.max_len:
            bytes_list = bytes_list + [0] * (self.max_len - len(bytes_list))
        x = torch.tensor([bytes_list[:self.max_len]], dtype=torch.long)
        with torch.no_grad():
            return self.id_to_mnemonic[self.model(x).argmax(dim=1).item()]

if __name__ == "__main__":
    m = Level0()
    for b, exp in [("c3","ret"),("55","push"),("4883ec20","sub"),("e800000000","call"),("ffe0","jmp")]:
        print(f"{b} -> {m.predict(b)} (exp: {exp})")
'''

with open("models/level0_100/inference.py", "w") as f:
    f.write(inference_code)

import zipfile
with zipfile.ZipFile("level0_100.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level0_100").iterdir():
        zf.write(f, f"models/level0_100/{f.name}")

print(f"  Created: level0_100.zip ({Path('level0_100.zip').stat().st_size/1024/1024:.1f} MB)")

try:
    import shutil
    os.makedirs("/content/drive/MyDrive/genesis_models", exist_ok=True)
    shutil.copy("level0_100.zip", "/content/drive/MyDrive/genesis_models/")
    print("  Saved to Google Drive")
except:
    pass

try:
    from google.colab import files
    files.download("level0_100.zip")
except:
    print(f"  Download: {os.path.abspath('level0_100.zip')}")

print("\n" + "=" * 60)
print(f"RESULT: {accuracy:.1f}%")
print("=" * 60)
