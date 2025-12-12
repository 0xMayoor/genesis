#!/usr/bin/env python3
"""
LEVEL 0: Byte-Level Classifier

Correct approach for learning bytes → mnemonic:
- Byte-level tokenization (256 vocab)
- Small transformer encoder
- Classification head (softmax over ~200 mnemonics)
- CrossEntropy loss

This LEARNS the patterns, doesn't memorize them.
"""

import os
import subprocess
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("LEVEL 0: BYTE-LEVEL CLASSIFIER")
print("=" * 60)

# ============================================================================
# SETUP
# ============================================================================
print("\n[SETUP]")

subprocess.run(["apt-get", "update"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "binutils", "gcc", "clang"], capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "torch", "capstone", "numpy"], capture_output=True)

# Handle genesis directory
if "genesis" not in os.getcwd():
    subprocess.run(["rm", "-rf", "genesis"], capture_output=True)
    subprocess.run(["git", "clone", "-q", "https://github.com/0xMayoor/genesis.git"])
    os.chdir("genesis")

print(f"  Dir: {os.getcwd()}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ============================================================================
# DATA COLLECTION
# ============================================================================
print("\n[DATA COLLECTION]")

def disassemble(binary_path):
    """Disassemble binary with Capstone - the ground truth."""
    try:
        result = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", 
             str(binary_path), "/dev/stdout"],
            capture_output=True, timeout=30
        )
        if not result.stdout:
            return []
        cs = Cs(CS_ARCH_X86, CS_MODE_64)
        # Return raw bytes (not hex string) and mnemonic
        return [(list(i.bytes), i.mnemonic) for i in cs.disasm(result.stdout, 0)]
    except:
        return []

# Find system binaries
print("  Finding system binaries...")
binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin", "/usr/lib/x86_64-linux-gnu"]:
    if Path(path).exists():
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
    if len(binaries) >= 200:
        break

print(f"    Found {len(binaries)} system binaries")

# Compile additional programs
print("  Compiling additional programs...")
PROGRAMS = {
    "basic": "int main(){return 0;}",
    "loop": "int main(){int s=0;for(int i=0;i<100;i++)s+=i;return s;}",
    "func": "int f(int x){return x*2+1;}int g(int x){return f(x)+f(x-1);}int main(){return g(10);}",
    "ptr": "void swap(int*a,int*b){int t=*a;*a=*b;*b=t;}int main(){int x=1,y=2;swap(&x,&y);return x;}",
    "arr": "int sum(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;}int main(){int a[]={1,2,3,4,5};return sum(a,5);}",
    "str": "int len(char*s){int n=0;while(s[n])n++;return n;}int main(){return len(\"hello\");}",
    "cond": "int max(int a,int b){return a>b?a:b;}int main(){return max(5,3);}",
    "switch": "int f(int x){switch(x){case 0:return 10;case 1:return 20;case 2:return 30;default:return 0;}}int main(){return f(1);}",
    "rec": "int fib(int n){return n<=1?n:fib(n-1)+fib(n-2);}int main(){return fib(10);}",
    "math": "int main(){int a=100,b=37;return ((a+b)*(a-b))/(a%b+1);}",
    "bit": "unsigned rev(unsigned n){unsigned r=0;for(int i=0;i<32;i++){r=(r<<1)|(n&1);n>>=1;}return r;}int main(){return rev(0x12345678)&0xFF;}",
    "mem": "void cpy(char*d,char*s,int n){while(n--)*d++=*s++;}int main(){char a[8],b[]=\"test\";cpy(a,b,5);return a[0];}",
    "struct": "struct P{int x,y;};int dist(struct P*p){return p->x*p->x+p->y*p->y;}int main(){struct P p={3,4};return dist(&p);}",
    "float": "int main(){float x=3.14f;return (int)(x*10);}",
    "call": "int a(){return 1;}int b(){return a()+2;}int c(){return b()+3;}int main(){return c();}",
}

import tempfile
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

print(f"    Total binaries: {len(binaries)}")

# Extract all instruction patterns
print("  Extracting instructions...")
all_samples = []  # (bytes_list, mnemonic)

for binary in binaries:
    samples = disassemble(binary)
    all_samples.extend(samples)

print(f"    Raw samples: {len(all_samples)}")

# Build mnemonic vocabulary
mnemonic_counts = defaultdict(int)
for _, mnemonic in all_samples:
    mnemonic_counts[mnemonic] += 1

# Filter: keep mnemonics with >= 10 samples
valid_mnemonics = {m for m, c in mnemonic_counts.items() if c >= 10}
print(f"    Valid mnemonics: {len(valid_mnemonics)}")

# Create mnemonic -> id mapping
mnemonic_to_id = {m: i for i, m in enumerate(sorted(valid_mnemonics))}
id_to_mnemonic = {i: m for m, i in mnemonic_to_id.items()}
num_classes = len(mnemonic_to_id)

print(f"    Classes: {num_classes}")

# Filter samples
samples = [(b, m) for b, m in all_samples if m in valid_mnemonics]
print(f"    Filtered samples: {len(samples)}")

# ============================================================================
# BALANCE DATA
# ============================================================================
print("\n[BALANCING]")

# Group by mnemonic
by_mnemonic = defaultdict(list)
for bytes_list, mnemonic in samples:
    by_mnemonic[mnemonic].append(bytes_list)

# Balance: min 50, max 500 per class
MIN_SAMPLES = 50
MAX_SAMPLES = 500

balanced = []
for mnemonic, byte_lists in by_mnemonic.items():
    # Deduplicate
    unique = list(set(tuple(b) for b in byte_lists))
    
    # Oversample if needed
    if len(unique) < MIN_SAMPLES:
        factor = (MIN_SAMPLES // len(unique)) + 1
        unique = unique * factor
    
    # Cap
    unique = unique[:MAX_SAMPLES]
    
    for b in unique:
        balanced.append((list(b), mnemonic))

random.shuffle(balanced)
print(f"  Balanced samples: {len(balanced)}")

# Split train/val
split = int(0.9 * len(balanced))
train_samples = balanced[:split]
val_samples = balanced[split:]

print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# ============================================================================
# DATASET
# ============================================================================
print("\n[DATASET]")

MAX_LEN = 15  # Max x86 instruction length
PAD_BYTE = 0  # Padding value

class ByteDataset(Dataset):
    def __init__(self, samples, mnemonic_to_id, max_len=MAX_LEN):
        self.samples = samples
        self.mnemonic_to_id = mnemonic_to_id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bytes_list, mnemonic = self.samples[idx]
        
        # Pad or truncate to max_len
        if len(bytes_list) < self.max_len:
            bytes_list = bytes_list + [PAD_BYTE] * (self.max_len - len(bytes_list))
        else:
            bytes_list = bytes_list[:self.max_len]
        
        # Convert to tensor
        x = torch.tensor(bytes_list, dtype=torch.long)
        y = torch.tensor(self.mnemonic_to_id[mnemonic], dtype=torch.long)
        
        return x, y

train_dataset = ByteDataset(train_samples, mnemonic_to_id)
val_dataset = ByteDataset(val_samples, mnemonic_to_id)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# ============================================================================
# MODEL
# ============================================================================
print("\n[MODEL]")

class ByteClassifier(nn.Module):
    """
    Small transformer-based classifier for byte sequences.
    
    Architecture:
    - Byte embedding (256 vocab)
    - Positional embedding
    - 4 transformer encoder layers
    - Classification head
    """
    def __init__(self, num_classes, max_len=MAX_LEN, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        
        # Embeddings
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=PAD_BYTE)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.max_len = max_len
    
    def forward(self, x):
        # x: (batch, seq_len) - byte values 0-255
        batch_size, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.byte_embed(x) + self.pos_embed(positions)
        
        # Transformer
        encoded = self.transformer(embeddings)
        
        # Pool: take mean of all positions
        pooled = encoded.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits

model = ByteClassifier(num_classes=num_classes)
model = model.to(device)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

# ============================================================================
# TRAINING
# ============================================================================
print("\n[TRAINING]")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return total_loss / len(loader), correct / total

# Training loop
best_val_acc = 0
patience = 20
wait = 0
best_state = None

print("  Training until convergence...")

for epoch in range(500):  # Max epochs
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
        print(f"    Epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} {marker}")
    
    # Early stopping
    if wait >= patience and epoch >= 50:
        print(f"    Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(best_state)
print(f"\n  Best val accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

# ============================================================================
# TESTING ON GATE TEST PROGRAMS
# ============================================================================
print("\n[GATE TEST]")

GATE_PROGRAMS = {
    "reverse_string": 'void reverse(char*s,int len){for(int i=0;i<len/2;i++){char t=s[i];s[i]=s[len-1-i];s[len-1-i]=t;}}int main(){char s[]="hello";reverse(s,5);return s[0];}',
    "count_ones": 'int popcount(unsigned n){int c=0;while(n){c+=n&1;n>>=1;}return c;}int main(){return popcount(0xFF);}',
    "array_max": 'int find_max(int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]>m)m=a[i];return m;}int main(){int a[]={3,1,4,1,5,9,2,6};return find_max(a,8);}',
}

import re

correct = 0
total = 0
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
                
                # Get instructions
                for bytes_list, expected in disassemble(binary):
                    if expected not in mnemonic_to_id:
                        continue  # Skip unknown mnemonics
                    
                    # Prepare input
                    if len(bytes_list) < MAX_LEN:
                        bytes_list = bytes_list + [PAD_BYTE] * (MAX_LEN - len(bytes_list))
                    else:
                        bytes_list = bytes_list[:MAX_LEN]
                    
                    x = torch.tensor([bytes_list], dtype=torch.long).to(device)
                    
                    # Predict
                    model.eval()
                    with torch.no_grad():
                        logits = model(x)
                        pred_id = logits.argmax(dim=1).item()
                        pred = id_to_mnemonic[pred_id]
                    
                    total += 1
                    if pred == expected:
                        correct += 1
                    elif len(errors) < 20:
                        errors.append(f"{bytes(bytes_list[:8]).hex()}: {pred} vs {expected}")

accuracy = 100 * correct / total if total > 0 else 0
print(f"\n  Gate Test Accuracy: {correct}/{total} = {accuracy:.1f}%")

if errors:
    print(f"\n  Sample errors:")
    for e in errors[:10]:
        print(f"    {e}")

# ============================================================================
# SAVE
# ============================================================================
print("\n[SAVE]")

os.makedirs("models/level0_classifier", exist_ok=True)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'mnemonic_to_id': mnemonic_to_id,
    'id_to_mnemonic': id_to_mnemonic,
    'num_classes': num_classes,
    'max_len': MAX_LEN,
}, "models/level0_classifier/model.pt")

# Save config
config = {
    'num_classes': num_classes,
    'max_len': MAX_LEN,
    'embed_dim': 128,
    'num_heads': 4,
    'num_layers': 4,
    'mnemonic_to_id': mnemonic_to_id,
    'id_to_mnemonic': {str(k): v for k, v in id_to_mnemonic.items()},
}
with open("models/level0_classifier/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Create inference script
inference_code = '''
import torch
import torch.nn as nn
import json

MAX_LEN = 15
PAD_BYTE = 0

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

class Level0Classifier:
    def __init__(self, model_dir="models/level0_classifier"):
        checkpoint = torch.load(f"{model_dir}/model.pt", map_location="cpu")
        self.mnemonic_to_id = checkpoint["mnemonic_to_id"]
        self.id_to_mnemonic = checkpoint["id_to_mnemonic"]
        self.num_classes = checkpoint["num_classes"]
        self.max_len = checkpoint["max_len"]
        
        self.model = ByteClassifier(self.num_classes, self.max_len)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def predict(self, bytes_input):
        """Predict mnemonic from bytes. Accepts hex string or list of ints."""
        if isinstance(bytes_input, str):
            bytes_input = bytes_input.replace(" ", "")
            bytes_list = [int(bytes_input[i:i+2], 16) for i in range(0, len(bytes_input), 2)]
        else:
            bytes_list = list(bytes_input)
        
        # Pad
        if len(bytes_list) < self.max_len:
            bytes_list = bytes_list + [0] * (self.max_len - len(bytes_list))
        else:
            bytes_list = bytes_list[:self.max_len]
        
        x = torch.tensor([bytes_list], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            pred_id = logits.argmax(dim=1).item()
        
        return self.id_to_mnemonic[pred_id]

if __name__ == "__main__":
    model = Level0Classifier()
    tests = [("c3", "ret"), ("55", "push"), ("4883ec20", "sub"), ("e800000000", "call")]
    for b, exp in tests:
        pred = model.predict(b)
        print(f"{b} -> {pred} (expected: {exp}) {"✓" if pred == exp else "✗"}")
'''

with open("models/level0_classifier/inference.py", "w") as f:
    f.write(inference_code)

# Package
import zipfile
with zipfile.ZipFile("level0_classifier.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level0_classifier").iterdir():
        zf.write(f, f"models/level0_classifier/{f.name}")

size_mb = Path("level0_classifier.zip").stat().st_size / 1024 / 1024
print(f"\n  Created: level0_classifier.zip ({size_mb:.1f} MB)")

# Save to Drive
try:
    import shutil
    os.makedirs("/content/drive/MyDrive/genesis_models", exist_ok=True)
    shutil.copy("level0_classifier.zip", "/content/drive/MyDrive/genesis_models/")
    print("  Saved to Google Drive")
except:
    pass

try:
    from google.colab import files
    files.download("level0_classifier.zip")
except:
    print(f"  Download: {os.path.abspath('level0_classifier.zip')}")

print("\n" + "=" * 60)
print("DONE")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"  Parameters: {num_params:,}")
print("=" * 60)
