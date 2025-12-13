#!/usr/bin/env python3
"""
LEVEL 0: COMPLETE COVERAGE TRAINING

This script ensures 100% coverage of ALL 369 mnemonics found in real binaries.
It uses the verified mnemonic database as the REQUIRED vocabulary.

NO EXCEPTIONS. ALL 369 MNEMONICS MUST BE LEARNED.
"""

import os
import subprocess
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("LEVEL 0: COMPLETE COVERAGE (369 MNEMONICS)")
print("=" * 70)

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

# =============================================================================
# LOAD REQUIRED MNEMONIC DATABASE
# =============================================================================
print("\n[MNEMONIC DATABASE]")

# Load the verified complete mnemonic database
DB_PATH = Path("tests/complete_mnemonic_db.json")
if not DB_PATH.exists():
    print("ERROR: tests/complete_mnemonic_db.json not found!")
    print("Run: python tests/verify_level0_100_percent.py first")
    sys.exit(1)

with open(DB_PATH) as f:
    MNEMONIC_DB = json.load(f)

REQUIRED_MNEMONICS = sorted(MNEMONIC_DB.keys())
print(f"  Required mnemonics: {len(REQUIRED_MNEMONICS)}")

# Create mnemonic -> index mapping
MNEMONIC_TO_ID = {m: i for i, m in enumerate(REQUIRED_MNEMONICS)}
ID_TO_MNEMONIC = {i: m for m, i in MNEMONIC_TO_ID.items()}
NUM_CLASSES = len(REQUIRED_MNEMONICS)

print(f"  Vocabulary size: {NUM_CLASSES}")

# =============================================================================
# DATA COLLECTION - MUST COVER ALL 369 MNEMONICS
# =============================================================================
print("\n[DATA COLLECTION]")

cs = Cs(CS_ARCH_X86, CS_MODE_64)

def extract_samples_from_binary(binary_path, max_per_mnemonic=100):
    """Extract instruction samples from a binary."""
    samples = []
    try:
        result = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section=.text", str(binary_path), "/dev/stdout"],
            capture_output=True, timeout=30
        )
        if result.stdout:
            for insn in cs.disasm(result.stdout, 0):
                mnemonic = insn.mnemonic
                if mnemonic in MNEMONIC_TO_ID:
                    samples.append((list(insn.bytes), mnemonic))
    except:
        pass
    return samples

# Collect from system binaries
print("  Scanning system binaries...")
binaries = []
for path in ["/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/lib"]:
    if Path(path).exists():
        try:
            for f in Path(path).iterdir():
                if f.is_file() and not f.is_symlink():
                    try:
                        with open(f, 'rb') as fp:
                            if fp.read(4) == b'\x7fELF':
                                binaries.append(f)
                    except: pass
                if len(binaries) >= 500:
                    break
        except: pass
    if len(binaries) >= 500:
        break

print(f"    Found {len(binaries)} binaries")

# Collect samples, tracking coverage
samples_by_mnemonic = defaultdict(list)
print("  Extracting samples...")

for i, binary in enumerate(binaries):
    samples = extract_samples_from_binary(binary)
    for bytes_list, mnemonic in samples:
        if len(samples_by_mnemonic[mnemonic]) < 500:  # Max 500 per mnemonic
            samples_by_mnemonic[mnemonic].append(bytes_list)
    
    if (i + 1) % 100 == 0:
        covered = sum(1 for m in REQUIRED_MNEMONICS if samples_by_mnemonic[m])
        print(f"    Processed {i+1}/{len(binaries)}, covered {covered}/{NUM_CLASSES}")

# Check coverage
covered = [m for m in REQUIRED_MNEMONICS if samples_by_mnemonic[m]]
missing = [m for m in REQUIRED_MNEMONICS if not samples_by_mnemonic[m]]

print(f"\n  Coverage after scanning: {len(covered)}/{NUM_CLASSES}")

# For missing mnemonics, use examples from the database
if missing:
    print(f"  Adding {len(missing)} missing mnemonics from database...")
    for mnemonic in missing:
        for ex in MNEMONIC_DB[mnemonic]:
            try:
                hex_bytes = ex['bytes']
                bytes_list = list(bytes.fromhex(hex_bytes))
                samples_by_mnemonic[mnemonic].append(bytes_list)
            except:
                pass

# Verify all covered now
final_covered = [m for m in REQUIRED_MNEMONICS if samples_by_mnemonic[m]]
final_missing = [m for m in REQUIRED_MNEMONICS if not samples_by_mnemonic[m]]

if final_missing:
    print(f"  ERROR: Still missing {len(final_missing)} mnemonics!")
    for m in final_missing[:10]:
        print(f"    - {m}")
    print("  Cannot proceed without complete coverage.")
    sys.exit(1)

print(f"  ✓ All {NUM_CLASSES} mnemonics covered")

# =============================================================================
# BALANCE DATASET
# =============================================================================
print("\n[BALANCING]")

MIN_SAMPLES = 50
MAX_SAMPLES = 300

balanced_samples = []
for mnemonic in REQUIRED_MNEMONICS:
    samples = samples_by_mnemonic[mnemonic]
    
    # Ensure minimum samples by repeating
    while len(samples) < MIN_SAMPLES:
        samples = samples + samples
    
    # Limit to max
    if len(samples) > MAX_SAMPLES:
        samples = random.sample(samples, MAX_SAMPLES)
    else:
        samples = samples[:MAX_SAMPLES]
    
    for bytes_list in samples:
        balanced_samples.append((bytes_list, mnemonic))

random.shuffle(balanced_samples)
print(f"  Balanced samples: {len(balanced_samples)}")
print(f"  Per mnemonic: {MIN_SAMPLES}-{MAX_SAMPLES}")

# =============================================================================
# DATASET
# =============================================================================
print("\n[DATASET]")

MAX_LEN = 15

class MnemonicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        bytes_list, mnemonic = self.samples[idx]
        
        # Pad to MAX_LEN
        if len(bytes_list) < MAX_LEN:
            bytes_list = bytes_list + [0] * (MAX_LEN - len(bytes_list))
        
        x = torch.tensor(bytes_list[:MAX_LEN], dtype=torch.long)
        y = torch.tensor(MNEMONIC_TO_ID[mnemonic], dtype=torch.long)
        return x, y

# Split
split = int(0.9 * len(balanced_samples))
train_samples = balanced_samples[:split]
val_samples = balanced_samples[split:]

train_dataset = MnemonicDataset(train_samples)
val_dataset = MnemonicDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# =============================================================================
# MODEL
# =============================================================================
print("\n[MODEL]")

class ByteClassifier(nn.Module):
    def __init__(self, num_classes, max_len=15, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.byte_embed(x) + self.pos_embed(positions)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

model = ByteClassifier(NUM_CLASSES).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Classes: {NUM_CLASSES}")

# =============================================================================
# TRAINING
# =============================================================================
print("\n[TRAINING]")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Drive checkpointing
DRIVE_PATH = "/content/drive/MyDrive/genesis_models"
try:
    os.makedirs(DRIVE_PATH, exist_ok=True)
    SAVE_TO_DRIVE = True
    print(f"  Checkpoints: {DRIVE_PATH}")
except:
    SAVE_TO_DRIVE = False

def train_epoch(model, loader):
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

def eval_epoch(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total

best_val_acc = 0
patience, wait = 30, 0
best_state = None

print("  Training...")
for epoch in range(200):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc = eval_epoch(model, val_loader)
    scheduler.step()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
        
        if SAVE_TO_DRIVE and (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': best_state,
                'mnemonic_to_id': MNEMONIC_TO_ID,
                'id_to_mnemonic': ID_TO_MNEMONIC,
                'num_classes': NUM_CLASSES,
                'max_len': MAX_LEN,
            }, f"{DRIVE_PATH}/level0_checkpoint.pt")
            print(f"    Epoch {epoch+1}: loss={train_loss:.4f} val={val_acc:.4f} * [saved]")
        elif (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={train_loss:.4f} val={val_acc:.4f} *")
    else:
        wait += 1
    
    if wait >= patience and epoch >= 50:
        print(f"    Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
model.to(device)

print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

# =============================================================================
# SAVE
# =============================================================================
print("\n[SAVE]")

os.makedirs("models/level0", exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'mnemonic_to_id': MNEMONIC_TO_ID,
    'id_to_mnemonic': ID_TO_MNEMONIC,
    'num_classes': NUM_CLASSES,
    'max_len': MAX_LEN,
}, "models/level0/model.pt")

with open("models/level0/config.json", "w") as f:
    json.dump({
        'num_classes': NUM_CLASSES,
        'max_len': MAX_LEN,
        'mnemonics': REQUIRED_MNEMONICS,
    }, f, indent=2)

import zipfile
with zipfile.ZipFile("level0_complete.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level0").iterdir():
        zf.write(f, f"models/level0/{f.name}")

print(f"  Created: level0_complete.zip")

# Save to Drive
if SAVE_TO_DRIVE:
    import shutil
    shutil.copy("level0_complete.zip", f"{DRIVE_PATH}/")
    print(f"  Saved to Drive")

# Download
try:
    from google.colab import files
    files.download("level0_complete.zip")
except:
    print(f"  Download: {os.path.abspath('level0_complete.zip')}")

# =============================================================================
# VERIFY
# =============================================================================
print("\n[VERIFICATION]")
print("  Run: python tests/verify_level0_100_percent.py")
print("  Model MUST pass before Level 0 is complete")

print("\n" + "=" * 70)
print("TRAINING COMPLETE - NOW VERIFY WITH EXTERNAL TEST")
print("=" * 70)
