# ============================================================================
# GENESIS - Proper Training Pipeline
# Uses REAL code from ExeBench (not synthetic)
# Colab Pro - Single Cell
# ============================================================================
# 
# PHILOSOPHY:
# - Zero hallucination
# - Only train on verified ground truth
# - Multiple compilers, multiple opt levels
# - Capstone + objdump cross-validation
# - Strict gate test with unseen binaries
#
# ============================================================================

import os
import subprocess
import sys

print("=" * 70)
print("GENESIS PROPER TRAINING PIPELINE")
print("=" * 70)

# 1. Clone repo
print("\n[1/6] Cloning repository...")
if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")
print(f"  Working directory: {os.getcwd()}")

# 2. Install dependencies
print("\n[2/6] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "torch", "transformers", "peft", "accelerate",
                "datasets", "zstandard", "capstone"], check=True)

# 3. Generate training data from ExeBench
print("\n[3/6] Generating training data from real code...")
print("  This uses ExeBench (real C functions from GitHub)")
print("  Ground truth verified with Capstone + objdump")

# Run the generator
result = subprocess.run([sys.executable, "genesis_datasets/generators/anghabench_generator.py"],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Generator error: {result.stderr}")

# Check what we got
import json
from pathlib import Path

datasets = {}
for level in ["level0_angha", "level1_angha"]:
    path = Path(f"genesis_datasets/{level}/train.jsonl")
    if path.exists():
        with open(path) as f:
            count = sum(1 for _ in f)
        datasets[level] = count
        print(f"  {level}: {count} samples")

if not datasets or sum(datasets.values()) < 100:
    print("WARNING: Not enough training data generated!")
    print("Using fallback + real binary data...")
    subprocess.run([sys.executable, "genesis_datasets/generators/real_binary_generator.py"])

# 4. Training
print("\n[4/6] Training models...")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Config
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 32
EPOCHS = 150
LR = 3e-5
PATIENCE = 20
LORA_R = 64
LORA_ALPHA = 128


class SimpleDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompt = f"{d['input']}\n{d['output']}"
                self.samples.append(prompt)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, 
                            padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone()
        }


def train_model(name: str, dataset_path: str, output_path: str, max_len: int = 256):
    """Train a single model."""
    print(f"\n  Training {name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
    )
    
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    dataset = SimpleDataset(dataset_path, tokenizer, max_len)
    print(f"    Dataset size: {len(dataset)}")
    
    if len(dataset) < 10:
        print(f"    Skipping {name} - not enough data")
        return None
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        
        if avg < best_loss - 0.001:
            best_loss = avg
            no_improve = 0
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"    Epoch {epoch+1}: {avg:.4f} * (saved)")
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"    Epoch {epoch+1}: {avg:.4f}")
        
        if no_improve >= PATIENCE and epoch > 30:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    print(f"    Best loss: {best_loss:.4f}")
    return best_loss


# Train each level
results = {}

# Check what datasets we have
for level, suffix in [("level0", "angha"), ("level1", "angha")]:
    path = f"genesis_datasets/{level}_{suffix}/train.jsonl"
    if Path(path).exists():
        r = train_model(f"Level {level[-1]}", path, f"models/{level}_proper", 
                       max_len=128 if level == "level0" else 256)
        if r:
            results[level] = r

# Also try real binary data if exists
for level in ["level0_real", "level1_real", "level2_real"]:
    path = f"genesis_datasets/{level}/train.jsonl"
    if Path(path).exists():
        r = train_model(level, path, f"models/{level}", 
                       max_len=128 if "level0" in level else 256 if "level1" in level else 512)
        if r:
            results[level] = r

# 5. Run gate test
print("\n[5/6] Running strict gate test...")
result = subprocess.run([sys.executable, "tests/strict_real_world_gate.py"],
                       capture_output=True, text=True)
print(result.stdout)

# 6. Package and save
print("\n[6/6] Packaging models...")
import zipfile

with zipfile.ZipFile("genesis_models_proper.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for model_dir in Path("models").iterdir():
        if model_dir.is_dir() and ("proper" in model_dir.name or "real" in model_dir.name):
            for f in model_dir.iterdir():
                zf.write(f, f"models/{model_dir.name}/{f.name}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
for name, loss in results.items():
    print(f"  {name}: best loss = {loss:.4f}")

print("\nDownload: genesis_models_proper.zip")

from IPython.display import FileLink
display(FileLink("genesis_models_proper.zip"))
