# ============================================================================
# GENESIS Training - ALL LEVELS with REAL BINARY DATA
# Colab Pro - Single Cell
# ============================================================================

import os
import subprocess

# Clone repo
if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")
print(f"Dir: {os.getcwd()}")

# Install
subprocess.run(["pip", "install", "-q", "torch", "transformers", "peft", "accelerate"], check=True)

# Generate real binary training data
print("\n" + "="*60)
print("GENERATING REAL BINARY TRAINING DATA")
print("="*60)
subprocess.run(["python", "genesis_datasets/generators/real_binary_generator.py"], check=True)

# Imports
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Config
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 32
EPOCHS = 150
LR = 3e-5
PATIENCE = 20
LORA_R = 64
LORA_ALPHA = 128


class RealDataset(Dataset):
    """Dataset from real binary data."""
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
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def train_level(level: int, dataset_path: str, output_path: str, max_len: int = 256):
    """Train a single level."""
    print(f"\n{'='*60}")
    print(f"TRAINING LEVEL {level}")
    print(f"{'='*60}")
    
    # Load tokenizer and model
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
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = RealDataset(dataset_path, tokenizer, max_len)
    print(f"Dataset: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("ERROR: Empty dataset!")
        return
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
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
            scheduler.step()
            
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        improved = ""
        
        if avg < best_loss - 0.001:
            best_loss = avg
            no_improve = 0
            improved = " * (saved)"
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
        else:
            no_improve += 1
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg:.4f}{improved}")
        
        if no_improve >= PATIENCE and epoch > 50:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Best loss: {best_loss:.4f}")
    return best_loss


# Train all levels
results = {}

# Level 0: Bytes -> Mnemonic
if Path("genesis_datasets/level0_real/train.jsonl").exists():
    results["level0"] = train_level(
        level=0,
        dataset_path="genesis_datasets/level0_real/train.jsonl",
        output_path="models/level0_real",
        max_len=128
    )

# Level 1: Instruction -> Semantics  
if Path("genesis_datasets/level1_real/train.jsonl").exists():
    results["level1"] = train_level(
        level=1,
        dataset_path="genesis_datasets/level1_real/train.jsonl",
        output_path="models/level1_real",
        max_len=256
    )

# Level 2: Instructions -> CFG
if Path("genesis_datasets/level2_real/train.jsonl").exists():
    results["level2"] = train_level(
        level=2,
        dataset_path="genesis_datasets/level2_real/train.jsonl",
        output_path="models/level2_real",
        max_len=512
    )

# Summary
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
for level, loss in results.items():
    print(f"  {level}: best loss = {loss:.4f}")

# Package models
print("\nPackaging models...")
import zipfile

with zipfile.ZipFile("models_real.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_real", "level1_real", "level2_real"]:
        model_path = Path(f"models/{level}")
        if model_path.exists():
            for f in model_path.iterdir():
                zf.write(f, f"models/{level}/{f.name}")
                
print("Done! Download models_real.zip")

from IPython.display import FileLink
display(FileLink("models_real.zip"))
