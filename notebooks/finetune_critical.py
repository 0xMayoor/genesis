# ============================================================================
# GENESIS Critical Fix - Fine-tune on missing instructions
# Run after train_genesis_v2.py or v3
# ============================================================================

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from pathlib import Path

print("=" * 60)
print("CRITICAL INSTRUCTION FIX")
print("=" * 60)

# Critical missing patterns
CRITICAL_L0 = [
    # ret - THE MOST IMPORTANT FIX
    ("c3", "ret"), ("c3", "ret"), ("c3", "ret"),  # Repeat for emphasis
    ("c2", "ret"), ("cb", "retf"), ("ca", "retf"),
    
    # Common single-byte push/pop
    ("50", "push"), ("51", "push"), ("52", "push"), ("53", "push"),
    ("54", "push"), ("55", "push"), ("56", "push"), ("57", "push"),
    ("58", "pop"), ("59", "pop"), ("5a", "pop"), ("5b", "pop"),
    ("5c", "pop"), ("5d", "pop"), ("5e", "pop"), ("5f", "pop"),
    
    # endbr64/32
    ("f30f1efa", "endbr64"), ("f30f1efb", "endbr32"),
    
    # nop
    ("90", "nop"), ("0f1f00", "nop"), ("0f1f4000", "nop"),
    
    # movzx/movsx variants
    ("0fb6c0", "movzx"), ("0fb6c1", "movzx"), ("0fb6c2", "movzx"),
    ("0fb6d0", "movzx"), ("0fb6d1", "movzx"), ("0fb6d2", "movzx"),
    ("0fbec0", "movsx"), ("0fbec1", "movsx"), ("0fbec2", "movsx"),
    ("0fbed0", "movsx"), ("0fbed1", "movsx"), ("0fbed2", "movsx"),
    ("480fbed0", "movsx"), ("480fbec8", "movsx"),
    ("480fb6c0", "movzx"), ("480fb6c8", "movzx"),
    
    # call variants (with dummy offsets)
    ("e800000000", "call"), ("e8fcffffff", "call"), ("e8f0ffffff", "call"),
    ("ff15", "call"), ("ffd0", "call"), ("ffd1", "call"), ("ffd2", "call"),
    
    # leave
    ("c9", "leave"),
    
    # cdqe, cdq, cqo
    ("4898", "cdqe"), ("99", "cdq"), ("4899", "cqo"),
    
    # xchg
    ("87c0", "xchg"), ("87d0", "xchg"), ("4887c0", "xchg"),
    
    # int3
    ("cc", "int3"),
]

CRITICAL_L1 = [
    # ret semantics - CRITICAL
    ("ret", "return from function; control flow; pop return address; jump back"),
    ("ret 0x10", "return from function; control flow; pop return address; jump back; cleanup stack"),
    ("retf", "far return; control flow; pop return address and segment"),
    
    # endbr semantics
    ("endbr64", "end branch 64; control flow integrity; security; CET marker"),
    ("endbr32", "end branch 32; control flow integrity; security; CET marker"),
    
    # movzx/movsx
    ("movzx eax, al", "move zero-extend; write destination; read source; register transfer; copy with zero extension"),
    ("movzx eax, byte ptr [rax]", "move zero-extend; write destination; read memory; register transfer; copy with zero extension"),
    ("movsx eax, al", "move sign-extend; write destination; read source; register transfer; copy with sign extension"),
    ("movsx rax, eax", "move sign-extend; write destination; read source; register transfer; copy with sign extension"),
    
    # call
    ("call 0x1000", "call function; control flow; push return address; jump to target"),
    ("call rax", "call function; control flow; push return address; jump to register"),
    ("call [rax]", "call function; control flow; push return address; jump to memory"),
    
    # leave
    ("leave", "leave stack frame; restore rbp rsp; stack cleanup; function epilogue"),
    
    # cdqe, cdq, cqo
    ("cdqe", "sign extend eax to rax; conversion; arithmetic"),
    ("cdq", "sign extend eax to edx:eax; conversion; arithmetic"),
    ("cqo", "sign extend rax to rdx:rax; conversion; arithmetic"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class CriticalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
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

def finetune_level(level, base_model_path, critical_samples, output_path, epochs=50):
    print(f"\nFine-tuning {level}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load existing model
    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base, base_model_path)
    model.to(device)
    
    # Prepare data - repeat critical samples many times
    samples = []
    for _ in range(50):  # 50 copies of each critical sample
        for item in critical_samples:
            if level == "level0":
                text = f"Bytes: {item[0]}\nInstruction: {item[1]}"
            else:
                text = f"Instruction: {item[0]}\n{item[1]}"
            samples.append(text)
    
    print(f"  Training samples: {len(samples)}")
    
    dataset = CriticalDataset(samples, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Low LR for fine-tuning
    
    model.train()
    for epoch in range(epochs):
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
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1}: {total_loss/len(loader):.4f}")
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Saved to {output_path}")

# Check which models exist
os.chdir("/content/genesis")

for version in ["v2", "v3"]:
    l0_path = f"models/level0_{version}"
    l1_path = f"models/level1_{version}"
    
    if Path(l0_path).exists():
        print(f"\nFound {version} models")
        finetune_level("level0", l0_path, CRITICAL_L0, f"models/level0_{version}_fixed", epochs=30)
        finetune_level("level1", l1_path, CRITICAL_L1, f"models/level1_{version}_fixed", epochs=30)
        break
else:
    print("No v2 or v3 models found! Run train_genesis_v2.py first.")

# Quick test
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

for level in ["level0", "level1"]:
    for version in ["v2_fixed", "v3_fixed"]:
        path = f"models/{level}_{version}"
        if not Path(path).exists():
            continue
        
        print(f"\n{level}_{version}:")
        tokenizer = AutoTokenizer.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model = PeftModel.from_pretrained(base, path).to(device).eval()
        
        if level == "level0":
            tests = ["Bytes: c3", "Bytes: 55", "Bytes: f30f1efa"]
        else:
            tests = ["Instruction: ret", "Instruction: endbr64"]
        
        for test in tests:
            inputs = tokenizer(test + "\n", return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, do_sample=False, 
                                    pad_token_id=tokenizer.eos_token_id)
            result = tokenizer.decode(out[0], skip_special_tokens=True).split("\n")[-1][:50]
            print(f"  {test} → {result}")

# Package
print("\n" + "=" * 60)
print("PACKAGING")
print("=" * 60)

import zipfile
with zipfile.ZipFile("genesis_fixed.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0", "level1"]:
        for version in ["v2_fixed", "v3_fixed"]:
            path = Path(f"models/{level}_{version}")
            if path.exists():
                for f in path.iterdir():
                    zf.write(f, f"models/{level}_{version}/{f.name}")

from google.colab import files
files.download("genesis_fixed.zip")
