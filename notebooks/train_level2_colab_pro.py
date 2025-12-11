# ============================================================================
# GENESIS Level 2 Training - Colab Pro/Pro+ Optimized
# Single cell - takes advantage of A100/V100 GPU
# ============================================================================

# --- Setup ---
import os
import subprocess

# Clone repo
if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")
print(f"Working dir: {os.getcwd()}")

# Install dependencies
subprocess.run(["pip", "install", "-q", "torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes"], check=True)

# --- Imports ---
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path

# --- Colab Pro Config (Optimized for A100/V100) ---
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 32          # Larger batch - more VRAM available
GRADIENT_ACCUM = 2       # Effective batch = 64
EPOCHS = 100
LR = 3e-5                # Slightly higher LR with larger batch
PATIENCE = 15            # More patience - we have time
LORA_R = 64              # Larger LoRA rank for better capacity
LORA_ALPHA = 128
MAX_LENGTH = 512
USE_FP16 = True          # Mixed precision for speed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Dataset ---
class Level2Dataset(Dataset):
    def __init__(self, path: str, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                # Format input
                instrs = d["instructions"]
                instr_strs = []
                for i in instrs[:20]:
                    ops = ",".join(i["operands"]) if i["operands"] else ""
                    cf = ""
                    if i["control_flow"]:
                        cf = f" [{i['control_flow']['type']}"
                        if i["control_flow"]["target"]:
                            cf += f"->{i['control_flow']['target']}"
                        cf += "]"
                    instr_strs.append(f"{i['offset']:#x}:{i['mnemonic']} {ops}{cf}")
                
                input_text = "Instructions:\n" + "\n".join(instr_strs)
                
                # Format output
                out = d["output"]
                output_parts = []
                
                blocks = out["basic_blocks"]
                block_strs = [f"BB{b['id']}({b['start']:#x}-{b['end']:#x},{b['exit_type']})" for b in blocks]
                output_parts.append(f"blocks: {'; '.join(block_strs)}")
                
                edges = out["cfg_edges"]
                edge_strs = [f"BB{e['source']}->BB{e['target']}({e['type']})" for e in edges]
                if edge_strs:
                    output_parts.append(f"edges: {'; '.join(edge_strs)}")
                
                loops = out["loops"]
                if loops:
                    loop_strs = [f"loop(header=BB{l['header']},type={l['type']})" for l in loops]
                    output_parts.append(f"loops: {'; '.join(loop_strs)}")
                
                funcs = out["functions"]
                if funcs:
                    func_strs = [f"func({f['entry']:#x},{len(f['blocks'])}blocks)" for f in funcs]
                    output_parts.append(f"functions: {'; '.join(func_strs)}")
                
                output_text = "; ".join(output_parts)
                prompt = f"{input_text}\nAnalysis: {output_text}"
                self.samples.append(prompt)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

# --- Load Model ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if USE_FP16 else torch.float32
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,  # Lower dropout
    target_modules=["c_attn", "c_proj"],
)

model = get_peft_model(base_model, lora_config)
model.to(device)
model.print_trainable_parameters()

# --- Load Dataset ---
dataset = Level2Dataset("genesis_datasets/level2/train.jsonl", tokenizer)
print(f"Dataset: {len(dataset)} samples")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# --- Training with Mixed Precision ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = EPOCHS * len(loader) // GRADIENT_ACCUM
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=total_steps)
scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

print(f"\nTraining Level 2 (Colab Pro)...")
print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUM}")
print(f"  FP16: {USE_FP16}")
print(f"  LoRA rank: {LORA_R}")

best_loss = float("inf")
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRADIENT_ACCUM
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRADIENT_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRADIENT_ACCUM
    
    avg = total_loss / len(loader)
    improved = ""
    
    if avg < best_loss - 0.001:
        best_loss = avg
        no_improve = 0
        improved = " * (saved)"
        model.save_pretrained("models/level2_best")
        tokenizer.save_pretrained("models/level2_best")
    else:
        no_improve += 1
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg:.4f}{improved}")
    
    if no_improve >= PATIENCE and epoch > 30:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\n{'='*50}")
print(f"Training complete! Best loss: {best_loss:.4f}")

# --- Verification ---
print("\nQuick verification:")
model.eval()

test_cases = [
    "Instructions:\n0x0:push rbp\n0x1:mov rbp,rsp\n0x2:ret [return]",
    "Instructions:\n0x0:cmp rax,0\n0x1:je 0x10 [conditional->0x10]\n0x2:mov rax,1\n0x3:ret [return]\n0x10:mov rax,0\n0x11:ret [return]",
    "Instructions:\n0x0:cmp rcx,0\n0x1:je 0x10 [conditional->0x10]\n0x2:dec rcx\n0x3:jmp 0x0 [jump->0x0]\n0x10:ret [return]",
]

for i, test in enumerate(test_cases):
    prompt = f"{test}\nAnalysis:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_FP16):
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    analysis = result.split("Analysis:")[-1].strip()[:80]
    print(f"  Test {i+1}: {analysis}...")

# --- Save & Download ---
print("\nPackaging model...")
import zipfile
with zipfile.ZipFile("level2_model.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level2_best").iterdir():
        zf.write(f, f"models/level2_best/{f.name}")

print("✓ Done! Download level2_model.zip")

from IPython.display import FileLink
display(FileLink("level2_model.zip"))
