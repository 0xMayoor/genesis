# ============================================================================
# GENESIS Level 2 Training - Control Flow Analysis
# Works on Kaggle AND Colab (single cell)
# ============================================================================

# --- Setup ---
import os
import subprocess

# Clone repo
if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")
print(os.getcwd())

# Install dependencies
subprocess.run(["pip", "install", "-q", "torch", "transformers", "peft", "datasets", "accelerate"], check=True)

# --- Imports ---
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path

# --- Config ---
MODEL_NAME = "distilgpt2"
BATCH_SIZE = 8  # Smaller batches - Level 2 has longer sequences
EPOCHS = 100
LR = 2e-5
PATIENCE = 10
LORA_R = 32
LORA_ALPHA = 64
MAX_LENGTH = 512  # Longer for CFG output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
                for i in instrs[:20]:  # Limit instructions shown
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
                
                # Blocks
                blocks = out["basic_blocks"]
                block_strs = [f"BB{b['id']}({b['start']:#x}-{b['end']:#x},{b['exit_type']})" for b in blocks]
                output_parts.append(f"blocks: {'; '.join(block_strs)}")
                
                # Edges
                edges = out["cfg_edges"]
                edge_strs = [f"BB{e['source']}->BB{e['target']}({e['type']})" for e in edges]
                if edge_strs:
                    output_parts.append(f"edges: {'; '.join(edge_strs)}")
                
                # Loops
                loops = out["loops"]
                if loops:
                    loop_strs = [f"loop(header=BB{l['header']},type={l['type']})" for l in loops]
                    output_parts.append(f"loops: {'; '.join(loop_strs)}")
                
                # Functions
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
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

# --- Load Model ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
)

model = get_peft_model(base_model, lora_config)
model.to(device)
model.print_trainable_parameters()

# --- Load Dataset ---
dataset = Level2Dataset("genesis_datasets/level2/train.jsonl", tokenizer)
print(f"Dataset: {len(dataset)} samples")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=EPOCHS * len(loader))

print(f"Training Level 2...")
best_loss = float("inf")
no_improve = 0

for epoch in range(EPOCHS):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg = total_loss / len(loader)
    improved = ""
    
    if avg < best_loss - 0.001:
        best_loss = avg
        no_improve = 0
        improved = " *"
        # Save best
        model.save_pretrained("models/level2_best")
        tokenizer.save_pretrained("models/level2_best")
    else:
        no_improve += 1
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg:.4f}{improved}")
    
    # Early stopping
    if no_improve >= PATIENCE and epoch > 30:
        print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

print(f"\nBest loss: {best_loss:.4f}")

# --- Verification ---
print("\nVerification:")
model.eval()

test_cases = [
    # Simple linear
    "Instructions:\n0x0:push rbp\n0x1:mov rbp,rsp\n0x2:ret [return]",
    # Simple branch
    "Instructions:\n0x0:cmp rax,0\n0x1:je 0x10 [conditional->0x10]\n0x2:mov rax,1\n0x3:ret [return]\n0x10:mov rax,0\n0x11:ret [return]",
    # Simple loop
    "Instructions:\n0x0:cmp rcx,0\n0x1:je 0x10 [conditional->0x10]\n0x2:dec rcx\n0x3:jmp 0x0 [jump->0x0]\n0x10:ret [return]",
]

for test in test_cases:
    prompt = f"{test}\nAnalysis:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    analysis = result.split("Analysis:")[-1].strip()[:100]
    print(f"  {test[:40]}... -> {analysis}...")

# --- Save & Zip ---
print("\nSaving model...")
model.save_pretrained("models/level2_best")
tokenizer.save_pretrained("models/level2_best")

import zipfile
with zipfile.ZipFile("level2_model.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path("models/level2_best").iterdir():
        zf.write(f, f"models/level2_best/{f.name}")

print("Done! Download level2_model.zip")

# For Colab/Kaggle download
try:
    from IPython.display import FileLink
    display(FileLink("level2_model.zip"))
except:
    pass
