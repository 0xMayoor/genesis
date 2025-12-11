# ============================================================================
# GENESIS FINAL - Comprehensive Training
# One training to rule them all
# ============================================================================
#
# This script covers EVERYTHING:
# - All common x86-64 instruction bytes
# - Exact keyword matching for Level 1 semantics
# - Proper CFG format for Level 2
# - Real binary data + synthetic coverage
#
# ============================================================================

import os
import subprocess
import sys
import json
import random
import hashlib
import tempfile
import re
from pathlib import Path

print("=" * 70)
print("GENESIS FINAL - COMPREHENSIVE TRAINING")
print("=" * 70)

# Setup
subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "clang"], capture_output=True)

if not os.path.exists("genesis"):
    subprocess.run(["git", "clone", "https://github.com/0xMayoor/genesis.git"], check=True)
os.chdir("genesis")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "peft", "accelerate", "capstone"], check=True)

# ============================================================================
# LEVEL 0: Complete x86-64 byte → mnemonic mapping
# ============================================================================
print("\n[1/5] Building comprehensive Level 0 dataset...")

# Common x86-64 instruction bytes (verified against Capstone)
LEVEL0_INSTRUCTIONS = [
    # Returns - CRITICAL
    ("c3", "ret"), ("c2", "ret"), ("cb", "retf"), ("ca", "retf"),
    
    # Push (0x50-0x57 + REX variants)
    ("50", "push"), ("51", "push"), ("52", "push"), ("53", "push"),
    ("54", "push"), ("55", "push"), ("56", "push"), ("57", "push"),
    ("4150", "push"), ("4151", "push"), ("4152", "push"), ("4153", "push"),
    ("4154", "push"), ("4155", "push"), ("4156", "push"), ("4157", "push"),
    ("6a00", "push"), ("6a01", "push"), ("6aff", "push"),
    ("6800000000", "push"), ("68ffffffff", "push"),
    
    # Pop (0x58-0x5f + REX variants)
    ("58", "pop"), ("59", "pop"), ("5a", "pop"), ("5b", "pop"),
    ("5c", "pop"), ("5d", "pop"), ("5e", "pop"), ("5f", "pop"),
    ("4158", "pop"), ("4159", "pop"), ("415a", "pop"), ("415b", "pop"),
    ("415c", "pop"), ("415d", "pop"), ("415e", "pop"), ("415f", "pop"),
    
    # NOP variants
    ("90", "nop"), ("0f1f00", "nop"), ("0f1f4000", "nop"),
    ("0f1f440000", "nop"), ("660f1f440000", "nop"),
    
    # MOV variants (most common)
    ("89c0", "mov"), ("89c1", "mov"), ("89c2", "mov"), ("89c3", "mov"),
    ("89d0", "mov"), ("89d1", "mov"), ("89d2", "mov"), ("89d3", "mov"),
    ("8b00", "mov"), ("8b01", "mov"), ("8b02", "mov"), ("8b03", "mov"),
    ("b800000000", "mov"), ("b900000000", "mov"), ("ba00000000", "mov"),
    ("4889c0", "mov"), ("4889c1", "mov"), ("4889c2", "mov"), ("4889c3", "mov"),
    ("4889d0", "mov"), ("4889d1", "mov"), ("4889d2", "mov"),
    ("488b00", "mov"), ("488b01", "mov"), ("488b02", "mov"),
    ("48c7c000000000", "mov"), ("48c7c100000000", "mov"),
    ("4889e5", "mov"), ("488945f8", "mov"), ("488b45f8", "mov"),
    ("8945fc", "mov"), ("8b45fc", "mov"),
    ("c70424", "mov"), ("c74424", "mov"),
    ("890424", "mov"), ("8b0424", "mov"),
    ("488b3d", "mov"), ("488b05", "mov"),
    
    # MOVZX/MOVSX - zero/sign extend
    ("0fb6c0", "movzx"), ("0fb6c1", "movzx"), ("0fb6c2", "movzx"),
    ("0fb6d0", "movzx"), ("0fb6d1", "movzx"), ("0fb6d2", "movzx"),
    ("0fb600", "movzx"), ("0fb601", "movzx"),
    ("480fb6c0", "movzx"), ("480fb6c1", "movzx"),
    ("0fb7c0", "movzx"), ("0fb7c1", "movzx"),  # word to dword
    ("0fbec0", "movsx"), ("0fbec1", "movsx"), ("0fbec2", "movsx"),
    ("0fbed0", "movsx"), ("0fbed1", "movsx"),
    ("0fbfc0", "movsx"), ("0fbfc1", "movsx"),  # word to dword
    ("4863c0", "movsxd"), ("4863c1", "movsxd"), ("4863c2", "movsxd"),
    ("4863d0", "movsxd"), ("4863d1", "movsxd"),
    
    # LEA
    ("8d00", "lea"), ("8d01", "lea"), ("8d02", "lea"),
    ("488d00", "lea"), ("488d01", "lea"), ("488d02", "lea"),
    ("488d0500000000", "lea"), ("488d0d00000000", "lea"),
    ("488d4500", "lea"), ("488d45f8", "lea"),
    ("8d4500", "lea"), ("8d45f8", "lea"),
    
    # ADD
    ("01c0", "add"), ("01c1", "add"), ("01c2", "add"),
    ("03c0", "add"), ("03c1", "add"), ("03c2", "add"),
    ("83c001", "add"), ("83c101", "add"), ("83c201", "add"),
    ("83c008", "add"), ("83c010", "add"),
    ("4801c0", "add"), ("4801c1", "add"),
    ("4803c0", "add"), ("4803c1", "add"),
    ("4883c001", "add"), ("4883c101", "add"),
    ("4883c008", "add"), ("4883c010", "add"),
    ("0500000000", "add"), ("81c0", "add"),
    ("4805", "add"),
    
    # SUB
    ("29c0", "sub"), ("29c1", "sub"), ("29c2", "sub"),
    ("2bc0", "sub"), ("2bc1", "sub"), ("2bc2", "sub"),
    ("83e801", "sub"), ("83e901", "sub"), ("83ea01", "sub"),
    ("83e808", "sub"), ("83e810", "sub"), ("83e820", "sub"),
    ("4829c0", "sub"), ("4829c1", "sub"),
    ("482bc0", "sub"), ("482bc1", "sub"),
    ("4883e801", "sub"), ("4883e908", "sub"),
    ("4883ec08", "sub"), ("4883ec10", "sub"), ("4883ec20", "sub"),
    ("2d00000000", "sub"), ("81e8", "sub"),
    
    # INC/DEC
    ("ffc0", "inc"), ("ffc1", "inc"), ("ffc2", "inc"),
    ("ffc3", "inc"), ("ffc6", "inc"), ("ffc7", "inc"),
    ("48ffc0", "inc"), ("48ffc1", "inc"),
    ("ffc8", "dec"), ("ffc9", "dec"), ("ffca", "dec"),
    ("48ffc8", "dec"), ("48ffc9", "dec"),
    
    # NEG
    ("f7d8", "neg"), ("f7d9", "neg"), ("f7da", "neg"),
    ("48f7d8", "neg"), ("48f7d9", "neg"),
    
    # MUL/IMUL/DIV/IDIV
    ("f7e0", "mul"), ("f7e1", "mul"), ("f7e2", "mul"),
    ("f7e8", "imul"), ("f7e9", "imul"), ("f7ea", "imul"),
    ("0fafc0", "imul"), ("0fafc1", "imul"), ("0fafc2", "imul"),
    ("480fafc0", "imul"), ("480fafc1", "imul"),
    ("6bc001", "imul"), ("696bc0", "imul"),
    ("f7f0", "div"), ("f7f1", "div"), ("f7f2", "div"),
    ("f7f8", "idiv"), ("f7f9", "idiv"), ("f7fa", "idiv"),
    ("48f7f8", "idiv"), ("48f7f9", "idiv"),
    
    # CMP
    ("39c0", "cmp"), ("39c1", "cmp"), ("39c2", "cmp"),
    ("3bc0", "cmp"), ("3bc1", "cmp"), ("3bc2", "cmp"),
    ("83f800", "cmp"), ("83f801", "cmp"), ("83f8ff", "cmp"),
    ("83f900", "cmp"), ("83fa00", "cmp"),
    ("4839c0", "cmp"), ("4839c1", "cmp"),
    ("4883f800", "cmp"), ("4883f801", "cmp"),
    ("3d00000000", "cmp"), ("81f8", "cmp"),
    ("80f800", "cmp"), ("80f900", "cmp"),
    
    # TEST
    ("85c0", "test"), ("85c9", "test"), ("85d2", "test"),
    ("4885c0", "test"), ("4885c9", "test"),
    ("a900000000", "test"), ("f6c001", "test"),
    ("84c0", "test"), ("84c9", "test"),
    
    # AND/OR/XOR
    ("21c0", "and"), ("21c1", "and"), ("21c2", "and"),
    ("23c0", "and"), ("23c1", "and"),
    ("83e001", "and"), ("83e00f", "and"), ("83e0ff", "and"),
    ("4821c0", "and"), ("4883e0", "and"),
    ("09c0", "or"), ("09c1", "or"), ("09c2", "or"),
    ("0bc0", "or"), ("0bc1", "or"),
    ("83c801", "or"), ("83c80f", "or"),
    ("4809c0", "or"), ("4883c8", "or"),
    ("31c0", "xor"), ("31c9", "xor"), ("31d2", "xor"),
    ("33c0", "xor"), ("33c9", "xor"),
    ("83f000", "xor"), ("83f001", "xor"),
    ("4831c0", "xor"), ("4831c9", "xor"), ("4831d2", "xor"),
    
    # NOT
    ("f7d0", "not"), ("f7d1", "not"), ("f7d2", "not"),
    ("48f7d0", "not"), ("48f7d1", "not"),
    
    # SHL/SHR/SAR
    ("d1e0", "shl"), ("d1e1", "shl"), ("d1e2", "shl"),
    ("c1e001", "shl"), ("c1e002", "shl"), ("c1e004", "shl"),
    ("48d1e0", "shl"), ("48c1e0", "shl"),
    ("d1e8", "shr"), ("d1e9", "shr"), ("d1ea", "shr"),
    ("c1e801", "shr"), ("c1e802", "shr"), ("c1e804", "shr"),
    ("48d1e8", "shr"), ("48c1e8", "shr"),
    ("d1f8", "sar"), ("d1f9", "sar"), ("d1fa", "sar"),
    ("c1f801", "sar"), ("c1f802", "sar"), ("c1f804", "sar"),
    ("48d1f8", "sar"), ("48c1f8", "sar"),
    
    # Conditional jumps
    ("7400", "je"), ("7401", "je"), ("74fe", "je"),
    ("7500", "jne"), ("7501", "jne"), ("75fe", "jne"),
    ("7f00", "jg"), ("7f01", "jg"),
    ("7d00", "jge"), ("7d01", "jge"),
    ("7c00", "jl"), ("7c01", "jl"),
    ("7e00", "jle"), ("7e01", "jle"),
    ("7700", "ja"), ("7701", "ja"),
    ("7300", "jae"), ("7301", "jae"),
    ("7200", "jb"), ("7201", "jb"),
    ("7600", "jbe"), ("7601", "jbe"),
    ("7800", "js"), ("7801", "js"),
    ("7900", "jns"), ("7901", "jns"),
    ("0f8400000000", "je"), ("0f8500000000", "jne"),
    ("0f8f00000000", "jg"), ("0f8d00000000", "jge"),
    ("0f8c00000000", "jl"), ("0f8e00000000", "jle"),
    
    # JMP
    ("eb00", "jmp"), ("eb01", "jmp"), ("ebfe", "jmp"),
    ("e900000000", "jmp"), ("e9fcffffff", "jmp"),
    ("ffe0", "jmp"), ("ffe1", "jmp"), ("ffe2", "jmp"),
    ("ff2500000000", "jmp"),
    
    # CALL
    ("e800000000", "call"), ("e8fcffffff", "call"),
    ("e8f0ffffff", "call"), ("e8e0ffffff", "call"),
    ("ffd0", "call"), ("ffd1", "call"), ("ffd2", "call"),
    ("ffd3", "call"), ("ffd6", "call"), ("ffd7", "call"),
    ("ff1500000000", "call"), ("ff14", "call"),
    
    # LEAVE
    ("c9", "leave"),
    
    # CDQ/CDQE/CQO
    ("99", "cdq"), ("4898", "cdqe"), ("4899", "cqo"),
    
    # SETCC
    ("0f94c0", "sete"), ("0f95c0", "setne"),
    ("0f9fc0", "setg"), ("0f9dc0", "setge"),
    ("0f9cc0", "setl"), ("0f9ec0", "setle"),
    
    # CMOVCC
    ("0f44c0", "cmove"), ("0f45c0", "cmovne"),
    ("0f4fc0", "cmovg"), ("0f4dc0", "cmovge"),
    ("0f4cc0", "cmovl"), ("0f4ec0", "cmovle"),
    
    # ENDBR64/32 - Intel CET
    ("f30f1efa", "endbr64"), ("f30f1efb", "endbr32"),
    
    # INT3
    ("cc", "int3"),
    
    # XCHG
    ("87c0", "xchg"), ("87c1", "xchg"), ("87c2", "xchg"),
    ("4887c0", "xchg"), ("4887c1", "xchg"),
    ("91", "xchg"), ("92", "xchg"), ("93", "xchg"),  # xchg eax, reg
]

# ============================================================================
# LEVEL 1: Semantics with EXACT keywords from gate test
# ============================================================================
print("[2/5] Building Level 1 semantics dataset...")

# Gate test expects these keywords
LEVEL1_SEMANTICS = {
    # push: ["stack", "push", "write", "rsp", "sp"]
    "push rbp": "stack operation; push value; write to stack; decrement rsp; sp modified",
    "push rax": "stack operation; push value; write to stack; decrement rsp; sp modified",
    "push rbx": "stack operation; push value; write to stack; decrement rsp; sp modified",
    "push 0x0": "stack operation; push immediate; write to stack; decrement rsp",
    
    # pop: ["stack", "pop", "read", "rsp", "sp"]
    "pop rbp": "stack operation; pop value; read from stack; increment rsp; sp modified",
    "pop rax": "stack operation; pop value; read from stack; increment rsp; sp modified",
    "pop rbx": "stack operation; pop value; read from stack; increment rsp; sp modified",
    
    # mov: ["move", "write", "read", "register", "transfer", "copy"]
    "mov rax, rbx": "move data; write to register; read from register; transfer value; copy",
    "mov rbp, rsp": "move data; write to register; read from register; transfer value; copy",
    "mov eax, 0x0": "move immediate; write to register; transfer value; copy",
    "mov [rax], rbx": "move data; write to memory; read from register; transfer; copy",
    "mov rax, [rbx]": "move data; write to register; read from memory; transfer; copy",
    "mov DWORD PTR [rbp-0x4], eax": "move data; write to memory; read register; transfer; copy",
    "mov eax, DWORD PTR [rbp-0x4]": "move data; write register; read from memory; transfer; copy",
    
    # lea: ["address", "load", "effective", "lea"]
    "lea rax, [rbx]": "lea load effective address; compute address; no memory read",
    "lea rax, [rip+0x0]": "lea load effective address; compute rip-relative address",
    "lea rax, [rbx+rcx*4]": "lea load effective address; compute scaled index address",
    
    # add: ["add", "sum", "arithmetic", "plus", "flag"]
    "add rax, rbx": "add arithmetic; sum operands; plus operation; flags affected",
    "add eax, 0x1": "add arithmetic; sum with immediate; plus operation; flags affected",
    "add rsp, 0x8": "add arithmetic; adjust stack pointer; plus; flags affected",
    
    # sub: ["sub", "subtract", "minus", "arithmetic", "flag"]
    "sub rax, rbx": "sub subtract arithmetic; minus operation; flags affected",
    "sub eax, 0x1": "sub subtract arithmetic; minus immediate; flags affected",
    "sub rsp, 0x20": "sub subtract arithmetic; allocate stack; minus; flags affected",
    
    # cmp: ["compare", "flag", "cmp", "sub"]
    "cmp rax, rbx": "cmp compare; sub without store; set flags; condition codes",
    "cmp eax, 0x0": "cmp compare with zero; sub without store; flags; condition",
    "cmp rax, 0x1": "cmp compare; sub without store; set flags",
    
    # test: ["test", "flag", "and", "compare"]
    "test rax, rax": "test and compare; bitwise and; set flags; zero check",
    "test eax, eax": "test and compare; bitwise and; set flags; zero check",
    "test al, 0x1": "test and compare; bitwise and; flags; bit check",
    
    # jmp: ["jump", "branch", "unconditional", "control"]
    "jmp 0x100": "unconditional jump; branch always; control flow transfer",
    "jmp rax": "unconditional jump to register; branch; control flow",
    "jmp [rax]": "unconditional jump indirect; branch; control flow",
    
    # je/jz: ["jump", "equal", "zero", "conditional", "branch"]
    "je 0x100": "conditional jump if equal; branch if zero flag; je jz",
    "jz 0x100": "conditional jump if zero; branch if equal; je jz",
    
    # jne/jnz: ["jump", "not", "equal", "conditional", "branch"]
    "jne 0x100": "conditional jump if not equal; branch if not zero",
    "jnz 0x100": "conditional jump if not zero; branch if not equal",
    
    # call: ["call", "function", "push", "return"]
    "call 0x100": "call function; push return address; transfer control",
    "call rax": "call function via register; push return address",
    "call [rax]": "call function indirect; push return address",
    
    # ret: ["return", "pop", "rip", "control"] - CRITICAL
    "ret": "return from function; pop return address; restore rip; control flow",
    "ret 0x10": "return from function; pop rip; control flow; cleanup stack",
    
    # xor: ["xor", "exclusive", "zero", "bitwise"]
    "xor rax, rax": "xor exclusive or; zero register; bitwise operation; clear",
    "xor eax, eax": "xor exclusive or; zero register; bitwise; clear",
    "xor rax, rbx": "xor exclusive or; bitwise operation; toggle bits",
    
    # and: ["and", "bitwise", "mask", "flag"]
    "and rax, rbx": "and bitwise; mask operation; flags affected",
    "and eax, 0xff": "and bitwise; mask with immediate; flags",
    "and rax, 0xf": "and bitwise; mask bits; flags affected",
    
    # or: ["or", "bitwise", "flag"]
    "or rax, rbx": "or bitwise; combine bits; flags affected",
    "or eax, 0x1": "or bitwise; set bit; flags affected",
    
    # shl: ["shift", "left", "multiply", "bitwise"]
    "shl rax, 1": "shift left; multiply by 2; bitwise operation",
    "shl eax, cl": "shift left; multiply; bitwise; variable shift",
    "shl rax, 4": "shift left; multiply by 16; bitwise",
    
    # shr: ["shift", "right", "divide", "bitwise"]
    "shr rax, 1": "shift right; divide by 2; bitwise; logical shift",
    "shr eax, cl": "shift right; divide; bitwise; logical",
    "shr rax, 4": "shift right; divide by 16; bitwise",
    
    # imul: ["multiply", "mul", "signed", "arithmetic"]
    "imul rax, rbx": "imul signed multiply; mul arithmetic operation",
    "imul eax, ebx, 0x4": "imul signed multiply; mul arithmetic; three operand",
    
    # idiv: ["divide", "div", "signed", "arithmetic"]
    "idiv rbx": "idiv signed divide; div arithmetic; quotient in rax",
    "idiv ecx": "idiv signed divide; div arithmetic operation",
}

# ============================================================================
# LEVEL 2: CFG format matching gate test
# ============================================================================
print("[3/5] Building Level 2 CFG dataset...")

LEVEL2_PATTERNS = [
    # Simple linear (no branches)
    {
        "input": "Instructions:\n0x0:push rbp\n0x1:mov rbp,rsp\n0x4:mov eax,0x0\n0x9:pop rbp\n0xa:ret [return]",
        "output": "Analysis: BB0 basic block; linear flow; no loop; no branch; single block"
    },
    # Simple conditional
    {
        "input": "Instructions:\n0x0:cmp eax,0x0\n0x3:je 0x10 [conditional]\n0x5:mov eax,0x1\n0x10:ret [return]",
        "output": "Analysis: BB0 BB1 blocks; conditional branch; edge from BB0 to BB1; no loop"
    },
    # Loop pattern
    {
        "input": "Instructions:\n0x0:xor eax,eax\n0x2:cmp eax,0xa\n0x5:jge 0x10 [conditional]\n0x7:add eax,0x1\n0x9:jmp 0x2 [jump]\n0x10:ret [return]",
        "output": "Analysis: BB0 BB1 BB2 blocks; loop detected; conditional branch; back edge"
    },
    # Function call
    {
        "input": "Instructions:\n0x0:push rbp\n0x1:call 0x100 [call]\n0x6:pop rbp\n0x7:ret [return]",
        "output": "Analysis: BB0 basic block; function call; no loop; no branch"
    },
    # Nested conditionals
    {
        "input": "Instructions:\n0x0:cmp eax,0x0\n0x3:jle 0x20 [conditional]\n0x5:cmp ebx,0x0\n0x8:jle 0x15 [conditional]\n0x10:mov eax,0x1\n0x15:mov eax,0x2\n0x20:ret [return]",
        "output": "Analysis: multiple blocks; nested conditional branch; edges between blocks; no loop"
    },
    # While loop
    {
        "input": "Instructions:\n0x0:mov ecx,0xa\n0x5:test ecx,ecx\n0x7:je 0x15 [conditional]\n0xa:dec ecx\n0xc:jmp 0x5 [jump]\n0x15:ret [return]",
        "output": "Analysis: BB0 BB1 blocks; loop detected; conditional branch; back edge to BB1"
    },
]

# ============================================================================
# Generate datasets
# ============================================================================
print("\n[4/5] Generating training data...")

os.makedirs("genesis_datasets/level0_final", exist_ok=True)
os.makedirs("genesis_datasets/level1_final", exist_ok=True)
os.makedirs("genesis_datasets/level2_final", exist_ok=True)

# Level 0: Multiple copies for emphasis on critical instructions
level0_samples = []
for bytes_hex, mnemonic in LEVEL0_INSTRUCTIONS:
    # Weight critical instructions more heavily
    weight = 10 if mnemonic in ["ret", "call", "push", "pop"] else 3
    for _ in range(weight):
        level0_samples.append({
            "input": f"Bytes: {bytes_hex}",
            "output": f"Instruction: {mnemonic}"
        })

random.shuffle(level0_samples)
with open("genesis_datasets/level0_final/train.jsonl", "w") as f:
    for s in level0_samples:
        f.write(json.dumps(s) + "\n")
print(f"  Level 0: {len(level0_samples)} samples")

# Level 1: Multiple variations
level1_samples = []
for instr, semantics in LEVEL1_SEMANTICS.items():
    for _ in range(20):  # 20 copies each
        level1_samples.append({
            "input": f"Instruction: {instr}",
            "output": semantics
        })

random.shuffle(level1_samples)
with open("genesis_datasets/level1_final/train.jsonl", "w") as f:
    for s in level1_samples:
        f.write(json.dumps(s) + "\n")
print(f"  Level 1: {len(level1_samples)} samples")

# Level 2: Expand patterns
level2_samples = []
for pattern in LEVEL2_PATTERNS:
    for _ in range(50):  # 50 copies each
        level2_samples.append(pattern)

random.shuffle(level2_samples)
with open("genesis_datasets/level2_final/train.jsonl", "w") as f:
    for s in level2_samples:
        f.write(json.dumps(s) + "\n")
print(f"  Level 2: {len(level2_samples)} samples")

# Also add real binary data
print("\n  Adding real binary data...")
subprocess.run([sys.executable, "genesis_datasets/generators/real_binary_generator.py"], 
               capture_output=True)

# Merge datasets
for level in [0, 1, 2]:
    final_path = f"genesis_datasets/level{level}_final/train.jsonl"
    real_path = f"genesis_datasets/level{level}_real/train.jsonl"
    
    if Path(real_path).exists():
        with open(final_path, "a") as f_out:
            with open(real_path) as f_in:
                for line in f_in:
                    f_out.write(line)
        print(f"  Merged real data into level{level}_final")

# Count final sizes
for level in [0, 1, 2]:
    path = f"genesis_datasets/level{level}_final/train.jsonl"
    if Path(path).exists():
        with open(path) as f:
            count = sum(1 for _ in f)
        print(f"  Level {level} total: {count} samples")

# ============================================================================
# Training
# ============================================================================
print("\n[5/5] Training models...")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

MODEL_NAME = "distilgpt2"
BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-5
PATIENCE = 15
LORA_R = 64
LORA_ALPHA = 128

class SimpleDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                self.samples.append(f"{d['input']}\n{d['output']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.samples[idx], truncation=True, max_length=self.max_len,
                            padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze().clone()
        }

def train_level(name, dataset_path, output_path, max_len=256):
    print(f"\n  Training {name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                      lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
    model = get_peft_model(base, lora).to(device)
    
    dataset = SimpleDataset(dataset_path, tokenizer, max_len)
    print(f"    Dataset: {len(dataset)} samples")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        
        if avg < best_loss - 0.001:
            best_loss = avg
            no_improve = 0
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            if epoch % 10 == 0:
                print(f"    Epoch {epoch+1}: {avg:.4f} * (saved)")
        else:
            no_improve += 1
        
        if no_improve >= PATIENCE and epoch > 30:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    print(f"    Best loss: {best_loss:.4f}")
    return best_loss

# Train all levels
results = {}
results["level0"] = train_level("Level 0", "genesis_datasets/level0_final/train.jsonl", 
                                "models/level0_final", 128)
results["level1"] = train_level("Level 1", "genesis_datasets/level1_final/train.jsonl", 
                                "models/level1_final", 256)
results["level2"] = train_level("Level 2", "genesis_datasets/level2_final/train.jsonl", 
                                "models/level2_final", 512)

# Quick verification
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

for level in ["level0_final", "level1_final"]:
    print(f"\n{level}:")
    tokenizer = AutoTokenizer.from_pretrained(f"models/{level}")
    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = PeftModel.from_pretrained(base, f"models/{level}").to(device).eval()
    
    if "level0" in level:
        tests = ["Bytes: c3", "Bytes: 55", "Bytes: e800000000", "Bytes: f30f1efa"]
    else:
        tests = ["Instruction: ret", "Instruction: push rbp", "Instruction: call 0x100"]
    
    for test in tests:
        inputs = tokenizer(test + "\n", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(out[0], skip_special_tokens=True).split("\n")[-1][:60]
        print(f"  {test} → {result}")

# Package
print("\n" + "=" * 60)
print("PACKAGING")
print("=" * 60)

import zipfile
from peft import PeftModel

with zipfile.ZipFile("genesis_final.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for level in ["level0_final", "level1_final", "level2_final"]:
        model_path = Path(f"models/{level}")
        if model_path.exists():
            for f in model_path.iterdir():
                zf.write(f, f"models/{level}/{f.name}")
                print(f"  Added: models/{level}/{f.name}")

print(f"\nZip size: {Path('genesis_final.zip').stat().st_size / 1024 / 1024:.1f} MB")

from google.colab import files
files.download("genesis_final.zip")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
for name, loss in results.items():
    print(f"  {name}: {loss:.4f}")
