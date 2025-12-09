#!/usr/bin/env python3
"""
Create a balanced, high-quality Level 0 training dataset v2.
Key improvement: Generate VARIATIONS, not duplicates.
"""

import json
import random
from pathlib import Path
from collections import Counter

# Instruction patterns with operand variations
# Format: (base_pattern, mnemonic, variation_type)
# variation_type: "fixed" = use as-is, "reg" = vary register, "imm" = vary immediate

PATTERNS = [
    # === SINGLE-BYTE FIXED ===
    ("90", "nop", "fixed"),
    ("c3", "ret", "fixed"),
    ("cc", "int3", "fixed"),
    ("c9", "leave", "fixed"),
    ("f4", "hlt", "fixed"),
    ("9c", "pushf", "fixed"),
    ("9d", "popf", "fixed"),
    ("fc", "cld", "fixed"),
    ("fd", "std", "fixed"),
    ("cb", "retf", "fixed"),
    
    # === PUSH register (0x50-0x57) ===
    ("5X", "push", "reg8"),  # X = 0-7
    ("415X", "push", "reg8"),  # r8-r15
    
    # === POP register (0x58-0x5f) ===
    ("5Y", "pop", "reg8"),  # Y = 8-f (0x58-0x5f)
    ("415Y", "pop", "reg8"),  # r8-r15
    
    # === PUSH immediate ===
    ("6aXX", "push", "imm8"),
    ("68XXXXXXXX", "push", "imm32"),
    
    # === SYSCALL/INT ===
    ("0f05", "syscall", "fixed"),
    ("cdXX", "int", "imm8"),
    
    # === XOR reg, reg ===
    ("31c0", "xor", "fixed"),  # xor eax, eax
    ("31c9", "xor", "fixed"),  # xor ecx, ecx
    ("31d2", "xor", "fixed"),  # xor edx, edx
    ("31db", "xor", "fixed"),  # xor ebx, ebx
    ("31f6", "xor", "fixed"),  # xor esi, esi
    ("31ff", "xor", "fixed"),  # xor edi, edi
    ("4831c0", "xor", "fixed"),  # xor rax, rax
    ("4831c9", "xor", "fixed"),  # xor rcx, rcx
    ("4831d2", "xor", "fixed"),  # xor rdx, rdx
    ("33c0", "xor", "fixed"),  # alt encoding
    ("33c9", "xor", "fixed"),
    ("33d2", "xor", "fixed"),
    
    # === ADD variants ===
    ("01c0", "add", "fixed"),  # add eax, eax
    ("01c8", "add", "fixed"),  # add eax, ecx
    ("01d0", "add", "fixed"),  # add eax, edx
    ("01d8", "add", "fixed"),  # add eax, ebx
    ("4801c0", "add", "fixed"),  # add rax, rax
    ("83c0XX", "add", "imm8"),  # add eax, imm8
    ("4883c0XX", "add", "imm8"),  # add rax, imm8
    ("05XXXXXXXX", "add", "imm32"),  # add eax, imm32
    
    # === SUB variants ===
    ("29c0", "sub", "fixed"),
    ("29c8", "sub", "fixed"),
    ("29d0", "sub", "fixed"),
    ("2bc0", "sub", "fixed"),  # alt encoding
    ("83e8XX", "sub", "imm8"),
    ("4883e8XX", "sub", "imm8"),
    ("4883ecXX", "sub", "imm8"),  # sub rsp, imm (common)
    
    # === AND/OR ===
    ("21c0", "and", "fixed"),
    ("21c8", "and", "fixed"),
    ("21d0", "and", "fixed"),
    ("4821c0", "and", "fixed"),
    ("83e0XX", "and", "imm8"),
    ("09c0", "or", "fixed"),
    ("09c8", "or", "fixed"),
    ("09d0", "or", "fixed"),
    ("4809c0", "or", "fixed"),
    ("83c8XX", "or", "imm8"),
    
    # === NOT/NEG ===
    ("f7d0", "not", "fixed"),
    ("f7d1", "not", "fixed"),
    ("f7d2", "not", "fixed"),
    ("f7d3", "not", "fixed"),
    ("48f7d0", "not", "fixed"),
    ("f7d8", "neg", "fixed"),
    ("f7d9", "neg", "fixed"),
    ("f7da", "neg", "fixed"),
    ("f7db", "neg", "fixed"),
    ("48f7d8", "neg", "fixed"),
    
    # === SHIFTS ===
    ("d1e0", "shl", "fixed"),
    ("d1e1", "shl", "fixed"),
    ("d1e2", "shl", "fixed"),
    ("d1e3", "shl", "fixed"),
    ("c1e0XX", "shl", "imm8"),
    ("d1e8", "shr", "fixed"),
    ("d1e9", "shr", "fixed"),
    ("d1ea", "shr", "fixed"),
    ("c1e8XX", "shr", "imm8"),
    ("d1f8", "sar", "fixed"),
    ("d1f9", "sar", "fixed"),
    ("d1fa", "sar", "fixed"),
    ("c1f8XX", "sar", "imm8"),
    
    # === MOV reg, reg ===
    ("89c0", "mov", "fixed"),
    ("89c1", "mov", "fixed"),
    ("89c2", "mov", "fixed"),
    ("89c8", "mov", "fixed"),
    ("89d0", "mov", "fixed"),
    ("89d8", "mov", "fixed"),
    ("4889c0", "mov", "fixed"),
    ("4889e5", "mov", "fixed"),  # mov rbp, rsp
    ("4889ec", "mov", "fixed"),  # mov rsp, rbp
    
    # === MOV reg, [mem] ===
    ("8b00", "mov", "fixed"),
    ("8b01", "mov", "fixed"),
    ("8b02", "mov", "fixed"),
    ("8b03", "mov", "fixed"),
    ("8b40XX", "mov", "imm8"),  # mov eax, [rax+imm8]
    ("8b4424XX", "mov", "imm8"),  # mov eax, [rsp+imm8]
    ("890424", "mov", "fixed"),  # mov [rsp], eax
    
    # === MOV reg, imm ===
    ("b0XX", "mov", "imm8"),  # mov al, imm8
    ("b1XX", "mov", "imm8"),  # mov cl, imm8
    ("b8XXXXXXXX", "mov", "imm32"),  # mov eax, imm32
    ("b9XXXXXXXX", "mov", "imm32"),  # mov ecx, imm32
    ("48c7c0XXXXXXXX", "mov", "imm32"),  # mov rax, imm32
    ("c60424XX", "mov", "imm8"),  # mov byte [rsp], imm8
    
    # === CMP ===
    ("39c0", "cmp", "fixed"),
    ("39c8", "cmp", "fixed"),
    ("39d0", "cmp", "fixed"),
    ("39d8", "cmp", "fixed"),
    ("4839c0", "cmp", "fixed"),
    ("83f8XX", "cmp", "imm8"),
    ("3cXX", "cmp", "imm8"),  # cmp al, imm8
    
    # === TEST ===
    ("85c0", "test", "fixed"),
    ("85c9", "test", "fixed"),
    ("85d2", "test", "fixed"),
    ("85db", "test", "fixed"),
    ("4885c0", "test", "fixed"),
    ("a8XX", "test", "imm8"),  # test al, imm8
    
    # === JMP ===
    ("ebXX", "jmp", "imm8"),  # jmp short
    ("e9XXXXXXXX", "jmp", "imm32"),  # jmp near
    ("ff20", "jmp", "fixed"),  # jmp [rax]
    ("ff21", "jmp", "fixed"),  # jmp [rcx]
    ("ffe0", "jmp", "fixed"),  # jmp rax
    ("ffe1", "jmp", "fixed"),  # jmp rcx
    
    # === Jcc ===
    ("74XX", "je", "imm8"),
    ("75XX", "jne", "imm8"),
    ("7cXX", "jl", "imm8"),
    ("7dXX", "jge", "imm8"),
    ("7eXX", "jle", "imm8"),
    ("7fXX", "jg", "imm8"),
    ("72XX", "jb", "imm8"),
    ("73XX", "jae", "imm8"),
    ("76XX", "jbe", "imm8"),
    ("77XX", "ja", "imm8"),
    ("78XX", "js", "imm8"),
    ("79XX", "jns", "imm8"),
    
    # === CALL ===
    ("e8XXXXXXXX", "call", "imm32"),
    ("ff10", "call", "fixed"),
    ("ff11", "call", "fixed"),
    ("ffd0", "call", "fixed"),
    ("ffd1", "call", "fixed"),
    
    # === LEA ===
    ("8d00", "lea", "fixed"),
    ("8d01", "lea", "fixed"),
    ("8d0400", "lea", "fixed"),  # lea eax, [rax+rax]
    ("8d0480", "lea", "fixed"),  # lea eax, [rax+rax*4]
    ("8d04c5XXXXXXXX", "lea", "imm32"),
    ("488d4424XX", "lea", "imm8"),  # lea rax, [rsp+imm8]
    
    # === STRING OPS ===
    ("a4", "movsb", "fixed"),
    ("a5", "movsd", "fixed"),
    ("66a5", "movsw", "fixed"),
    ("48a5", "movsq", "fixed"),
    ("aa", "stosb", "fixed"),
    ("ab", "stosd", "fixed"),
    ("66ab", "stosw", "fixed"),
    ("48ab", "stosq", "fixed"),
    ("ac", "lodsb", "fixed"),
    ("ad", "lodsd", "fixed"),
    ("66ad", "lodsw", "fixed"),
    ("48ad", "lodsq", "fixed"),
    ("ae", "scasb", "fixed"),
    ("af", "scasd", "fixed"),
    ("66af", "scasw", "fixed"),
    ("48af", "scasq", "fixed"),
    
    # === REP prefixes ===
    ("f3a4", "rep", "fixed"),
    ("f3a5", "rep", "fixed"),
    ("f3aa", "rep", "fixed"),
    ("f3ab", "rep", "fixed"),
    ("f3ae", "rep", "fixed"),
    ("f2ae", "repne", "fixed"),
    
    # === XCHG ===
    ("91", "xchg", "fixed"),
    ("92", "xchg", "fixed"),
    ("93", "xchg", "fixed"),
    ("94", "xchg", "fixed"),
    ("95", "xchg", "fixed"),
    ("96", "xchg", "fixed"),
    ("97", "xchg", "fixed"),
    ("87c1", "xchg", "fixed"),
    ("87c2", "xchg", "fixed"),
    ("87d0", "xchg", "fixed"),
    
    # === BSWAP ===
    ("0fc8", "bswap", "fixed"),
    ("0fc9", "bswap", "fixed"),
    ("0fca", "bswap", "fixed"),
    ("0fcb", "bswap", "fixed"),
    ("0fcc", "bswap", "fixed"),
    ("0fcd", "bswap", "fixed"),
    ("0fce", "bswap", "fixed"),
    ("0fcf", "bswap", "fixed"),
    ("480fc8", "bswap", "fixed"),
    
    # === BSF/BSR ===
    ("0fbcc0", "bsf", "fixed"),
    ("0fbcc1", "bsf", "fixed"),
    ("0fbcc2", "bsf", "fixed"),
    ("0fbdc0", "bsr", "fixed"),
    ("0fbdc1", "bsr", "fixed"),
    ("0fbdc2", "bsr", "fixed"),
    
    # === CPUID/RDTSC ===
    ("0fa2", "cpuid", "fixed"),
    ("0f31", "rdtsc", "fixed"),
    
    # === RET variants ===
    ("c20000", "ret", "fixed"),
    ("c20200", "ret", "fixed"),
    ("c20400", "ret", "fixed"),
    ("c20800", "ret", "fixed"),
    
    # === INC/DEC ===
    ("ffc0", "inc", "fixed"),
    ("ffc1", "inc", "fixed"),
    ("ffc2", "inc", "fixed"),
    ("ffc3", "inc", "fixed"),
    ("48ffc0", "inc", "fixed"),
    ("48ffc1", "inc", "fixed"),
    ("ffc8", "dec", "fixed"),
    ("ffc9", "dec", "fixed"),
    ("ffca", "dec", "fixed"),
    ("ffcb", "dec", "fixed"),
    ("48ffc8", "dec", "fixed"),
    
    # === MUL/DIV ===
    ("f7e0", "mul", "fixed"),
    ("f7e1", "mul", "fixed"),
    ("f7e2", "mul", "fixed"),
    ("f7e3", "mul", "fixed"),
    ("f7f0", "div", "fixed"),
    ("f7f1", "div", "fixed"),
    ("f7f2", "div", "fixed"),
    ("f7f3", "div", "fixed"),
    ("f7e8", "imul", "fixed"),
    ("f7f8", "idiv", "fixed"),
    
    # === SETCC ===
    ("0f94c0", "sete", "fixed"),
    ("0f94c1", "sete", "fixed"),
    ("0f95c0", "setne", "fixed"),
    ("0f95c1", "setne", "fixed"),
    ("0f9cc0", "setl", "fixed"),
    ("0f9fc0", "setg", "fixed"),
    ("0f92c0", "setb", "fixed"),
    ("0f93c0", "setae", "fixed"),
    
    # === CMOV ===
    ("0f44c1", "cmove", "fixed"),
    ("0f44c2", "cmove", "fixed"),
    ("0f45c1", "cmovne", "fixed"),
    ("0f45c2", "cmovne", "fixed"),
    ("0f4cc1", "cmovl", "fixed"),
    ("0f4fc1", "cmovg", "fixed"),
    
    # === MOVZX/MOVSX ===
    ("0fb6c0", "movzx", "fixed"),
    ("0fb6c1", "movzx", "fixed"),
    ("0fb6c2", "movzx", "fixed"),
    ("0fb7c0", "movzx", "fixed"),
    ("0fbec0", "movsx", "fixed"),
    ("0fbec1", "movsx", "fixed"),
    ("0fbfc0", "movsx", "fixed"),
    ("4863c0", "movsxd", "fixed"),
    ("4863c1", "movsxd", "fixed"),
]

def expand_pattern(pattern, var_type):
    """Expand a pattern with variations."""
    results = []
    
    if var_type == "fixed":
        results.append(pattern)
    
    elif var_type == "reg8":
        # Replace X with 0-7 or Y with 8-f
        if 'X' in pattern:
            for i in range(8):
                results.append(pattern.replace('X', format(i, 'x')))
        elif 'Y' in pattern:
            for i in range(8, 16):
                results.append(pattern.replace('Y', format(i, 'x')))
    
    elif var_type == "imm8":
        # Replace XX with various 8-bit immediates
        imm_values = ['00', '01', '02', '04', '08', '10', '20', '40', '80', 'ff', 
                      '7f', '3f', '0f', 'fe', 'f0', 'aa', '55', '33', 'cc']
        for imm in imm_values:
            results.append(pattern.replace('XX', imm))
    
    elif var_type == "imm32":
        # Replace XXXXXXXX with various 32-bit immediates
        imm_values = ['00000000', '01000000', 'ffffffff', '00000001', 
                      '10000000', '00010000', 'deadbeef', 'cafebabe']
        for imm in imm_values:
            results.append(pattern.replace('XXXXXXXX', imm))
    
    return results


def create_dataset():
    """Create balanced training dataset with variations."""
    random.seed(42)
    
    samples = []
    all_patterns = []
    
    # Expand all patterns
    for pattern, mnemonic, var_type in PATTERNS:
        expanded = expand_pattern(pattern, var_type)
        for p in expanded:
            all_patterns.append((p, mnemonic))
    
    print(f"Expanded to {len(all_patterns)} unique patterns")
    
    # Count patterns per mnemonic
    mnemonic_patterns = Counter(m for _, m in all_patterns)
    print(f"Unique mnemonics: {len(mnemonic_patterns)}")
    
    # Target ~100 samples per pattern, but balance by mnemonic
    target_per_mnemonic = 200
    
    for pattern, mnemonic in all_patterns:
        patterns_for_this_mnemonic = mnemonic_patterns[mnemonic]
        copies = max(2, target_per_mnemonic // patterns_for_this_mnemonic)
        
        for _ in range(copies):
            samples.append({
                "raw_bytes": pattern.lower(),
                "expected_mnemonic": mnemonic,
                "source": "curated",
                "is_valid": True,
            })
    
    print(f"Valid samples: {len(samples)}")
    
    # Add adversarial samples
    adversarial = [
        # Incomplete prefixes
        ("0f", "Incomplete 2-byte opcode"),
        ("66", "Lone operand size prefix"),
        ("67", "Lone address size prefix"),
        ("f0", "Lone LOCK prefix"),
        ("f2", "Lone REPNE prefix"),
        ("f3", "Lone REP prefix"),
        ("48", "Lone REX.W prefix"),
        ("4c", "Lone REX.WR prefix"),
        ("660f", "Incomplete with prefix"),
        ("480f", "Incomplete with REX"),
    ]
    
    # Add each adversarial pattern
    for pattern, desc in adversarial:
        for _ in range(50):  # 50 copies each
            samples.append({
                "raw_bytes": pattern,
                "expected_mnemonic": "unknown",
                "source": "adversarial",
                "is_valid": False,
            })
    
    # Add random garbage
    for _ in range(200):
        length = random.randint(8, 20)
        rand_bytes = ''.join(random.choice('0123456789abcdef') for _ in range(length * 2))
        samples.append({
            "raw_bytes": rand_bytes,
            "expected_mnemonic": "unknown",
            "source": "adversarial_random",
            "is_valid": False,
        })
    
    print(f"Total samples: {len(samples)}")
    
    # Shuffle
    random.shuffle(samples)
    
    # Verify
    final_mnemonics = Counter(s["expected_mnemonic"] for s in samples)
    unique_bytes = len(set(s["raw_bytes"] for s in samples))
    
    print(f"\nUnique byte sequences: {unique_bytes}")
    print(f"Duplicate rate: {100*(len(samples)-unique_bytes)/len(samples):.1f}%")
    
    print("\n=== Mnemonic Distribution (bottom 15) ===")
    for m, c in final_mnemonics.most_common()[-15:]:
        print(f"  {m}: {c}")
    
    return samples


def main():
    output_path = Path("genesis_datasets/level0/train.jsonl")
    
    # Create new dataset
    samples = create_dataset()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\n✅ Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
