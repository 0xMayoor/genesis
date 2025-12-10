#!/usr/bin/env python3
"""
Create a balanced, high-quality Level 0 training dataset.
Target: 100% accuracy on all exam patterns.
"""

import json
import random
from pathlib import Path
from collections import Counter

# All instruction patterns we need to learn - SINGLE INSTRUCTIONS ONLY
# Format: (hex_bytes, mnemonic)
INSTRUCTION_PATTERNS = [
    # === BASIC SINGLE-BYTE ===
    ("90", "nop"),
    ("c3", "ret"),
    ("cc", "int3"),
    ("c9", "leave"),
    ("f4", "hlt"),
    ("9c", "pushf"),
    ("9d", "popf"),
    ("fc", "cld"),
    ("fd", "std"),
    
    # === PUSH/POP REGISTERS (single byte) ===
    ("50", "push"),  # rax
    ("51", "push"),  # rcx
    ("52", "push"),  # rdx
    ("53", "push"),  # rbx
    ("54", "push"),  # rsp
    ("55", "push"),  # rbp
    ("56", "push"),  # rsi
    ("57", "push"),  # rdi
    ("58", "pop"),   # rax
    ("59", "pop"),   # rcx
    ("5a", "pop"),   # rdx
    ("5b", "pop"),   # rbx
    ("5c", "pop"),   # rsp
    ("5d", "pop"),   # rbp
    ("5e", "pop"),   # rsi
    ("5f", "pop"),   # rdi
    
    # === PUSH/POP with REX (r8-r15) ===
    ("4150", "push"),  # r8
    ("4151", "push"),  # r9
    ("4152", "push"),  # r10
    ("4153", "push"),  # r11
    ("4154", "push"),  # r12
    ("4155", "push"),  # r13
    ("4156", "push"),  # r14
    ("4157", "push"),  # r15
    ("4158", "pop"),   # r8
    ("4159", "pop"),   # r9
    ("415a", "pop"),   # r10
    ("415b", "pop"),   # r11
    ("415c", "pop"),   # r12
    ("415d", "pop"),   # r13
    ("415e", "pop"),   # r14
    ("415f", "pop"),   # r15
    
    # === SYSCALL/INT ===
    ("0f05", "syscall"),
    ("cd80", "int"),      # int 0x80
    ("cd03", "int"),      # int 3
    ("cd21", "int"),      # int 0x21 (DOS)
    
    # === XOR (zero registers) ===
    ("31c0", "xor"),      # xor eax, eax
    ("31db", "xor"),      # xor ebx, ebx
    ("31c9", "xor"),      # xor ecx, ecx
    ("31d2", "xor"),      # xor edx, edx
    ("31f6", "xor"),      # xor esi, esi
    ("31ff", "xor"),      # xor edi, edi
    ("4831c0", "xor"),    # xor rax, rax
    ("4831db", "xor"),    # xor rbx, rbx
    ("4831c9", "xor"),    # xor rcx, rcx
    ("4831d2", "xor"),    # xor rdx, rdx
    ("33c0", "xor"),      # xor eax, eax (alt encoding)
    
    # === ADD ===
    ("01c0", "add"),      # add eax, eax
    ("01d8", "add"),      # add eax, ebx
    ("01c8", "add"),      # add eax, ecx
    ("4801c0", "add"),    # add rax, rax
    ("83c001", "add"),    # add eax, 1
    ("83c010", "add"),    # add eax, 16
    ("4883c001", "add"),  # add rax, 1
    ("4883c008", "add"),  # add rax, 8
    ("0500000001", "add"), # add eax, 0x01000000
    
    # === SUB ===
    ("29c0", "sub"),      # sub eax, eax
    ("29d8", "sub"),      # sub eax, ebx
    ("2bc0", "sub"),      # sub eax, eax (alt)
    ("83e801", "sub"),    # sub eax, 1
    ("4883e808", "sub"),  # sub rax, 8
    ("4883ec10", "sub"),  # sub rsp, 16 (stack alloc)
    ("4883ec20", "sub"),  # sub rsp, 32
    
    # === AND ===
    ("21c0", "and"),      # and eax, eax
    ("21d8", "and"),      # and eax, ebx
    ("4821c0", "and"),    # and rax, rax
    ("83e00f", "and"),    # and eax, 0xf
    ("83e0ff", "and"),    # and eax, 0xff
    
    # === OR ===
    ("09c0", "or"),       # or eax, eax
    ("09d8", "or"),       # or eax, ebx
    ("4809c0", "or"),     # or rax, rax
    ("83c801", "or"),     # or eax, 1
    
    # === NOT/NEG ===
    ("f7d0", "not"),      # not eax
    ("f7d3", "not"),      # not ebx
    ("48f7d0", "not"),    # not rax
    ("f7d8", "neg"),      # neg eax
    ("f7db", "neg"),      # neg ebx
    ("48f7d8", "neg"),    # neg rax
    
    # === SHIFTS ===
    ("d1e0", "shl"),      # shl eax, 1
    ("d1e3", "shl"),      # shl ebx, 1
    ("c1e004", "shl"),    # shl eax, 4
    ("d1e8", "shr"),      # shr eax, 1
    ("d1eb", "shr"),      # shr ebx, 1
    ("c1e804", "shr"),    # shr eax, 4
    ("d1f8", "sar"),      # sar eax, 1
    ("d1fb", "sar"),      # sar ebx, 1
    ("c1f804", "sar"),    # sar eax, 4
    
    # === MOV ===
    ("89c0", "mov"),      # mov eax, eax
    ("89d8", "mov"),      # mov eax, ebx
    ("89c8", "mov"),      # mov eax, ecx
    ("4889c0", "mov"),    # mov rax, rax
    ("4889e5", "mov"),    # mov rbp, rsp (prologue)
    ("4889ec", "mov"),    # mov rsp, rbp
    ("8b00", "mov"),      # mov eax, [rax]
    ("8b4004", "mov"),    # mov eax, [rax+4]
    ("890424", "mov"),    # mov [rsp], eax
    ("b800000000", "mov"), # mov eax, 0
    ("b801000000", "mov"), # mov eax, 1
    ("b03b", "mov"),      # mov al, 59
    ("b8ffffffff", "mov"), # mov eax, -1
    ("48c7c000000000", "mov"), # mov rax, 0
    ("c6042490", "mov"),  # mov byte [rsp], 0x90
    
    # === CMP ===
    ("39c0", "cmp"),      # cmp eax, eax
    ("39d8", "cmp"),      # cmp eax, ebx
    ("4839c0", "cmp"),    # cmp rax, rax
    ("83f800", "cmp"),    # cmp eax, 0
    ("83f801", "cmp"),    # cmp eax, 1
    ("3c00", "cmp"),      # cmp al, 0
    
    # === TEST ===
    ("85c0", "test"),     # test eax, eax
    ("85db", "test"),     # test ebx, ebx
    ("4885c0", "test"),   # test rax, rax
    ("a801", "test"),     # test al, 1
    ("a900010000", "test"), # test eax, 0x100
    
    # === JMP ===
    ("eb00", "jmp"),      # jmp short +0
    ("eb10", "jmp"),      # jmp short +16
    ("ebfe", "jmp"),      # jmp short -2 (infinite loop)
    ("eb01", "jmp"),      # jmp short +1
    ("e900000000", "jmp"), # jmp near +0
    ("ff20", "jmp"),      # jmp [rax]
    ("ffe0", "jmp"),      # jmp rax
    
    # === Jcc (conditional jumps) ===
    ("7400", "je"),       # je/jz short
    ("7410", "je"),
    ("7500", "jne"),      # jne/jnz short
    ("7510", "jne"),
    ("7c00", "jl"),       # jl short
    ("7c10", "jl"),
    ("7f00", "jg"),       # jg short
    ("7f10", "jg"),
    ("7200", "jb"),       # jb/jc short
    ("7300", "jae"),      # jae/jnc short
    ("7600", "jbe"),      # jbe short
    ("7700", "ja"),       # ja short
    ("7800", "js"),       # js short
    ("7900", "jns"),      # jns short
    
    # === CALL ===
    ("e800000000", "call"), # call near +0
    ("ff10", "call"),     # call [rax]
    ("ffd0", "call"),     # call rax
    
    # === LEA ===
    ("8d00", "lea"),      # lea eax, [rax]
    ("8d0400", "lea"),    # lea eax, [rax+rax]
    ("8d0480", "lea"),    # lea eax, [rax+rax*4]
    ("488d0425", "lea"),  # lea rax, [...]
    ("488d4c2408", "lea"), # lea rcx, [rsp+8]
    
    # === STRING OPS ===
    ("a4", "movsb"),
    ("a5", "movsd"),
    ("66a5", "movsw"),
    ("48a5", "movsq"),
    ("aa", "stosb"),
    ("ab", "stosd"),
    ("66ab", "stosw"),
    ("48ab", "stosq"),
    ("ac", "lodsb"),
    ("ad", "lodsd"),
    ("66ad", "lodsw"),
    ("48ad", "lodsq"),
    ("ae", "scasb"),
    ("af", "scasd"),
    ("66af", "scasw"),
    ("48af", "scasq"),
    
    # === REP prefixes ===
    ("f3a4", "rep"),      # rep movsb
    ("f3a5", "rep"),      # rep movsd
    ("f3ab", "rep"),      # rep stosd
    ("f3ae", "rep"),      # rep scasb (repz)
    ("f2ae", "repne"),    # repne scasb
    
    # === XCHG ===
    ("91", "xchg"),       # xchg eax, ecx
    ("92", "xchg"),       # xchg eax, edx
    ("93", "xchg"),       # xchg eax, ebx
    ("87c1", "xchg"),     # xchg ecx, eax
    ("87d0", "xchg"),     # xchg edx, eax
    
    # === BSWAP ===
    ("0fc8", "bswap"),    # bswap eax
    ("0fc9", "bswap"),    # bswap ecx
    ("0fca", "bswap"),    # bswap edx
    ("0fcb", "bswap"),    # bswap ebx
    ("480fc8", "bswap"),  # bswap rax
    
    # === BSF/BSR ===
    ("0fbcc0", "bsf"),    # bsf eax, eax
    ("0fbcc1", "bsf"),    # bsf eax, ecx
    ("0fbdc0", "bsr"),    # bsr eax, eax
    ("0fbdc1", "bsr"),    # bsr eax, ecx
    
    # === CPUID/RDTSC ===
    ("0fa2", "cpuid"),
    ("0f31", "rdtsc"),
    
    # === RET variants ===
    ("c3", "ret"),
    ("c20000", "ret"),    # ret 0
    ("c20800", "ret"),    # ret 8
    ("cb", "retf"),       # far return
    
    # === PUSH immediate ===
    ("6a00", "push"),     # push 0
    ("6a01", "push"),     # push 1
    ("6a68", "push"),     # push 0x68
    ("6aff", "push"),     # push -1
    ("6800000000", "push"), # push dword 0
    
    # === INC/DEC ===
    ("ffc0", "inc"),      # inc eax
    ("ffc1", "inc"),      # inc ecx
    ("48ffc0", "inc"),    # inc rax
    ("ffc8", "dec"),      # dec eax
    ("ffc9", "dec"),      # dec ecx
    ("48ffc8", "dec"),    # dec rax
    
    # === MUL/DIV ===
    ("f7e0", "mul"),      # mul eax
    ("f7e3", "mul"),      # mul ebx
    ("f7f0", "div"),      # div eax
    ("f7f3", "div"),      # div ebx
    ("f7e8", "imul"),     # imul eax
    ("f7f8", "idiv"),     # idiv eax
    
    # === SETCC ===
    ("0f94c0", "sete"),   # sete al
    ("0f95c0", "setne"),  # setne al
    ("0f9cc0", "setl"),   # setl al
    ("0f9fc0", "setg"),   # setg al
    
    # === CMOV ===
    ("0f44c1", "cmove"),  # cmove eax, ecx
    ("0f45c1", "cmovne"), # cmovne eax, ecx
    
    # === MOVZX/MOVSX ===
    ("0fb6c0", "movzx"),  # movzx eax, al
    ("0fb7c0", "movzx"),  # movzx eax, ax
    ("0fbec0", "movsx"),  # movsx eax, al
    ("0fbfc0", "movsx"),  # movsx eax, ax
    ("4863c0", "movsxd"), # movsxd rax, eax
]

# Adversarial patterns (must refuse) - reduced count
ADVERSARIAL_PATTERNS = [
    # Invalid hex (handled at input level, but include some)
    ("", "Empty input"),
    
    # Incomplete instructions
    ("0f", "Incomplete two-byte opcode"),
    ("66", "Lone operand size prefix"),
    ("67", "Lone address size prefix"),
    ("f0", "Lone LOCK prefix"),
    ("f2", "Lone REPNE prefix"),
    ("f3", "Lone REP prefix"),
    ("48", "Lone REX.W prefix"),
    ("4c", "Lone REX.WR prefix"),
    ("660f", "Incomplete with prefix"),
    
    # Random garbage (high entropy)
    ("a5f0df6dfa57750e", "Random bytes 1"),
    ("746a5f5b245391a9", "Random bytes 2"),
    ("b4ee0d06468f038e", "Random bytes 3"),
]


def create_dataset():
    """Create balanced training dataset."""
    random.seed(42)
    
    samples = []
    
    # Calculate how many copies of each pattern
    # Target: ~100 samples per unique mnemonic, ~8000 total valid samples
    mnemonic_counts = Counter(m for _, m in INSTRUCTION_PATTERNS)
    print(f"Unique mnemonics: {len(mnemonic_counts)}")
    print(f"Unique patterns: {len(INSTRUCTION_PATTERNS)}")
    
    # Add each pattern multiple times based on mnemonic frequency
    # Less common mnemonics get more copies
    target_per_mnemonic = 150
    
    for hex_bytes, mnemonic in INSTRUCTION_PATTERNS:
        # How many patterns share this mnemonic?
        patterns_with_mnemonic = mnemonic_counts[mnemonic]
        # Copies needed to reach target
        copies = max(10, target_per_mnemonic // patterns_with_mnemonic)
        
        for _ in range(copies):
            # Add with slight variations (case)
            variant = hex_bytes if random.random() > 0.2 else hex_bytes.upper()
            samples.append({
                "raw_bytes": variant.lower(),  # normalize to lowercase
                "expected_mnemonic": mnemonic,
                "source": "curated",
                "is_valid": True,
            })
    
    print(f"Valid samples: {len(samples)}")
    
    # Add adversarial samples (reduced - only 500)
    adversarial_copies = 500 // len(ADVERSARIAL_PATTERNS)
    for hex_bytes, desc in ADVERSARIAL_PATTERNS:
        for _ in range(adversarial_copies):
            if hex_bytes:  # Skip empty
                samples.append({
                    "raw_bytes": hex_bytes.lower(),
                    "expected_mnemonic": "unknown",
                    "source": "adversarial",
                    "is_valid": False,
                    "description": desc,
                })
    
    # Add some random adversarial
    for _ in range(100):
        rand_bytes = ''.join(random.choice('0123456789abcdef') for _ in range(random.randint(16, 32)))
        samples.append({
            "raw_bytes": rand_bytes,
            "expected_mnemonic": "unknown",
            "source": "adversarial_random",
            "is_valid": False,
        })
    
    print(f"Total samples: {len(samples)}")
    
    # Shuffle
    random.shuffle(samples)
    
    # Verify distribution
    final_mnemonics = Counter(s["expected_mnemonic"] for s in samples)
    print("\n=== Final Distribution ===")
    for m, c in final_mnemonics.most_common(20):
        print(f"  {m}: {c}")
    
    return samples


def main():
    output_path = Path("genesis_datasets/level0/train.jsonl")
    
    # Backup old dataset
    if output_path.exists():
        backup_path = output_path.with_suffix(".jsonl.bak")
        output_path.rename(backup_path)
        print(f"Backed up old dataset to {backup_path}")
    
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
