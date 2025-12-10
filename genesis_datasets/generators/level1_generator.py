"""Level 1 Dataset Generator.

Generates training data for assembly semantics understanding.
Each sample consists of:
- Input: Instruction (mnemonic + operands)
- Output: Semantic effects (registers, memory, flags, control flow)

Uses the deterministic Level1Module as ground truth.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly import Level1Module, Level1Input


@dataclass
class Level1Sample:
    """A single training sample for Level 1."""
    
    # Input
    mnemonic: str
    operands: list[str]
    
    # Expected output (from deterministic module)
    reads_registers: list[str]
    writes_registers: list[str]
    memory_reads: list[dict]
    memory_writes: list[dict]
    flag_effects: list[dict]
    control_flow_type: str
    control_flow_condition: str | None
    
    # Metadata
    is_valid: bool
    category: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mnemonic": self.mnemonic,
            "operands": self.operands,
            "reads_registers": self.reads_registers,
            "writes_registers": self.writes_registers,
            "memory_reads": self.memory_reads,
            "memory_writes": self.memory_writes,
            "flag_effects": self.flag_effects,
            "control_flow_type": self.control_flow_type,
            "control_flow_condition": self.control_flow_condition,
            "is_valid": self.is_valid,
            "category": self.category,
        }


# Register sets for generating variations
REGS_64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
REGS_32 = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"]
REGS_16 = ["ax", "bx", "cx", "dx", "si", "di", "bp", "sp"]
REGS_8 = ["al", "bl", "cl", "dl", "ah", "bh", "ch", "dh", "sil", "dil"]

# Immediate values
IMM_VALUES = ["0", "1", "2", "4", "8", "16", "32", "64", "0xff", "0x100", "0x1000", "-1", "-8"]

# Memory operands
MEM_OPS = ["[rax]", "[rbx]", "[rcx]", "[rdx]", "[rsp]", "[rbp]", 
           "[rax + 8]", "[rbx + 16]", "[rsp + 8]", "[rbp - 8]", "[rbp - 16]",
           "[rax + rcx]", "[rbx + rdx*4]", "[rsp + rax*8]"]

# Jump targets
JUMP_TARGETS = ["0x1000", "0x2000", "0x4000", "0x8000", "rax", "rcx", "[rax]"]

# Instruction templates
INSTRUCTION_TEMPLATES = {
    # Data Movement
    "mov_reg_reg": ("mov", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "mov_reg_imm": ("mov", lambda: [random.choice(REGS_64), random.choice(IMM_VALUES)]),
    "mov_reg_mem": ("mov", lambda: [random.choice(REGS_64), random.choice(MEM_OPS)]),
    "mov_mem_reg": ("mov", lambda: [random.choice(MEM_OPS), random.choice(REGS_64)]),
    "movzx": ("movzx", lambda: [random.choice(REGS_32), random.choice(REGS_8)]),
    "movsx": ("movsx", lambda: [random.choice(REGS_64), random.choice(REGS_32)]),
    "lea": ("lea", lambda: [random.choice(REGS_64), random.choice(MEM_OPS)]),
    "xchg": ("xchg", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    
    # Stack
    "push_reg": ("push", lambda: [random.choice(REGS_64)]),
    "push_imm": ("push", lambda: [random.choice(IMM_VALUES)]),
    "pop_reg": ("pop", lambda: [random.choice(REGS_64)]),
    "leave": ("leave", lambda: []),
    "enter": ("enter", lambda: ["16", "0"]),
    
    # Arithmetic
    "add_reg_reg": ("add", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "add_reg_imm": ("add", lambda: [random.choice(REGS_64), random.choice(IMM_VALUES)]),
    "sub_reg_reg": ("sub", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "sub_reg_imm": ("sub", lambda: [random.choice(REGS_64), random.choice(IMM_VALUES)]),
    "inc": ("inc", lambda: [random.choice(REGS_64)]),
    "dec": ("dec", lambda: [random.choice(REGS_64)]),
    "neg": ("neg", lambda: [random.choice(REGS_64)]),
    "mul": ("mul", lambda: [random.choice(REGS_64)]),
    "imul_1": ("imul", lambda: [random.choice(REGS_64)]),
    "imul_2": ("imul", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "div": ("div", lambda: [random.choice(REGS_64)]),
    "idiv": ("idiv", lambda: [random.choice(REGS_64)]),
    
    # Logic
    "and_reg_reg": ("and", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "and_reg_imm": ("and", lambda: [random.choice(REGS_64), random.choice(IMM_VALUES)]),
    "or_reg_reg": ("or", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "or_reg_imm": ("or", lambda: [random.choice(REGS_64), random.choice(IMM_VALUES)]),
    "xor_reg_reg": ("xor", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "xor_self": ("xor", lambda: [REGS_64[0], REGS_64[0]]),  # xor rax, rax
    "not": ("not", lambda: [random.choice(REGS_64)]),
    "shl": ("shl", lambda: [random.choice(REGS_64), random.choice(["1", "2", "4", "8"])]),
    "shr": ("shr", lambda: [random.choice(REGS_64), random.choice(["1", "2", "4", "8"])]),
    "sar": ("sar", lambda: [random.choice(REGS_64), random.choice(["1", "2", "4", "8"])]),
    "rol": ("rol", lambda: [random.choice(REGS_64), "1"]),
    "ror": ("ror", lambda: [random.choice(REGS_64), "1"]),
    
    # Comparison
    "cmp_reg_reg": ("cmp", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "cmp_reg_imm": ("cmp", lambda: [random.choice(REGS_64), random.choice(IMM_VALUES)]),
    "test_reg_reg": ("test", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "test_self": ("test", lambda: [REGS_64[0], REGS_64[0]]),  # test rax, rax
    
    # Control Flow
    "jmp_rel": ("jmp", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jmp_reg": ("jmp", lambda: [random.choice(REGS_64)]),
    "je": ("je", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jne": ("jne", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jl": ("jl", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jle": ("jle", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jg": ("jg", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jge": ("jge", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jb": ("jb", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jbe": ("jbe", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "ja": ("ja", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jae": ("jae", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "js": ("js", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "jns": ("jns", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "call_rel": ("call", lambda: [random.choice(JUMP_TARGETS[:4])]),
    "call_reg": ("call", lambda: [random.choice(REGS_64)]),
    "ret": ("ret", lambda: []),
    
    # System
    "nop": ("nop", lambda: []),
    "syscall": ("syscall", lambda: []),
    "int_80": ("int", lambda: ["0x80"]),
    "int3": ("int3", lambda: []),
    "hlt": ("hlt", lambda: []),
    
    # Flags
    "pushf": ("pushf", lambda: []),
    "popf": ("popf", lambda: []),
    "cld": ("cld", lambda: []),
    "std": ("std", lambda: []),
    
    # String
    "movsb": ("movsb", lambda: []),
    "movsw": ("movsw", lambda: []),
    "movsd": ("movsd", lambda: []),
    "movsq": ("movsq", lambda: []),
    "stosb": ("stosb", lambda: []),
    "stosd": ("stosd", lambda: []),
    "lodsb": ("lodsb", lambda: []),
    "lodsd": ("lodsd", lambda: []),
    "scasb": ("scasb", lambda: []),
    "rep": ("rep", lambda: []),
    
    # Bit manipulation
    "bsf": ("bsf", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "bsr": ("bsr", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "bswap": ("bswap", lambda: [random.choice(REGS_32)]),
    
    # Conditional set/move
    "sete": ("sete", lambda: [random.choice(REGS_8)]),
    "setne": ("setne", lambda: [random.choice(REGS_8)]),
    "setl": ("setl", lambda: [random.choice(REGS_8)]),
    "setg": ("setg", lambda: [random.choice(REGS_8)]),
    "cmove": ("cmove", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    "cmovne": ("cmovne", lambda: [random.choice(REGS_64), random.choice(REGS_64)]),
    
    # Misc
    "cpuid": ("cpuid", lambda: []),
    "rdtsc": ("rdtsc", lambda: []),
}

# Adversarial patterns (should be refused)
ADVERSARIAL_PATTERNS = [
    ("invalid_op", []),
    ("fakeinstr", ["rax"]),
    ("mov", ["xyz"]),  # Wrong operand count
    ("add", []),       # Missing operands
    ("mov", ["abc", "def"]),  # Invalid registers
    ("push", []),      # Missing operand
]


class Level1DatasetGenerator:
    """Generates training datasets for Level 1."""
    
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._module = Level1Module()
        random.seed(seed)
    
    def _make_instruction(self, mnemonic: str, operands: list[str]) -> Instruction:
        """Create an Instruction object."""
        return Instruction(
            offset=0,
            raw_bytes=b"\x90",
            mnemonic=mnemonic,
            operands=tuple(operands),
            size=1,
            category=InstructionCategory.DATA_TRANSFER,
        )
    
    def generate_sample(self, template_name: str) -> Level1Sample | None:
        """Generate a single sample from a template."""
        if template_name not in INSTRUCTION_TEMPLATES:
            return None
        
        mnemonic, operand_gen = INSTRUCTION_TEMPLATES[template_name]
        operands = operand_gen()
        
        # Get ground truth from module
        instr = self._make_instruction(mnemonic, operands)
        result = self._module.analyze(Level1Input(instruction=instr))
        
        if result.is_uncertain:
            return None
        
        return Level1Sample(
            mnemonic=mnemonic,
            operands=operands,
            reads_registers=result.reads_registers,
            writes_registers=result.writes_registers,
            memory_reads=[
                {"address": e.address_expr, "size": e.size}
                for e in result.memory_effects
                if e.operation.value == "read"
            ],
            memory_writes=[
                {"address": e.address_expr, "size": e.size}
                for e in result.memory_effects
                if e.operation.value == "write"
            ],
            flag_effects=[
                {"flag": e.flag, "operation": e.operation.value}
                for e in result.flag_effects
            ],
            control_flow_type=result.control_flow.type.value,
            control_flow_condition=result.control_flow.condition,
            is_valid=True,
            category=template_name.split("_")[0],
        )
    
    def generate_adversarial(self) -> Level1Sample:
        """Generate an adversarial sample."""
        mnemonic, operands = self._rng.choice(ADVERSARIAL_PATTERNS)
        
        return Level1Sample(
            mnemonic=mnemonic,
            operands=operands,
            reads_registers=[],
            writes_registers=[],
            memory_reads=[],
            memory_writes=[],
            flag_effects=[],
            control_flow_type="unknown",
            control_flow_condition=None,
            is_valid=False,
            category="adversarial",
        )
    
    def generate_dataset(
        self,
        samples_per_template: int = 100,
        adversarial_count: int = 500,
    ) -> list[Level1Sample]:
        """Generate complete dataset."""
        samples = []
        
        # Generate from each template
        for template_name in INSTRUCTION_TEMPLATES:
            for _ in range(samples_per_template):
                sample = self.generate_sample(template_name)
                if sample:
                    samples.append(sample)
        
        # Add adversarial samples
        for _ in range(adversarial_count):
            samples.append(self.generate_adversarial())
        
        # Shuffle
        self._rng.shuffle(samples)
        
        return samples
    
    def save_dataset(self, samples: list[Level1Sample], output_path: Path) -> None:
        """Save dataset to JSON Lines format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
    
    def print_stats(self, samples: list[Level1Sample]) -> None:
        """Print dataset statistics."""
        print(f"Total samples: {len(samples)}")
        
        valid = sum(1 for s in samples if s.is_valid)
        print(f"Valid: {valid} ({100*valid/len(samples):.1f}%)")
        print(f"Adversarial: {len(samples) - valid}")
        
        # Category distribution
        categories = Counter(s.category for s in samples)
        print("\nCategory distribution:")
        for cat, count in categories.most_common(15):
            print(f"  {cat}: {count}")
        
        # Mnemonic distribution
        mnemonics = Counter(s.mnemonic for s in samples if s.is_valid)
        print(f"\nUnique mnemonics: {len(mnemonics)}")


if __name__ == "__main__":
    generator = Level1DatasetGenerator(seed=42)
    
    print("Generating Level 1 dataset...")
    samples = generator.generate_dataset(
        samples_per_template=100,
        adversarial_count=500,
    )
    
    generator.print_stats(samples)
    
    # Save
    output_path = Path("genesis_datasets/level1/train.jsonl")
    generator.save_dataset(samples, output_path)
    print(f"\n✅ Saved to {output_path}")
