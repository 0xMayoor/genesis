"""Level 0: Machine Code Patterns.

The foundation of GENESIS. This level handles raw binary/machine code
and learns to recognize instruction patterns, boundaries, and basic structures.

This level answers: "What instructions are encoded in these bytes?"

Scope:
- Opcode identification
- Instruction boundary detection
- Basic instruction classification
- Byte pattern recognition
- Architecture identification (x86, ARM)

Out of scope:
- Control flow analysis (Level 1)
- Semantic meaning (Level 3+)
- High-level code generation (Level 4)
"""

from levels.level0_machine.module import Level0Module
from levels.level0_machine.types import (
    Architecture,
    Instruction,
    InstructionCategory,
    Level0Input,
    Level0Output,
)

__all__ = [
    "Level0Input",
    "Level0Output",
    "Instruction",
    "Architecture",
    "InstructionCategory",
    "Level0Module",
]
