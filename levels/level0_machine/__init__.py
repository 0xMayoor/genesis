"""Level 0: Machine Code Patterns.

The foundation of GENESIS. This level handles raw binary/machine code
and learns to recognize instruction patterns, boundaries, and basic structures.

This level answers: "What instructions are encoded in these bytes?"

Status: COMPLETE ✓ (100% accuracy on gate test)

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

from levels.level0_machine.classifier import Level0Classifier, predict_mnemonic
from levels.level0_machine.module import Level0Module
from levels.level0_machine.types import (
    Architecture,
    Instruction,
    InstructionCategory,
    Level0Input,
    Level0Output,
)

__all__ = [
    # Main classifier
    "Level0Classifier",
    "predict_mnemonic",
    # Module interface
    "Level0Module",
    # Types
    "Level0Input",
    "Level0Output",
    "Instruction",
    "Architecture",
    "InstructionCategory",
]
