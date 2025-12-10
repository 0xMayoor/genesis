"""Level 2: Intermediate Representations.

This module provides control flow analysis:
- Basic block detection
- Control flow graph construction
- Function boundary detection
- Loop detection
"""

from levels.level2_ir.types import (
    # Enums
    EntryType,
    ExitType,
    EdgeType,
    LoopType,
    # Data structures
    BasicBlock,
    CFGEdge,
    Function,
    CallEdge,
    Loop,
    # Input/Output
    Level2Input,
    Level2Output,
)

from levels.level2_ir.module import Level2Module

__all__ = [
    # Enums
    "EntryType",
    "ExitType", 
    "EdgeType",
    "LoopType",
    # Data structures
    "BasicBlock",
    "CFGEdge",
    "Function",
    "CallEdge",
    "Loop",
    # Input/Output
    "Level2Input",
    "Level2Output",
    # Module
    "Level2Module",
]
