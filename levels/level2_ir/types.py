"""Level 2 Types: Intermediate Representations.

Defines types for control flow analysis:
- Basic blocks
- Control flow graphs
- Functions
- Loops
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from levels.level1_assembly.types import Level1Output


class EntryType(Enum):
    """How execution enters a basic block."""
    FUNCTION_START = "function_start"
    JUMP_TARGET = "jump_target"
    FALL_THROUGH = "fall_through"
    EXCEPTION_HANDLER = "exception_handler"
    UNKNOWN = "unknown"


class ExitType(Enum):
    """How execution exits a basic block."""
    FALL_THROUGH = "fall_through"
    UNCONDITIONAL_JUMP = "unconditional_jump"
    CONDITIONAL_JUMP = "conditional_jump"
    CALL = "call"
    RETURN = "return"
    HALT = "halt"
    UNKNOWN = "unknown"


class EdgeType(Enum):
    """Type of control flow edge."""
    UNCONDITIONAL = "unconditional"
    CONDITIONAL_TRUE = "conditional_true"
    CONDITIONAL_FALSE = "conditional_false"
    CALL = "call"
    RETURN = "return"
    FALL_THROUGH = "fall_through"
    EXCEPTION = "exception"


class LoopType(Enum):
    """Type of loop structure."""
    PRE_TESTED = "pre_tested"   # while loop
    POST_TESTED = "post_tested"  # do-while loop
    COUNTED = "counted"          # for loop with counter
    INFINITE = "infinite"        # infinite loop
    UNKNOWN = "unknown"


@dataclass
class BasicBlock:
    """A sequence of instructions with single entry/exit.
    
    Properties:
    - Execution enters only at first instruction
    - Execution exits only at last instruction
    - No internal jumps
    """
    id: int
    start_offset: int
    end_offset: int
    instructions: list[Level1Output] = field(default_factory=list)
    entry_type: EntryType = EntryType.UNKNOWN
    exit_type: ExitType = ExitType.UNKNOWN
    
    @property
    def size(self) -> int:
        """Number of instructions in block."""
        return len(self.instructions)
    
    @property
    def is_empty(self) -> bool:
        """Check if block has no instructions."""
        return len(self.instructions) == 0
    
    def __repr__(self) -> str:
        return f"BB{self.id}[{self.start_offset:#x}-{self.end_offset:#x}]({self.size} instrs)"


@dataclass
class CFGEdge:
    """Edge in control flow graph."""
    source_block: int
    target_block: int
    edge_type: EdgeType
    condition: Optional[str] = None  # For conditional edges (e.g., "ZF==1")
    
    def __repr__(self) -> str:
        cond = f" ({self.condition})" if self.condition else ""
        return f"BB{self.source_block} -> BB{self.target_block} [{self.edge_type.value}]{cond}"


@dataclass
class Function:
    """A callable unit consisting of basic blocks."""
    entry_offset: int
    entry_block: int
    exit_blocks: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    name: Optional[str] = None
    
    # Call relationships
    calls: list[int] = field(default_factory=list)     # Offsets of called functions
    callers: list[int] = field(default_factory=list)   # Offsets of callers
    
    @property
    def is_leaf(self) -> bool:
        """Check if function makes no calls."""
        return len(self.calls) == 0
    
    @property
    def block_count(self) -> int:
        """Number of basic blocks in function."""
        return len(self.blocks)
    
    def __repr__(self) -> str:
        name = self.name or f"func_{self.entry_offset:#x}"
        return f"{name}({self.block_count} blocks)"


@dataclass
class CallEdge:
    """Edge in call graph."""
    caller_offset: int
    callee_offset: int
    call_site: int  # Offset of call instruction
    is_direct: bool = True  # False for indirect calls
    
    def __repr__(self) -> str:
        call_type = "direct" if self.is_direct else "indirect"
        return f"{self.caller_offset:#x} -> {self.callee_offset:#x} ({call_type})"


@dataclass
class Loop:
    """A cycle in the control flow graph."""
    header_block: int        # Loop entry point (dominates all loop blocks)
    back_edge_block: int     # Block that jumps back to header
    body_blocks: list[int] = field(default_factory=list)
    loop_type: LoopType = LoopType.UNKNOWN
    nesting_depth: int = 0   # 0 = outermost
    parent_loop: Optional[int] = None  # Header of containing loop
    
    @property
    def block_count(self) -> int:
        """Number of blocks in loop body."""
        return len(self.body_blocks)
    
    def __repr__(self) -> str:
        return f"Loop(header=BB{self.header_block}, {self.block_count} blocks, {self.loop_type.value})"


@dataclass
class Level2Input:
    """Input for Level 2 analysis."""
    instructions: list[Level1Output]
    entry_point: int = 0
    architecture: str = "x86_64"
    
    @property
    def instruction_count(self) -> int:
        return len(self.instructions)


@dataclass
class Level2Output:
    """Output from Level 2 analysis."""
    basic_blocks: list[BasicBlock] = field(default_factory=list)
    cfg_edges: list[CFGEdge] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    call_edges: list[CallEdge] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    
    # Uncertainty tracking
    is_uncertain: bool = False
    uncertainty_reason: Optional[str] = None
    
    @property
    def block_count(self) -> int:
        return len(self.basic_blocks)
    
    @property
    def function_count(self) -> int:
        return len(self.functions)
    
    @property
    def loop_count(self) -> int:
        return len(self.loops)
    
    def get_block(self, block_id: int) -> Optional[BasicBlock]:
        """Get basic block by ID."""
        for block in self.basic_blocks:
            if block.id == block_id:
                return block
        return None
    
    def get_block_at_offset(self, offset: int) -> Optional[BasicBlock]:
        """Get basic block containing offset."""
        for block in self.basic_blocks:
            if block.start_offset <= offset <= block.end_offset:
                return block
        return None
    
    def get_successors(self, block_id: int) -> list[int]:
        """Get successor block IDs."""
        return [e.target_block for e in self.cfg_edges if e.source_block == block_id]
    
    def get_predecessors(self, block_id: int) -> list[int]:
        """Get predecessor block IDs."""
        return [e.source_block for e in self.cfg_edges if e.target_block == block_id]
    
    def __repr__(self) -> str:
        return (f"Level2Output({self.block_count} blocks, "
                f"{len(self.cfg_edges)} edges, "
                f"{self.function_count} functions, "
                f"{self.loop_count} loops)")
