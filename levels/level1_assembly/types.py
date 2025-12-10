"""Type definitions for Level 1: Assembly Semantics."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from levels.level0_machine.types import Instruction


class EffectOperation(Enum):
    """Types of register/memory operations."""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class FlagOperation(Enum):
    """Types of flag operations."""
    SET = "set"           # Always set to 1
    CLEAR = "clear"       # Always set to 0
    MODIFIED = "modified" # Changed based on result
    UNCHANGED = "unchanged"


class ControlFlowType(Enum):
    """Types of control flow."""
    SEQUENTIAL = "sequential"   # Normal execution
    JUMP = "jump"               # Unconditional jump
    CONDITIONAL = "conditional" # Conditional jump
    CALL = "call"               # Function call
    RETURN = "return"           # Function return
    INTERRUPT = "interrupt"     # System call/interrupt


@dataclass
class RegisterState:
    """Known register values for context."""
    registers: dict[str, int | None] = field(default_factory=dict)
    flags: dict[str, bool | None] = field(default_factory=dict)
    
    def get_register(self, name: str) -> int | None:
        """Get register value, None if unknown."""
        return self.registers.get(name.lower())
    
    def get_flag(self, name: str) -> bool | None:
        """Get flag value, None if unknown."""
        return self.flags.get(name.upper())


@dataclass
class Level1Input:
    """Input to Level 1 module."""
    instruction: Instruction
    architecture: str = "x86_64"
    context: RegisterState | None = None
    
    @classmethod
    def from_level0(cls, instruction: Instruction, architecture: str = "x86_64") -> "Level1Input":
        """Create from Level 0 output."""
        return cls(instruction=instruction, architecture=architecture)


@dataclass
class RegisterEffect:
    """Effect on a single register."""
    register: str
    operation: EffectOperation
    value_expr: str | None = None  # Expression like "rax + rbx"
    
    def __repr__(self) -> str:
        if self.value_expr:
            return f"{self.register} <- {self.value_expr}"
        return f"{self.register} ({self.operation.value})"


@dataclass
class MemoryEffect:
    """Effect on memory."""
    operation: EffectOperation
    address_expr: str
    size: int  # bytes
    
    def __repr__(self) -> str:
        size_name = {1: "byte", 2: "word", 4: "dword", 8: "qword"}.get(self.size, f"{self.size}b")
        return f"{self.operation.value} {size_name} [{self.address_expr}]"


@dataclass
class FlagEffect:
    """Effect on CPU flags."""
    flag: str  # ZF, CF, SF, OF, etc.
    operation: FlagOperation
    condition: str | None = None  # When flag is set/cleared
    
    def __repr__(self) -> str:
        if self.condition:
            return f"{self.flag}: {self.operation.value} ({self.condition})"
        return f"{self.flag}: {self.operation.value}"


@dataclass
class ControlFlowEffect:
    """Effect on control flow."""
    type: ControlFlowType
    target_expr: str | None = None  # Target address expression
    condition: str | None = None    # Condition for conditional jumps
    
    def __repr__(self) -> str:
        if self.type == ControlFlowType.SEQUENTIAL:
            return "sequential"
        if self.condition:
            return f"{self.type.value} to {self.target_expr} if {self.condition}"
        return f"{self.type.value} to {self.target_expr}"


@dataclass
class Level1Output:
    """Output from Level 1 analysis."""
    instruction: Instruction
    register_effects: list[RegisterEffect] = field(default_factory=list)
    memory_effects: list[MemoryEffect] = field(default_factory=list)
    flag_effects: list[FlagEffect] = field(default_factory=list)
    control_flow: ControlFlowEffect = field(
        default_factory=lambda: ControlFlowEffect(type=ControlFlowType.SEQUENTIAL)
    )
    confidence: float = 1.0
    is_uncertain: bool = False
    uncertainty_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instruction": {
                "mnemonic": self.instruction.mnemonic,
                "operands": list(self.instruction.operands),
            },
            "register_effects": [
                {"register": e.register, "operation": e.operation.value, "value_expr": e.value_expr}
                for e in self.register_effects
            ],
            "memory_effects": [
                {"operation": e.operation.value, "address_expr": e.address_expr, "size": e.size}
                for e in self.memory_effects
            ],
            "flag_effects": [
                {"flag": e.flag, "operation": e.operation.value, "condition": e.condition}
                for e in self.flag_effects
            ],
            "control_flow": {
                "type": self.control_flow.type.value,
                "target_expr": self.control_flow.target_expr,
                "condition": self.control_flow.condition,
            },
            "confidence": self.confidence,
            "is_uncertain": self.is_uncertain,
        }
    
    @property
    def reads_registers(self) -> list[str]:
        """Get list of registers read."""
        return [e.register for e in self.register_effects 
                if e.operation in (EffectOperation.READ, EffectOperation.READ_WRITE)]
    
    @property
    def writes_registers(self) -> list[str]:
        """Get list of registers written."""
        return [e.register for e in self.register_effects 
                if e.operation in (EffectOperation.WRITE, EffectOperation.READ_WRITE)]
    
    @property
    def modifies_flags(self) -> list[str]:
        """Get list of flags modified."""
        return [e.flag for e in self.flag_effects 
                if e.operation != FlagOperation.UNCHANGED]


# Common x86_64 registers for validation
X86_64_REGISTERS = {
    # 64-bit
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip",
    # 32-bit
    "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
    # 16-bit
    "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
    # 8-bit
    "al", "ah", "bl", "bh", "cl", "ch", "dl", "dh",
    "sil", "dil", "bpl", "spl",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
}

X86_64_FLAGS = {"CF", "PF", "AF", "ZF", "SF", "TF", "IF", "DF", "OF"}
