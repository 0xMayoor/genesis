# Level 1: Assembly Semantics

## Overview

Level 1 builds on Level 0's instruction recognition to understand what instructions **do**. It models the effects of instructions on registers, memory, and flags.

**Level 0 answers**: "What instructions are encoded in these bytes?"
**Level 1 answers**: "What are the effects of this instruction?"

---

## Scope

### In Scope
- Register read/write effects
- Memory read/write effects
- Flag effects (zero, carry, overflow, sign)
- Stack operations (push/pop semantics)
- Basic control flow (jumps, calls, returns)
- Instruction dependencies

### Out of Scope
- Full program analysis (Level 2+)
- High-level semantics/intent (Level 3)
- Cross-function analysis (Level 2)
- Symbolic execution

---

## Input/Output Specification

### Input Types

```python
@dataclass
class Level1Input:
    """Input from Level 0."""
    instruction: Instruction      # From Level 0
    architecture: str             # x86_64, arm64
    context: RegisterState | None # Optional: known register values
    
@dataclass
class RegisterState:
    """Known register values (for symbolic/concrete execution)."""
    registers: dict[str, int | None]  # None = unknown
    flags: dict[str, bool | None]
```

### Output Types

```python
@dataclass
class RegisterEffect:
    """Effect on a single register."""
    register: str           # e.g., "rax", "eflags"
    operation: str          # "write", "read", "read_write"
    value_expr: str | None  # e.g., "rax + rbx", "0", "[rsp]"

@dataclass
class MemoryEffect:
    """Effect on memory."""
    operation: str          # "read", "write"
    address_expr: str       # e.g., "rsp", "rbp - 8", "rax + rcx*4"
    size: int               # bytes: 1, 2, 4, 8

@dataclass
class FlagEffect:
    """Effect on CPU flags."""
    flag: str               # "ZF", "CF", "SF", "OF"
    operation: str          # "set", "clear", "modified", "unchanged"
    condition: str | None   # e.g., "result == 0" for ZF

@dataclass
class Level1Output:
    """Semantic analysis of an instruction."""
    instruction: Instruction
    register_effects: list[RegisterEffect]
    memory_effects: list[MemoryEffect]
    flag_effects: list[FlagEffect]
    control_flow: ControlFlowEffect | None
    confidence: float
    is_uncertain: bool

@dataclass
class ControlFlowEffect:
    """Control flow changes."""
    type: str               # "sequential", "jump", "call", "return", "conditional"
    target_expr: str | None # e.g., "rip + 0x10", "rax", "[rsp]"
    condition: str | None   # e.g., "ZF == 1" for je
```

---

## Instruction Semantics Examples

### Data Movement

| Instruction | Register Effects | Memory Effects | Flags |
|-------------|-----------------|----------------|-------|
| `mov rax, rbx` | rax ← rbx | none | unchanged |
| `mov rax, [rbp-8]` | rax ← read | read [rbp-8] | unchanged |
| `mov [rsp], rax` | none | write [rsp] | unchanged |
| `xchg rax, rbx` | rax ← rbx, rbx ← rax | none | unchanged |

### Arithmetic

| Instruction | Register Effects | Flags |
|-------------|-----------------|-------|
| `add rax, rbx` | rax ← rax + rbx | ZF, SF, CF, OF modified |
| `sub rax, 1` | rax ← rax - 1 | ZF, SF, CF, OF modified |
| `inc rax` | rax ← rax + 1 | ZF, SF, OF modified (not CF!) |
| `xor rax, rax` | rax ← 0 | ZF=1, SF=0, CF=0, OF=0 |

### Stack Operations

| Instruction | Register Effects | Memory Effects |
|-------------|-----------------|----------------|
| `push rax` | rsp ← rsp - 8 | write [rsp] |
| `pop rax` | rax ← read, rsp ← rsp + 8 | read [rsp] |
| `call addr` | rsp ← rsp - 8 | write [rsp] (return addr) |
| `ret` | rsp ← rsp + 8 | read [rsp] |

### Control Flow

| Instruction | Control Flow | Condition |
|-------------|--------------|-----------|
| `jmp addr` | jump to addr | always |
| `je addr` | jump to addr | ZF == 1 |
| `jne addr` | jump to addr | ZF == 0 |
| `jl addr` | jump to addr | SF != OF |
| `call addr` | call addr | always |
| `ret` | return | always |

---

## Success Criteria

### Accuracy Targets
- Register effect identification: 98%+
- Memory effect identification: 98%+
- Flag effect identification: 95%+
- Control flow identification: 99%+

### Adversarial Requirements
- Must refuse on:
  - Unknown/invalid instructions
  - Architecture-specific edge cases
  - Instructions with undefined behavior

---

## Training Data

### Structure

```json
{
  "instruction": {
    "mnemonic": "add",
    "operands": ["rax", "rbx"],
    "bytes": "4801d8"
  },
  "effects": {
    "registers": [
      {"register": "rax", "operation": "write", "value_expr": "rax + rbx"},
      {"register": "rax", "operation": "read"},
      {"register": "rbx", "operation": "read"}
    ],
    "memory": [],
    "flags": [
      {"flag": "ZF", "operation": "modified", "condition": "result == 0"},
      {"flag": "SF", "operation": "modified", "condition": "result < 0"},
      {"flag": "CF", "operation": "modified", "condition": "unsigned_overflow"},
      {"flag": "OF", "operation": "modified", "condition": "signed_overflow"}
    ],
    "control_flow": {"type": "sequential"}
  }
}
```

### Generation Strategy

1. **Manual annotation** for core instruction set (~200 instructions)
2. **Template expansion** for register/operand variants
3. **Intel manual extraction** for precise flag behavior
4. **Verification against emulators** (unicorn, qemu)

### Categories to Cover

1. **Data Transfer** (~50 patterns)
   - mov, movzx, movsx, lea, xchg, push, pop
   
2. **Arithmetic** (~40 patterns)
   - add, sub, mul, div, inc, dec, neg
   
3. **Logic** (~30 patterns)
   - and, or, xor, not, shl, shr, sar, rol, ror
   
4. **Comparison** (~20 patterns)
   - cmp, test
   
5. **Control Flow** (~40 patterns)
   - jmp, jcc (all conditions), call, ret, loop
   
6. **String Operations** (~20 patterns)
   - movs, stos, lods, scas, cmps with rep

---

## Verification

### Deterministic Checks

1. **Emulator validation**: Execute instruction in Unicorn, compare effects
2. **Intel manual consistency**: Flag behavior matches documentation
3. **Round-trip test**: Apply effects to state → compare with emulated state

### Tools
- `unicorn` — CPU emulator for verification
- Intel/AMD manuals — Ground truth for semantics
- Custom validators

---

## Test Cases

### Unit Tests

```python
def test_mov_reg_reg():
    """Test MOV register to register semantics."""
    instr = Instruction(mnemonic="mov", operands=["rax", "rbx"])
    result = level1.analyze(Level1Input(instruction=instr, architecture="x86_64"))
    
    assert len(result.register_effects) == 2
    assert any(e.register == "rax" and e.operation == "write" for e in result.register_effects)
    assert any(e.register == "rbx" and e.operation == "read" for e in result.register_effects)
    assert result.flag_effects == []  # MOV doesn't affect flags

def test_add_sets_flags():
    """Test ADD instruction flag effects."""
    instr = Instruction(mnemonic="add", operands=["rax", "rbx"])
    result = level1.analyze(Level1Input(instruction=instr, architecture="x86_64"))
    
    flag_names = {e.flag for e in result.flag_effects}
    assert "ZF" in flag_names
    assert "CF" in flag_names
    assert "SF" in flag_names
    assert "OF" in flag_names

def test_push_stack_effect():
    """Test PUSH stack semantics."""
    instr = Instruction(mnemonic="push", operands=["rax"])
    result = level1.analyze(Level1Input(instruction=instr, architecture="x86_64"))
    
    # RSP decremented
    rsp_effect = next(e for e in result.register_effects if e.register == "rsp")
    assert rsp_effect.operation == "write"
    assert "- 8" in rsp_effect.value_expr
    
    # Memory write
    assert len(result.memory_effects) == 1
    assert result.memory_effects[0].operation == "write"
```

### Integration Tests

```python
def test_level1_uses_level0_output():
    """Test that Level 1 accepts Level 0 output."""
    l0_output = level0.process(Level0Input(data=b"\x48\x01\xd8"))  # add rax, rbx
    l1_input = Level1Input.from_level0(l0_output.instructions[0])
    l1_output = level1.analyze(l1_input)
    
    assert not l1_output.is_uncertain
    assert l1_output.register_effects != []
```

### Adversarial Tests

```python
def test_invalid_instruction_refused():
    """Invalid instructions must be refused."""
    instr = Instruction(mnemonic="invalid_op", operands=[])
    result = level1.analyze(Level1Input(instruction=instr, architecture="x86_64"))
    assert result.is_uncertain

def test_unknown_register_refused():
    """Unknown registers must be refused."""
    instr = Instruction(mnemonic="mov", operands=["xyz", "abc"])
    result = level1.analyze(Level1Input(instruction=instr, architecture="x86_64"))
    assert result.is_uncertain
```

---

## Dependencies

### On Lower Levels
- **Level 0**: Provides parsed instructions

### On Higher Levels
- **Level 2**: Uses instruction effects for data flow analysis

---

## Milestones

| Milestone | Description | Target |
|-----------|-------------|--------|
| M1.1 | Interface defined | Week 1 |
| M1.2 | Core instruction semantics (~50) | Week 2 |
| M1.3 | Full x86_64 coverage (~200) | Week 3 |
| M1.4 | Test suite complete | Week 4 |
| M1.5 | Model trained, 95%+ accuracy | Week 5 |
| M1.6 | Integration with Level 0 and Level 2 | Week 6 |

---

## Open Questions

1. How detailed should value expressions be? (e.g., "rax + rbx" vs symbolic)
2. Should we track implicit register usage? (e.g., mul uses rdx:rax)
3. How to handle segment registers and memory models?
4. Should flag effects include precise conditions or just "modified"?

---

## References

- Intel® 64 and IA-32 Architectures Software Developer's Manual
- AMD64 Architecture Programmer's Manual
- Unicorn Engine documentation
- "Intel x86 Considered Harmful" (paper on x86 complexity)
