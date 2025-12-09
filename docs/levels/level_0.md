# Level 0: Machine Code Patterns

## Overview

Level 0 is the foundation of GENESIS. It handles raw binary/machine code and learns to recognize instruction patterns, boundaries, and basic structures.

**This level answers**: "What instructions are encoded in these bytes?"

---

## Scope

### In Scope
- Opcode identification
- Instruction boundary detection
- Basic instruction classification
- Byte pattern recognition
- Architecture identification (x86, ARM, etc.)

### Out of Scope
- Control flow analysis (Level 1)
- Semantic meaning (Level 3+)
- High-level code generation (Level 4)

---

## Input/Output Specification

### Input Types

```python
@dataclass
class Level0Input:
    bytes: bytes                    # Raw binary data
    architecture: str | None        # "x86_64", "arm64", None for auto-detect
    offset: int = 0                 # Starting offset
    context: dict | None = None     # Additional context
```

### Output Types

```python
@dataclass
class Instruction:
    offset: int                     # Byte offset
    bytes: bytes                    # Raw instruction bytes
    mnemonic: str                   # e.g., "mov", "jmp"
    operands: list[str]             # e.g., ["eax", "ebx"]
    size: int                       # Instruction size in bytes
    category: str                   # "data_transfer", "control_flow", etc.

@dataclass
class Level0Output:
    instructions: list[Instruction]
    architecture: str               # Detected or confirmed
    confidence: float
    is_uncertain: bool
    uncertainty_reason: str | None
    metadata: dict
```

---

## Categories

### Instruction Categories
1. **Data Transfer** — mov, push, pop, lea
2. **Arithmetic** — add, sub, mul, div, inc, dec
3. **Logic** — and, or, xor, not, shl, shr
4. **Control Flow** — jmp, call, ret, je, jne
5. **Comparison** — cmp, test
6. **Stack** — push, pop, enter, leave
7. **System** — syscall, int, nop
8. **Unknown** — Unrecognized patterns

---

## Success Criteria

### Accuracy Targets
- Instruction boundary detection: 99%+
- Opcode identification: 98%+
- Architecture detection: 99%+

### Adversarial Requirements
- Must refuse on:
  - Random/garbage bytes (no valid instructions)
  - Incomplete instruction sequences
  - Mixed architecture data
  - Encrypted/obfuscated sections

---

## Training Data

### Sources
- Compiled binaries from known source code
- Disassembler output (objdump, Ghidra)
- Synthetic instruction sequences

### Generation Strategy
```python
# Pseudocode for data generation
1. Take known C/assembly source
2. Compile with known compiler/flags
3. Extract binary sections
4. Disassemble with trusted tool
5. Create (bytes, instructions) pairs
6. Verify round-trip consistency
```

### Adversarial Samples
- Random byte sequences
- Truncated instructions
- Invalid opcode combinations
- Data sections (not code)

---

## Model Specification

### Architecture
- Small transformer encoder
- ~50M parameters
- Byte-level tokenization

### Training
- LoRA fine-tuning on pretrained base
- Batch size: 32-64
- Learning rate: 1e-4 to 5e-5

### Inference
- Input: Up to 512 bytes
- Output: Structured instruction list
- Confidence threshold: 0.85

---

## Verification

### Deterministic Checks
1. **Round-trip test**: Disassemble → Reassemble → Compare bytes
2. **Boundary validation**: Instructions don't overlap
3. **Size consistency**: Instruction sizes sum to input size
4. **Known pattern matching**: Compare against disassembler output

### Tools
- `objdump` — Reference disassembler
- `capstone` — Disassembly library
- Custom validators

---

## Test Cases

### Unit Tests
```python
def test_parse_mov_eax_ebx():
    """Test parsing simple MOV instruction."""
    input_bytes = b'\x89\xd8'  # mov eax, ebx
    result = level0.process(Level0Input(bytes=input_bytes, architecture="x86_64"))
    assert result.instructions[0].mnemonic == "mov"
    assert result.instructions[0].operands == ["eax", "ebx"]

def test_parse_invalid_bytes_returns_uncertain():
    """Test that random bytes are refused."""
    input_bytes = bytes([random.randint(0, 255) for _ in range(100)])
    result = level0.process(Level0Input(bytes=input_bytes))
    assert result.is_uncertain == True
```

### Integration Tests
```python
def test_level0_feeds_level1():
    """Test that Level 0 output is valid Level 1 input."""
    binary = load_test_binary("simple_function")
    l0_output = level0.process(Level0Input(bytes=binary))
    l1_input = Level1Input.from_level0(l0_output)
    assert l1_input.is_valid()
```

### Adversarial Tests
```python
def test_random_bytes_refused():
    """Random bytes must be refused."""
    ...

def test_truncated_instruction_refused():
    """Incomplete instructions must be refused."""
    ...

def test_data_section_refused():
    """Non-code data must be identified as such."""
    ...
```

---

## Dependencies

### On Lower Levels
None — This is the foundation.

### On Higher Levels
Level 1 depends on Level 0 output.

### External Tools
- `capstone` — Disassembly engine
- `keystone` — Assembly engine (for verification)

---

## Milestones

| Milestone | Description | Target |
|-----------|-------------|--------|
| M0.1 | Interface defined | Week 1 |
| M0.2 | Test suite complete | Week 2 |
| M0.3 | Dataset generated | Week 3 |
| M0.4 | Model trained | Week 4 |
| M0.5 | 98%+ accuracy achieved | Week 5 |
| M0.6 | Integration with Level 1 | Week 6 |

---

## Open Questions

1. How to handle self-modifying code?
2. Should we support multiple architectures from start?
3. How to handle variable-length instruction sets?

---

## References

- Intel x86 Manual
- ARM Architecture Reference
- Capstone documentation
- "Practical Binary Analysis" (book)
