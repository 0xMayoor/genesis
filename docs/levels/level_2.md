# Level 2: Intermediate Representations

## Overview

Level 2 builds on Level 1's instruction semantics to understand **control flow** and **structured program representations**. While Level 1 analyzes individual instructions, Level 2 sees the bigger picture: basic blocks, control flow graphs, and function boundaries.

## Scope

### What Level 2 Handles
- **Basic Block Detection** — Sequences of instructions with single entry/exit
- **Control Flow Graphs (CFG)** — How basic blocks connect
- **Function Boundaries** — Identifying function start/end
- **Call Graphs** — Which functions call which
- **Loop Detection** — Identifying loop structures
- **Dominator Analysis** — Control flow dominance relationships

### What Level 2 Does NOT Handle
- Source code semantics (Level 3)
- Variable naming/typing (Level 3)
- High-level language constructs (Level 4)
- LLVM IR text syntax (future extension)

## Input/Output

### Input
```python
@dataclass
class Level2Input:
    """Input for Level 2 analysis."""
    instructions: list[Level1Output]  # Sequence of analyzed instructions
    entry_point: int = 0              # Starting offset
    architecture: str = "x86_64"
```

### Output
```python
@dataclass
class Level2Output:
    """Output from Level 2 analysis."""
    basic_blocks: list[BasicBlock]
    cfg_edges: list[CFGEdge]
    functions: list[Function]
    call_graph: list[CallEdge]
    loops: list[Loop]
    is_uncertain: bool = False
    uncertainty_reason: str | None = None
```

## Core Concepts

### Basic Block
A sequence of instructions where:
- Execution enters only at the first instruction
- Execution exits only at the last instruction
- No jumps into the middle, no jumps out except at the end

```python
@dataclass
class BasicBlock:
    id: int
    start_offset: int
    end_offset: int
    instructions: list[Level1Output]
    entry_type: EntryType  # FUNCTION_START, JUMP_TARGET, FALL_THROUGH
    exit_type: ExitType    # FALL_THROUGH, JUMP, CONDITIONAL, RETURN, CALL
```

### CFG Edge
Connection between basic blocks:
```python
@dataclass  
class CFGEdge:
    source_block: int      # Block ID
    target_block: int      # Block ID
    edge_type: EdgeType    # UNCONDITIONAL, CONDITIONAL_TRUE, CONDITIONAL_FALSE, CALL, RETURN
    condition: str | None  # For conditional edges
```

### Function
A collection of basic blocks forming a callable unit:
```python
@dataclass
class Function:
    entry_block: int
    exit_blocks: list[int]
    blocks: list[int]      # All block IDs in function
    calls: list[int]       # Offsets of called functions
    callers: list[int]     # Offsets of callers
```

### Loop
A cycle in the control flow graph:
```python
@dataclass
class Loop:
    header_block: int      # Loop entry point
    back_edge_block: int   # Block that jumps back to header
    body_blocks: list[int] # All blocks in loop body
    loop_type: LoopType    # FOR, WHILE, DO_WHILE, UNKNOWN
```

## Algorithm: Basic Block Detection

```
1. Mark all instruction offsets as UNMARKED
2. Mark entry_point as BLOCK_START
3. For each instruction at offset O:
   a. If instruction is jump target: mark O as BLOCK_START
   b. If instruction is branch/jump/ret: mark O+size as BLOCK_START
   c. If instruction is call: mark called address as BLOCK_START
4. Create blocks between consecutive BLOCK_START markers
5. For each block, determine entry_type and exit_type
```

## Algorithm: CFG Construction

```
1. For each basic block B:
   a. Get last instruction I
   b. If I is unconditional jump:
      - Add edge B -> target_block (UNCONDITIONAL)
   c. If I is conditional jump:
      - Add edge B -> target_block (CONDITIONAL_TRUE)
      - Add edge B -> next_block (CONDITIONAL_FALSE)
   d. If I is return:
      - Mark B as function exit
   e. If I is call:
      - Add edge B -> called_function (CALL)
      - Add edge B -> next_block (FALL_THROUGH)
   f. Otherwise:
      - Add edge B -> next_block (FALL_THROUGH)
```

## Algorithm: Loop Detection (Natural Loops)

```
1. Compute dominators for all blocks
2. Find back edges: edge A -> B where B dominates A
3. For each back edge (A, B):
   - B is loop header
   - A is back edge source
   - Loop body = all blocks that can reach A without going through B
```

## Success Criteria

### Accuracy Requirements
- Basic block detection: 98%+ on test binaries
- CFG edge accuracy: 98%+ 
- Function boundary detection: 95%+
- Loop detection: 90%+

### Adversarial Cases (Must Refuse)
- Obfuscated control flow (opaque predicates)
- Self-modifying code
- Indirect jumps without resolution
- Overlapping instructions

## Test Categories

### 1. Basic Block Tests
- Simple linear code
- Single conditional
- Multiple conditionals
- Nested conditions
- Switch statements (jump tables)

### 2. CFG Tests
- Sequential blocks
- Diamond pattern (if-else)
- Multiple exits
- Unreachable code

### 3. Function Tests
- Simple function (prologue/epilogue)
- Recursive function
- Tail calls
- Multiple return points

### 4. Loop Tests
- While loop
- For loop
- Do-while loop
- Nested loops
- Break/continue patterns

### 5. Adversarial Tests
- Indirect calls/jumps
- Exception handling
- Computed gotos
- Overlapping instructions

## Training Data

### Sources
1. **Synthetic** — Generated control flow patterns
2. **Compiled** — Real binaries compiled with debug info
3. **Adversarial** — Edge cases that should be refused

### Format
```json
{
  "instructions": [
    {"offset": 0, "mnemonic": "push", "operands": ["rbp"], ...},
    {"offset": 1, "mnemonic": "mov", "operands": ["rbp", "rsp"], ...},
    ...
  ],
  "basic_blocks": [
    {"id": 0, "start": 0, "end": 10, "exit_type": "conditional"},
    {"id": 1, "start": 11, "end": 20, "exit_type": "jump"},
    ...
  ],
  "cfg_edges": [
    {"source": 0, "target": 1, "type": "conditional_true"},
    {"source": 0, "target": 2, "type": "conditional_false"},
    ...
  ],
  "functions": [...],
  "loops": [...]
}
```

## Dependencies

- **Level 0** — Raw instruction parsing
- **Level 1** — Instruction semantics (control flow type)
- **Capstone** — Disassembly (via Level 0)

## Verification Tools

1. **Ghidra** — Compare CFG with Ghidra's analysis
2. **radare2** — Cross-validate basic blocks
3. **angr** — Verify CFG construction
4. **objdump** — Basic block boundaries

## Milestones

| Milestone | Description | Target |
|-----------|-------------|--------|
| M1 | Types and interfaces defined | Day 1 |
| M2 | Basic block detection working | Day 2 |
| M3 | CFG construction working | Day 3 |
| M4 | Function detection working | Day 4 |
| M5 | Loop detection working | Day 5 |
| M6 | Comprehensive exam 98%+ | Day 6 |
| M7 | Dataset generated | Day 7 |
| M8 | Model trained and verified | Day 8 |

## Open Questions

1. How to handle indirect jumps? (refuse vs. heuristic)
2. Should we detect inlined functions?
3. How to represent exception handling?
4. Should Level 2 output be text or structured?

## References

- [Compilers: Principles, Techniques, and Tools](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools) — Dragon Book
- [angr CFG documentation](https://docs.angr.io/en/latest/)
- [Ghidra Decompiler](https://ghidra-sre.org/)
