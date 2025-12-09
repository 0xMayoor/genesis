# GENESIS System Architecture

## Overview

GENESIS is a modular, hierarchical code understanding system built from the ground up. It consists of 5 specialized levels, each handling a specific abstraction layer of code, coordinated by an orchestrator.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (CLI / API / IDE Plugin)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Router    │ │  Validator  │ │   Planner   │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Retry Logic │ │ Doubt Layer │ │   Memory    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   LEVEL 4     │     │   LEVEL 3     │     │   LEVEL 2     │
│  High-Level   │────▶│   Semantic    │────▶│      IR       │
│   Languages   │     │   Reasoning   │     │   Bytecode    │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                              ┌─────────────────────┘
                              ▼
                      ┌───────────────┐     ┌───────────────┐
                      │   LEVEL 1     │────▶│   LEVEL 0     │
                      │   Assembly    │     │ Machine Code  │
                      └───────────────┘     └───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFICATION LAYER                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Compiler   │ │   Static    │ │    Test     │               │
│  │   Check     │ │  Analysis   │ │   Runner    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. User Interface Layer

**Purpose**: Accept user requests and return results.

**Components**:
- CLI tool for command-line usage
- REST API for programmatic access
- (Future) IDE plugins

**Responsibilities**:
- Parse user input
- Format output
- Handle errors gracefully

---

### 2. Orchestrator

**Purpose**: Coordinate all modules and ensure correctness.

**Components**:

| Component | Purpose |
|-----------|---------|
| **Router** | Determine which level(s) to invoke |
| **Validator** | Check outputs against inputs |
| **Planner** | Break complex tasks into steps |
| **Retry Logic** | Handle failures, try alternatives |
| **Doubt Layer** | Challenge uncertain outputs |
| **Memory** | Track context across steps |

**Key Behaviors**:
- Never trust single module output blindly
- Always validate before returning
- Retry with feedback on failure
- Refuse when confidence is low

---

### 3. Level Modules

Each level is a self-contained module with:
- Its own small model (50M-300M params)
- Defined input/output interfaces
- Internal verification
- Test suite

#### Level 0: Machine Code Patterns
- **Input**: Raw bytes, binary sequences
- **Output**: Instruction boundaries, opcode identification
- **Model Size**: ~50M params
- **Key Skills**: Pattern recognition, byte parsing

#### Level 1: Assembly Language
- **Input**: Disassembled instructions
- **Output**: Control flow, stack analysis, pseudocode
- **Model Size**: ~100M params
- **Key Skills**: Instruction semantics, register tracking

#### Level 2: Intermediate Representations
- **Input**: IR (LLVM, bytecode), control flow graphs
- **Output**: Structured program representation
- **Model Size**: ~150M params
- **Key Skills**: Graph reasoning, optimization patterns

#### Level 3: Semantic Reasoning
- **Input**: AST, structured code
- **Output**: Intent, data flow, semantic meaning
- **Model Size**: ~200M params
- **Key Skills**: Type inference, intent recognition

#### Level 4: High-Level Languages
- **Input**: Source code, natural language requests
- **Output**: Code generation, translation, explanation
- **Model Size**: ~300M params
- **Key Skills**: Multi-language, debugging, refactoring

---

### 4. Verification Layer

**Purpose**: Provide ground truth for all outputs.

**Components**:

| Tool | Purpose |
|------|---------|
| **Compiler Check** | Syntax validation |
| **Static Analysis** | Code quality, type checking |
| **Test Runner** | Behavioral verification |
| **Sandbox** | Safe execution environment |

**Key Principle**: Verification output > Model confidence

---

## Data Flow

### Query Processing

```
1. User submits query
2. Orchestrator receives query
3. Router determines required levels
4. Planner creates execution plan
5. Levels execute in sequence/parallel
6. Each level output is validated
7. Orchestrator combines results
8. Doubt Layer challenges if uncertain
9. Final validation
10. Return to user (or refuse if uncertain)
```

### Verification Flow

```
1. Module generates output
2. Syntax check (compiler)
3. Static analysis
4. Test execution (if applicable)
5. Consistency check with input
6. Confidence scoring
7. Pass/Fail decision
```

---

## Module Interfaces

### Standard Module Interface

```python
class LevelModule(Protocol):
    """Interface all level modules must implement."""
    
    def process(self, input: LevelInput) -> LevelOutput:
        """Process input and return output."""
        ...
    
    def can_handle(self, input: LevelInput) -> bool:
        """Check if this module can handle the input."""
        ...
    
    def confidence(self, input: LevelInput, output: LevelOutput) -> float:
        """Return confidence score 0.0-1.0."""
        ...
    
    def verify(self, input: LevelInput, output: LevelOutput) -> VerificationResult:
        """Verify output using deterministic tools."""
        ...
```

### Standard Output Format

```python
@dataclass
class LevelOutput:
    result: Any                    # The actual output
    confidence: float              # 0.0-1.0
    is_uncertain: bool             # True if should refuse
    uncertainty_reason: str | None # Why uncertain
    verification: VerificationResult
    metadata: dict
```

---

## Uncertainty Handling

### When to Return Uncertain

1. **Ambiguous input** — Multiple valid interpretations
2. **Insufficient data** — Missing required information
3. **Out of scope** — Beyond module's training
4. **Low confidence** — Model unsure
5. **Verification failed** — Output doesn't validate
6. **Contradiction** — Conflicting information

### Uncertainty Response

```python
LevelOutput(
    result=None,
    confidence=0.0,
    is_uncertain=True,
    uncertainty_reason="Cannot determine: input has multiple valid interpretations",
    verification=VerificationResult(passed=False),
    metadata={"attempted_interpretations": [...]}
)
```

---

## Brainpower Mechanics

### Chain-of-Verification
```
Generate → Verify → If fail: Analyze error → Retry with feedback → Verify again
```

### Self-Consistency
```
Generate N candidates → Compare → If disagree: Return uncertain
```

### Decomposition
```
Complex input → Break into sub-problems → Solve each → Recombine → Verify whole
```

### Multi-Agent Debate
```
Module A output → Module B challenges → Resolve or escalate
```

---

## Security Considerations

1. **Sandboxed execution** — All code runs in isolation
2. **No network access** — Modules cannot make external calls
3. **Resource limits** — CPU, memory, time limits
4. **Input sanitization** — Validate all inputs
5. **Output filtering** — No sensitive data leakage

---

## Scalability Path

### Current Target (v1)
- Single machine
- 16GB RAM
- CPU-only training
- Sequential processing

### Future Scaling
- GPU acceleration
- Distributed training
- Parallel module execution
- Larger models per level

---

## Testing Strategy

### Per-Module Testing
- Unit tests for each function
- Integration tests with adjacent levels
- Adversarial tests (must refuse)
- Regression tests

### System Testing
- End-to-end query tests
- Cross-level consistency tests
- Performance benchmarks
- Stress tests

### Continuous Validation
- All training samples verified
- All outputs verified
- Metrics tracked over time
- Drift detection

---

## Dependencies

### Core
- Python 3.10+
- PyTorch (for models)
- Transformers (HuggingFace)
- PEFT (for LoRA)

### Verification
- GCC/Clang (C compilation)
- Python compiler
- Ruff (Python linting)
- Mypy (type checking)

### Testing
- Pytest
- Hypothesis (property-based testing)

### Infrastructure
- Git (version control)
- DVC (data versioning) — optional
