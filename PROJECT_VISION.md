# Project Codename: GENESIS

## The Ultimate Bottom-Up Code Agent

### Vision Statement

Build a **master-level coding AI agent** trained from the ground up like a human learns programming — starting from machine code primitives and progressively building to high-level reasoning. Unlike traditional LLMs that memorize patterns, this agent will **understand** code by knowing how every abstraction layer connects to the ones below it.

---

## Core Philosophy

### 1. Human-Like Learning
- Humans don't learn programming by seeing millions of examples
- They learn **atomic concepts** first, then **compose** them
- We train the same way: primitives → composition → abstraction → reasoning

### 2. Zero-Hallucination by Design
- Every output must be verifiable
- "I don't know" is a valid and expected response
- Deterministic tools (compilers, tests) are the source of truth
- No guessing, no extrapolation beyond training

### 3. Modular Architecture
- Many small specialized models, not one giant model
- Each module has a strict domain boundary
- Modules are independently testable and replaceable
- An orchestrator combines them into unified intelligence

### 4. Brainpower Through Mechanics
- Intelligence comes from clever code, not just model size
- Verification loops, retry logic, decomposition, self-consistency
- Hard constraints enforced by code, not learned

---

## The 5-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LEVEL 5: AGENT LAYER                     │
│         Planning, Tool Use, Multi-Step Reasoning            │
├─────────────────────────────────────────────────────────────┤
│                 LEVEL 4: HIGH-LEVEL LANGUAGES               │
│            Python, C, Rust, JS — Full Code Abilities        │
├─────────────────────────────────────────────────────────────┤
│              LEVEL 3: SEMANTIC REASONING                    │
│           AST, Control Flow, Intent, Data Flow              │
├─────────────────────────────────────────────────────────────┤
│              LEVEL 2: INTERMEDIATE REPRESENTATIONS          │
│              LLVM IR, Bytecode, Structured Graphs           │
├─────────────────────────────────────────────────────────────┤
│              LEVEL 1: ASSEMBLY LANGUAGE                     │
│           x86, ARM, Control Flow, Stack Frames              │
├─────────────────────────────────────────────────────────────┤
│              LEVEL 0: MACHINE CODE PATTERNS                 │
│            Binary, Opcodes, Instruction Boundaries          │
└─────────────────────────────────────────────────────────────┘
```

Each level:
- Has its own small model (50M-300M params)
- Has its own test suite (98%+ pass required)
- Must pass gate before contributing to higher levels
- Can be trained/updated independently

---

## Zero-Hallucination Strategy

### The 5 Pillars

1. **Explicit Uncertainty**
   - Models trained to output "cannot determine" when uncertain
   - Confidence scores on all outputs
   - Hard rejection thresholds

2. **Deterministic Verification**
   - All code outputs run through compiler/interpreter
   - Static analysis validates structure
   - Unit tests verify behavior

3. **Adversarial Testing**
   - Every level has trick questions
   - Ambiguous inputs must produce "uncertain" outputs
   - Regression tests prevent drift

4. **Domain Boundaries**
   - Each module only handles its domain
   - Out-of-scope inputs are rejected, not guessed
   - Clear interfaces between modules

5. **Composition Validation**
   - Orchestrator validates cross-module outputs
   - End-to-end tests span multiple modules
   - Consistency checks at every handoff

---

## Brainpower Mechanics (Code, Not Compute)

### For Code Modules:
- **Chain-of-Verification**: Generate → Verify → Retry if fail
- **Self-Consistency**: Multiple generations must agree
- **Decomposition**: Complex → Atomic sub-problems → Recombine
- **Symbolic Constraints**: Hard rules enforced by code

### For Orchestrator:
- **Multi-Agent Debate**: Modules argue, consensus required
- **Backtracking**: Analyze failures, try alternative approaches
- **Meta-Reasoning**: "Do I have enough information?"
- **Memory Management**: Short-term, long-term, episodic

---

## Technical Constraints

### Hardware Target
- 16GB RAM laptop
- No dedicated GPU required (nice to have)
- CPU-optimized training

### Model Strategy
- Base: Tiny pretrained models (TinyLlama, Phi-mini, etc.)
- Fine-tuning: LoRA adapters (~3-6% new params)
- No training from scratch
- Vocabulary pruning for efficiency

### What We DON'T Train On
- General internet knowledge
- Social/conversational data
- Wikipedia, news, books
- Anything not directly code-related

---

## Success Criteria

### Per-Level Gates
- 98%+ accuracy on level-specific test suite
- 100% on adversarial "trick" tests (must refuse)
- Zero regressions on previous level tests

### End-to-End
- Can analyze any code from binary to source
- Can explain code at any abstraction level
- Can translate between languages
- Can debug with reasoning traces
- Never hallucinates — refuses when uncertain

---

## Timeline (AI-Assisted)

| Phase | Duration | Focus |
|-------|----------|-------|
| A: Foundation | Weeks 1-4 | Architecture, test framework, infrastructure |
| B: Low-Level | Months 2-3 | Levels 0-2 training and validation |
| C: High-Level | Months 3-4 | Levels 3-4 training and validation |
| D: Agent | Months 5-6 | Orchestrator, integration, refinement |

**Target: Working v1 in 4-6 months**

---

## Project Structure

```
/genesis
├── docs/                    # Documentation
│   ├── architecture/        # System design docs
│   ├── levels/              # Per-level specifications
│   └── decisions/           # Architecture Decision Records
├── core/                    # Core framework
│   ├── testing/             # Test harness framework
│   ├── verification/        # Deterministic verification tools
│   ├── training/            # Training infrastructure
│   └── orchestrator/        # Module orchestration
├── levels/                  # The 5-level stack
│   ├── level0_machine/      # Machine code patterns
│   ├── level1_assembly/     # Assembly reasoning
│   ├── level2_ir/           # Intermediate representations
│   ├── level3_semantics/    # Code semantics
│   └── level4_highlevel/    # High-level languages
├── datasets/                # Training data
│   ├── generators/          # Programmatic data generation
│   └── validated/           # Verified training samples
├── models/                  # Trained model checkpoints
├── tools/                   # Utility scripts
└── tests/                   # Test suites
    ├── unit/                # Per-module tests
    ├── integration/         # Cross-module tests
    ├── adversarial/         # Trick questions
    └── regression/          # Prevent drift
```

---

## References

### Similar Projects (for inspiration, not copying)
- LLM4Decompile — Binary to C decompilation
- Neutron — Neural decompiler with rule-assisted recovery
- DecompilerAI — T5-based assembly to C
- jTrans — Binary code embeddings

### Key Differentiators
- Full bottom-up curriculum (none do this)
- Strict zero-hallucination enforcement
- Modular multi-model architecture
- Brainpower through mechanics, not size

---

## Guiding Principles

1. **Test First** — No code without tests
2. **Verify Everything** — Trust deterministic tools, not model confidence
3. **Small and Focused** — Each module does one thing well
4. **Refuse When Uncertain** — "I don't know" is success, not failure
5. **Build Up, Not Down** — Master primitives before abstractions
6. **Document Decisions** — Future AI teammates need context
7. **Incremental Progress** — Working checkpoints over perfect plans

---

*This document is the north star. All work should align with this vision.*
