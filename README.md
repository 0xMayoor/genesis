# GENESIS

> A bottom-up hierarchical code agent trained like a human learns programming.

## Vision

Build a master-level coding AI from the ground up — starting from machine code primitives and progressively building to high-level reasoning. Unlike traditional LLMs that memorize patterns, GENESIS **understands** code by knowing how every abstraction layer connects.

See [PROJECT_VISION.md](PROJECT_VISION.md) for the full vision.

## Architecture

```
Level 4: High-Level Languages (Python, C, Rust, JS)
Level 3: Semantic Reasoning (AST, Intent, Data Flow)
Level 2: Intermediate Representations (LLVM IR, Bytecode)
Level 1: Assembly Language (x86, ARM)
Level 0: Machine Code Patterns (Binary, Opcodes)
```

Each level is a small, specialized model (50M-300M params) that can be independently tested and verified.

## Key Principles

1. **Zero Hallucination** — Every output is verified; "I don't know" is valid
2. **Test First** — No implementation without tests
3. **Modular** — Small focused models, not one giant model
4. **Bottom-Up** — Master primitives before abstractions

## Setup

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Project Structure

```
genesis/
├── core/                 # Framework code
│   ├── testing/          # Test harness
│   ├── verification/     # Deterministic verification
│   ├── training/         # Training infrastructure
│   └── orchestrator/     # Module coordination
├── levels/               # The 5-level stack
│   ├── level0_machine/   # Machine code patterns
│   ├── level1_assembly/  # Assembly reasoning
│   ├── level2_ir/        # IR/bytecode
│   ├── level3_semantics/ # Code semantics
│   └── level4_highlevel/ # High-level languages
├── datasets/             # Training data
├── models/               # Trained checkpoints
├── tests/                # Test suites
└── docs/                 # Documentation
```

## Documentation

- [Project Vision](PROJECT_VISION.md)
- [System Architecture](docs/architecture/system-design.md)
- [Current Status](docs/STATUS.md)
- [Architecture Decisions](docs/decisions/)

## License

MIT
