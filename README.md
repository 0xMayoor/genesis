# GENESIS

> A bottom-up hierarchical code agent trained like a human learns programming.

## Current Status

| Level | Task | Status | Accuracy |
|-------|------|--------|----------|
| **Level 0** | bytes → mnemonic | ✅ **COMPLETE** | **100%** |
| Level 1 | instruction → semantics | 🔲 Pending | - |
| Level 2 | block → CFG | 🔲 Pending | - |
| Level 3 | function → intent | 🔲 Pending | - |
| Level 4 | program → reasoning | 🔲 Pending | - |

## Vision

Build a master-level coding AI from the ground up — starting from machine code primitives and progressively building to high-level reasoning. Unlike traditional LLMs that memorize patterns, GENESIS **understands** code by knowing how every abstraction layer connects.

See [PROJECT_VISION.md](PROJECT_VISION.md) for the full vision.

## Architecture

```
Level 4: High-Level Languages (Python, C, Rust, JS)      [Pending]
Level 3: Semantic Reasoning (AST, Intent, Data Flow)     [Pending]
Level 2: Intermediate Representations (LLVM IR, CFG)     [Pending]
Level 1: Assembly Language (x86, ARM semantics)          [Pending]
Level 0: Machine Code Patterns (bytes → mnemonic)        [COMPLETE ✓]
```

Each level is a small, specialized model (~1M params) that can be independently tested and verified.

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
├── models/               # Trained checkpoints
│   └── level0/           # ✓ 100% accuracy byte classifier
├── notebooks/            # Training scripts
│   └── train_level0.py   # Level 0 training (Colab-ready)
├── levels/               # Level implementations
│   ├── level0_machine/   # ✓ Complete
│   ├── level1_assembly/  # In progress
│   ├── level2_ir/        # Pending
│   ├── level3_semantics/ # Pending
│   └── level4_highlevel/ # Pending
├── core/                 # Framework code
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
