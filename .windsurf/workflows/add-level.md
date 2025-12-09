---
description: Add a new level to the GENESIS architecture
auto_execution_mode: 3
---

# Adding a New Level to GENESIS

Use this workflow when implementing a new level in the 5-level architecture.

## Prerequisites

- Previous level must be complete and passing all tests
- Level specification must exist in `/docs/levels/level_N.md`
- Test framework must be set up

## Steps

### 1. Create Level Specification (if not exists)

Create `/home/kali/code/docs/levels/level_N.md` with:
- Purpose and scope of this level
- Input/output formats
- Success criteria
- Test requirements
- Dependencies on lower levels

### 2. Create Directory Structure

```bash
mkdir -p levels/level{N}_{name}
mkdir -p levels/level{N}_{name}/src
mkdir -p levels/level{N}_{name}/tests
mkdir -p datasets/level{N}
```

### 3. Define Interfaces First

Create `/home/kali/code/levels/level{N}_{name}/src/interfaces.py`:
- Define input types
- Define output types
- Define error types
- Define the "uncertain" response type

### 4. Write Tests Before Implementation

Create test files in `/home/kali/code/levels/level{N}_{name}/tests/`:
- `test_unit.py` — Unit tests for each function
- `test_integration.py` — Integration with lower levels
- `test_adversarial.py` — Trick questions (must refuse)
- `test_regression.py` — Prevent drift

### 5. Create Dataset Generator

Create `/home/kali/code/datasets/generators/level{N}_generator.py`:
- Programmatic generation (not manual)
- Include verification step
- Generate both positive and adversarial samples

### 6. Implement the Level

Create implementation in `/home/kali/code/levels/level{N}_{name}/src/`:
- Follow the interfaces exactly
- Handle uncertainty explicitly
- Include verification hooks

### 7. Validate

// turbo
Run all tests:
```bash
pytest levels/level{N}_{name}/tests/ -v
```

### 8. Document

- Update `/docs/STATUS.md` with completion
- Create ADR if any architectural decisions were made
- Update `PROJECT_VISION.md` if scope changed

## Gate Requirements

Before marking level complete:
- [ ] 98%+ pass rate on unit tests
- [ ] 100% correct refusal on adversarial tests
- [ ] Zero regressions on lower level tests
- [ ] Documentation complete
- [ ] Code reviewed