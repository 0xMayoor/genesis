---
description: Verify model/code outputs using deterministic tools
auto_execution_mode: 3
---

# Output Verification Workflow

Use this workflow to verify any generated code or model output.

## Core Principle

**Never trust model confidence. Always verify with deterministic tools.**

## Verification Layers

### Layer 1: Syntax Verification
Check that output is syntactically valid.

```bash
# Python
python -m py_compile <file.py>

# C/C++
gcc -fsyntax-only <file.c>

# JavaScript
node --check <file.js>

# Rust
rustc --emit=metadata <file.rs>
```

### Layer 2: Static Analysis
Check for common issues without execution.

```bash
# Python
ruff check <file.py>
mypy <file.py>

# C
cppcheck <file.c>

# General
semgrep --config auto <file>
```

### Layer 3: Execution Verification
Run the code and verify behavior.

```bash
# Run tests
pytest <test_file.py> -v

# Run with coverage
pytest --cov=<module> <test_file.py>
```

### Layer 4: Semantic Verification
Verify the output means what we expect.

- Compare against known-good outputs
- Check invariants hold
- Verify edge cases handled

## Verification Checklist

For any generated code:

- [ ] Syntax valid (compiles/parses)
- [ ] No static analysis errors
- [ ] All tests pass
- [ ] Edge cases handled
- [ ] Uncertainty expressed when appropriate
- [ ] No hallucinated imports/functions

## When Verification Fails

1. **Log the failure** — What failed and why
2. **Analyze the cause** — Is it a model issue or test issue?
3. **Retry with feedback** — Give the model the error
4. **Escalate if persistent** — Flag for human review

## Automated Verification Script

Location: `/home/kali/code/tools/verify.py`

```python
# Usage
python tools/verify.py --file <path> --type <python|c|js|rust>
```

## Integration with Training

All training samples must pass verification before inclusion:
1. Generate sample
2. Run through verification pipeline
3. Only include if all checks pass
4. Log and analyze failures