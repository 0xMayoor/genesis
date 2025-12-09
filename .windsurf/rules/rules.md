---
trigger: always_on
---

# GENESIS Project Rules

## For All AI Teammates

These rules are **mandatory** for any AI working on this project. No exceptions.

---

## 🎯 Project Context

You are working on **GENESIS** — a bottom-up hierarchical code agent trained like a human learns programming. Read `PROJECT_VISION.md` first to understand the full context.

### Core Goals:
1. Build a master-level coding AI from machine code up to high-level reasoning
2. Achieve **zero hallucination** through rigorous testing and verification
3. Use small, modular models — not one giant model
4. Intelligence through clever mechanics, not just model size

---

## 🚨 Critical Rules (Never Violate)

### 1. TEST FIRST
- **Never** write implementation code without corresponding tests
- Tests define the contract; implementation fulfills it
- If you can't test it, don't build it

### 2. VERIFY EVERYTHING
- All generated code must pass through deterministic verification
- Compiler output > model confidence
- Static analysis > assumptions
- Unit tests > "it looks right"

### 3. NO HALLUCINATION TOLERANCE
- If uncertain, the correct output is "cannot determine"
- Never guess, extrapolate, or assume
- When in doubt, ask for clarification or refuse

### 4. RESPECT MODULE BOUNDARIES
- Each level/module has a strict domain
- Don't let modules handle out-of-scope inputs
- Clear interfaces between components

### 5. DOCUMENT DECISIONS
- Every architectural decision needs an ADR (Architecture Decision Record)
- Future AI teammates need to understand WHY, not just WHAT
- Update docs when things change

### 6. INCREMENTAL PROGRESS
- Small, working commits over large, broken changes
- Each checkpoint should be functional
- Don't break existing tests to add new features

---

## 📁 File Organization Rules

### Where Things Go:
- `/docs/` — All documentation
- `/core/` — Framework code (testing, verification, training)
- `/levels/` — The 5-level model stack
- `/datasets/` — Training data and generators
- `/models/` — Checkpoints (gitignored, except configs)
- `/tools/` — Utility scripts
- `/tests/` — All test suites

### Naming Conventions:
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_<module_name>.py`

### What NOT to Create:
- Random scratch files in root
- Duplicate implementations
- Temporary files (use `/tmp` if needed)
- Large binary files (use git-lfs or external storage)

---

## 🧪 Testing Requirements

### Every Module Must Have:
1. **Unit tests** — Test individual functions
2. **Integration tests** — Test module interactions
3. **Adversarial tests** — Trick questions that must fail gracefully
4. **Regression tests** — Ensure no drift

### Test Coverage Gates:
- Level promotion requires **98%+ pass rate**
- Adversarial tests require **100% correct refusal**
- Zero regressions allowed

### Test Naming:
```python
def test_<function>_<scenario>_<expected_outcome>():
    # test_parse_instruction_valid_mov_returns_parsed
    # test_parse_instruction_invalid_opcode_raises_error
    # test_parse_instruction_ambiguous_input_returns_uncertain
```

---

## 🔧 Code Quality Rules

### Python Standards:
- Python 3.10+ features allowed
- Type hints required for all public functions
- Docstrings required for all public functions
- Use `ruff` for linting, `black` for formatting

### Error Handling:
- Explicit error types, not generic exceptions
- Errors should be informative and actionable
- "Cannot determine" is a valid return, not an error

### Dependencies:
- Minimize external dependencies
- Pin versions in requirements.txt
- Document why each dependency is needed

---

## 📝 Documentation Rules

### Required Docs:
- `PROJECT_VISION.md` — North star (already exists)
- `docs/architecture/` — System design
- `docs/levels/level_N.md` — Spec for each level
- `docs/decisions/ADR-NNN.md` — Architecture decisions

### ADR Format:
```markdown
# ADR-NNN: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue we're addressing?

## Decision
What did we decide?

## Consequences
What are the trade-offs?
```

### Code Comments:
- Explain WHY, not WHAT
- Complex logic needs explanation
- TODO comments must have context and owner

---

## 🤖 AI Collaboration Rules

### When Starting Work:
1. Read `PROJECT_VISION.md` if unfamiliar
2. Check relevant `/docs/` for context
3. Review existing tests before implementing
4. Ask for clarification if requirements are unclear

### When Making Changes:
1. Write/update tests first
2. Implement the change
3. Verify all tests pass
4. Update documentation if needed
5. Summarize what was done

### When Uncertain:
- **Ask** rather than guess
- **Refuse** rather than hallucinate
- **Document** the uncertainty for future reference

### Handoff Protocol:
When ending a session, leave a clear status:
- What was completed
- What's in progress
- What's blocked
- Next steps

---

## 🚫 Anti-Patterns to Avoid

1. **"It works on my machine"** — Must work in defined environment
2. **"I'll add tests later"** — Tests come first
3. **"This is temporary"** — Temporary code becomes permanent
4. **"The model is confident"** — Confidence ≠ correctness
5. **"It's obvious"** — Document it anyway
6. **"One big change"** — Small incremental changes
7. **"Skip verification"** — Never skip verification

---

## 📊 Progress Tracking

### Use These Status Labels:
- `🔴 BLOCKED` — Cannot proceed without resolution
- `🟡 IN_PROGRESS` — Actively being worked on
- `🟢 COMPLETE` — Done and tested
- `⚪ PENDING` — Not yet started
- `🔵 REVIEW` — Needs human review

### Milestone Checkpoints:
Each level has defined checkpoints in `/docs/levels/level_N.md`

---

## 🔐 Security Rules

- No hardcoded secrets or API keys
- Use environment variables for configuration
- No execution of untrusted code outside sandbox
- Log security-relevant events

---

*These rules exist to ensure project success. Follow them.*
