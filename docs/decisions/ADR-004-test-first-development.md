# ADR-004: Test-First Development

## Status
Accepted

## Context
We need a development methodology that ensures:
- Code correctness
- Zero hallucination
- Maintainability
- Clear specifications

## Decision
Adopt **strict test-first development**:
- Tests are written BEFORE implementation
- No code merges without passing tests
- Tests define the contract
- 98%+ pass rate required for level promotion

## Rationale

### Why test-first:
1. **Tests as specification** — Tests define what the code should do before it exists
2. **Prevents hallucination** — Can't claim something works without proof
3. **Forces clear thinking** — Must understand requirements before coding
4. **Enables refactoring** — Safe to change implementation if tests pass
5. **Documentation** — Tests document expected behavior

### Test categories required:
1. **Unit tests** — Individual function behavior
2. **Integration tests** — Module interactions
3. **Adversarial tests** — Must-refuse scenarios
4. **Regression tests** — Prevent drift

## Consequences

### Positive
- Higher code quality
- Clear specifications
- Safe refactoring
- Documented behavior
- Confidence in changes

### Negative
- Slower initial development
- More code to maintain
- Requires discipline
- Some tests may be wrong

## Implementation Notes
- Use pytest as test framework
- Adversarial tests are mandatory, not optional
- 98%+ pass rate for unit tests
- 100% pass rate for adversarial tests (must refuse correctly)
- No level promotion without passing all tests
