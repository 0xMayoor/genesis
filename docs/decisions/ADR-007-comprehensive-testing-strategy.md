# ADR-007: Comprehensive Testing Strategy

## Status
Accepted

## Context
Building a zero-hallucination code agent requires absolute confidence in correctness. Traditional unit testing alone is insufficient because:

1. **Test author bias**: The same person writing code and tests may unconsciously avoid edge cases
2. **Limited coverage**: Manual tests only cover cases the author anticipates
3. **No external validation**: Tests may pass but produce incorrect results

We need a testing strategy that catches errors the test author didn't anticipate.

## Decision
Implement a four-layer testing strategy:

### Layer 1: Unit Tests
Traditional tests for expected behavior.
- Test happy paths
- Test error handling
- Test boundary conditions

### Layer 2: Property-Based Testing (Hypothesis)
Generate random inputs to verify invariants always hold.
- Module never crashes on any input
- Output types are always correct
- Confidence is always in valid range
- Uncertain outputs always have reasons

### Layer 3: Fuzzing
Adversarial inputs designed to break the module.
- Malformed data patterns
- Resource exhaustion attempts
- Architecture mismatches
- Boundary violations

### Layer 4: External Validation
Compare against external tools as ground truth.
- objdump (GNU binutils)
- radare2
- ndisasm (NASM)

If GENESIS disagrees with ALL external tools, GENESIS is likely wrong.

## Test Categories

| Category | Purpose | Pass Requirement |
|----------|---------|------------------|
| Unit | Expected behavior | 98%+ |
| Property | Invariant verification | 100% |
| Fuzz | Crash resistance | 100% no crashes |
| Adversarial | Correct refusal | 100% |
| External | Ground truth match | 100% (when tools available) |

## Consequences

### Positive
- Catches bugs the author didn't anticipate
- External validation provides ground truth
- Property tests find edge cases automatically
- Fuzz tests ensure robustness

### Negative
- More test code to maintain
- Slower test runs (property/fuzz tests)
- External tool dependencies (optional)

### Mitigations
- Property tests have configurable example counts
- External validation skips gracefully if tools unavailable
- Fast unit tests run first, slow tests run in CI

## Implementation
```
tests/
├── unit/           # Traditional unit tests
├── property/       # Hypothesis property tests
├── fuzz/           # Fuzzing tests
└── validation/     # External tool validation
```

## Related
- ADR-003: Zero-Hallucination Strategy
- ADR-004: Test-First Development
