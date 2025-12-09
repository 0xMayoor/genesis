# ADR-003: Zero-Hallucination Strategy

## Status
Accepted

## Context
Traditional LLMs hallucinate — they generate plausible-sounding but incorrect outputs. For a coding agent, hallucination is unacceptable. We need a strategy to eliminate or minimize hallucination.

## Decision
Implement a **multi-layered zero-hallucination strategy** with the following pillars:

1. **Explicit Uncertainty Training**
2. **Deterministic Verification**
3. **Adversarial Testing**
4. **Domain Boundaries**
5. **Composition Validation**

## Rationale

### Pillar 1: Explicit Uncertainty Training
- Train models to output "cannot determine" when uncertain
- Include confidence scores on all outputs
- Set hard rejection thresholds (refuse if confidence < 0.85)
- "I don't know" is a valid and expected response

### Pillar 2: Deterministic Verification
- All outputs verified by deterministic tools (compilers, tests)
- Tool output > model confidence
- Never trust model output without verification
- Build verification into the pipeline, not as afterthought

### Pillar 3: Adversarial Testing
- Every level has adversarial test suite
- Ambiguous inputs must produce "uncertain" outputs
- 100% correct refusal required on adversarial tests
- Continuous adversarial sample generation

### Pillar 4: Domain Boundaries
- Each module has strict scope
- Out-of-scope inputs are rejected, not guessed
- Clear interfaces prevent scope creep
- Orchestrator enforces boundaries

### Pillar 5: Composition Validation
- Cross-module outputs are validated
- End-to-end tests span multiple modules
- Consistency checks at every handoff
- Orchestrator validates final output

## Consequences

### Positive
- Dramatically reduced hallucination
- Trustworthy outputs
- Clear failure modes
- Auditable decisions

### Negative
- More refusals (some valid queries may be refused)
- Slower processing (verification overhead)
- More complex training (uncertainty labels needed)
- Higher test maintenance burden

## Implementation Notes
- Uncertainty must be a first-class output type
- Verification layer is mandatory, not optional
- Adversarial tests are as important as positive tests
- Gate promotion on adversarial test performance
