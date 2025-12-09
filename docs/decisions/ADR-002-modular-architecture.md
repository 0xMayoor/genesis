# ADR-002: Modular Multi-Model Architecture

## Status
Accepted

## Context
We need to decide between:
1. One large model trained on everything
2. Multiple small specialized models

Hardware constraint: 16GB RAM laptop, no dedicated GPU.

## Decision
Use **multiple small specialized models** (50M-300M params each), one per abstraction level, coordinated by an orchestrator.

## Rationale

### Why NOT one large model:
- Cannot train/run large models on target hardware
- Large models hallucinate more on specialized tasks
- Harder to verify and debug
- All-or-nothing failure mode
- Cannot update individual capabilities

### Why multiple small models:
- Each model is focused and verifiable
- Fits hardware constraints
- Can update/replace individual modules
- Easier to test and validate
- Failure is isolated to one module
- Matches human cognitive architecture

## Consequences

### Positive
- Lower compute requirements
- Better testability
- Modular updates
- Clear responsibility boundaries

### Negative
- More complex orchestration
- Inter-module communication overhead
- Need to maintain multiple models
- Integration testing complexity

## Implementation Notes
- Each level gets its own model
- Models communicate through defined interfaces
- Orchestrator handles routing and validation
- LoRA fine-tuning for efficiency
