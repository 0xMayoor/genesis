# ADR-005: LoRA Fine-Tuning Strategy

## Status
Accepted

## Context
We need to train specialized models on limited hardware (16GB RAM, no GPU). Full model training is not feasible. We need an efficient fine-tuning approach.

## Decision
Use **LoRA (Low-Rank Adaptation)** for all model fine-tuning.

## Rationale

### What is LoRA:
- Freezes pretrained model weights
- Adds small trainable adapter layers
- Typically 1-10% of original parameters
- Achieves comparable performance to full fine-tuning

### Why LoRA:
1. **Memory efficient** — Only adapter weights in GPU/RAM
2. **Fast training** — Fewer parameters to update
3. **Composable** — Can swap adapters for different tasks
4. **Preserves base knowledge** — Base model unchanged
5. **Hardware compatible** — Works on 16GB RAM

### Alternative considered:
- Full fine-tuning: Too memory intensive
- Prompt tuning: Less expressive
- Prefix tuning: Less flexible than LoRA

## Consequences

### Positive
- Fits hardware constraints
- Fast iteration
- Can experiment with different adapters
- Base model knowledge preserved

### Negative
- Slightly lower ceiling than full fine-tuning
- Requires compatible base model
- Additional complexity in model loading

## Implementation Notes
- Use HuggingFace PEFT library
- Typical LoRA config: r=8, alpha=16, dropout=0.1
- Target modules: query, key, value projections
- Save only adapter weights (small files)
