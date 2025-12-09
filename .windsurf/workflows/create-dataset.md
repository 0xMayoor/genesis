---
description: Create training datasets for GENESIS levels
auto_execution_mode: 3
---

# Dataset Creation Workflow

Use this workflow when creating training data for any level.

## Core Principles

1. **Programmatic generation** — Never manually write training samples
2. **Verification required** — Every sample must pass deterministic checks
3. **Include adversarial** — 10-20% of samples should be "trick" questions
4. **Balanced distribution** — Cover all categories evenly

## Dataset Structure

```
datasets/
├── generators/
│   └── level{N}_generator.py
├── validated/
│   └── level{N}/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── adversarial.jsonl
└── raw/
    └── level{N}/
        └── (unvalidated samples)
```

## Sample Format

```json
{
  "id": "level0_001234",
  "input": "...",
  "expected_output": "...",
  "category": "valid|adversarial|edge_case",
  "difficulty": "easy|medium|hard",
  "verified": true,
  "verification_method": "compiler|test|manual",
  "metadata": {}
}
```

## Steps to Create Dataset

### 1. Define Categories
What types of samples does this level need?
- Valid inputs with clear outputs
- Edge cases
- Adversarial (ambiguous, malformed, out-of-scope)

### 2. Create Generator Script

Location: `/home/kali/code/datasets/generators/level{N}_generator.py`

```python
"""
Dataset generator for Level {N}.

Generates:
- Valid samples: {description}
- Adversarial samples: {description}
- Edge cases: {description}
"""

def generate_valid_sample() -> dict:
    """Generate a valid input-output pair."""
    ...

def generate_adversarial_sample() -> dict:
    """Generate a sample that should be refused."""
    ...

def verify_sample(sample: dict) -> bool:
    """Verify sample using deterministic tools."""
    ...

def main():
    # Generate samples
    # Verify each
    # Save to validated/
    ...
```

### 3. Generate Raw Samples
// turbo
```bash
python datasets/generators/level{N}_generator.py --output datasets/raw/level{N}/ --count 10000
```

### 4. Verify All Samples
// turbo
```bash
python tools/verify_dataset.py --input datasets/raw/level{N}/ --output datasets/validated/level{N}/
```

### 5. Split into Train/Val/Test
// turbo
```bash
python tools/split_dataset.py --input datasets/validated/level{N}/ --train 0.8 --val 0.1 --test 0.1
```

### 6. Validate Distribution
- Check category balance
- Check difficulty distribution
- Ensure adversarial samples present

## Adversarial Sample Requirements

Every dataset MUST include adversarial samples:

| Type | Description | Expected Output |
|------|-------------|-----------------|
| Ambiguous | Multiple valid interpretations | "cannot determine" |
| Incomplete | Missing required information | "insufficient data" |
| Malformed | Invalid syntax/structure | Error or refusal |
| Out-of-scope | Beyond level's domain | "out of scope" |
| Contradictory | Conflicting information | "contradiction detected" |

## Quality Checks

Before using dataset:
- [ ] All samples verified
- [ ] Category distribution balanced
- [ ] Adversarial samples included (10-20%)
- [ ] No duplicate samples
- [ ] Metadata complete
- [ ] Format consistent