---
description: Add tests to GENESIS components
auto_execution_mode: 3
---

# Adding Tests to GENESIS

Use this workflow when adding tests to any component.

## Test Categories

### 1. Unit Tests
Test individual functions in isolation.

Location: `tests/unit/` or `levels/level{N}/tests/test_unit.py`

```python
def test_<function>_<scenario>_<expected>():
    """Test that <function> <does what> when <condition>."""
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected
```

### 2. Integration Tests
Test interactions between modules.

Location: `tests/integration/`

```python
def test_<moduleA>_<moduleB>_<scenario>():
    """Test that moduleA and moduleB work together for <scenario>."""
    ...
```

### 3. Adversarial Tests
Test that the system correctly refuses ambiguous/invalid inputs.

Location: `tests/adversarial/` or `levels/level{N}/tests/test_adversarial.py`

```python
def test_<function>_ambiguous_input_returns_uncertain():
    """Test that <function> refuses to guess on ambiguous input."""
    # Arrange
    ambiguous_input = ...
    
    # Act
    result = function_under_test(ambiguous_input)
    
    # Assert
    assert result.is_uncertain == True
    assert "cannot determine" in result.message.lower()
```

### 4. Regression Tests
Ensure previous functionality still works.

Location: `tests/regression/`

## Steps to Add Tests

### 1. Identify What to Test
- What function/module needs testing?
- What are the edge cases?
- What inputs should be refused?

### 2. Write the Test First
- Create test file if not exists
- Write failing test
- Document what the test verifies

### 3. Verify Test Fails
// turbo
```bash
pytest path/to/test_file.py -v -x
```

### 4. Implement to Pass
- Write minimal code to pass
- Don't over-engineer

### 5. Verify Test Passes
// turbo
```bash
pytest path/to/test_file.py -v
```

### 6. Run Full Suite
// turbo
```bash
pytest tests/ -v
```

## Adversarial Test Requirements

Every level MUST have adversarial tests for:
- Ambiguous inputs
- Missing information
- Malformed data
- Out-of-scope requests
- Contradictory inputs

The correct response is ALWAYS to refuse/return uncertain.

## Test Naming Convention

```
test_<function>_<scenario>_<expected_outcome>
```

Examples:
- `test_parse_opcode_valid_mov_returns_instruction`
- `test_parse_opcode_invalid_bytes_raises_error`
- `test_parse_opcode_ambiguous_prefix_returns_uncertain`