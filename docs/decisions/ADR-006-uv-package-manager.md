# ADR-006: Use UV as Python Package Manager

## Status
Accepted

## Context
We need a Python package manager for the project. Options include:
- pip (standard, slow)
- poetry (feature-rich, complex)
- pdm (modern, PEP-compliant)
- uv (extremely fast, Rust-based)

## Decision
Use **uv** as the Python package manager.

## Rationale

### Why uv:
1. **Extremely fast** — 10-100x faster than pip
2. **Drop-in replacement** — Compatible with pip commands
3. **Built-in venv management** — `uv venv`, `uv pip install`
4. **Lockfile support** — Reproducible environments
5. **Modern** — Active development, good defaults
6. **Low overhead** — Single binary, no Python dependency for installation

### Commands we'll use:
```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Add new dependency
uv pip install <package>

# Sync from lockfile
uv pip sync requirements.txt
```

## Consequences

### Positive
- Fast iteration cycles
- Simple mental model
- Good CI/CD performance
- Modern tooling

### Negative
- Newer tool (less battle-tested)
- Team members need uv installed
- Some edge cases may differ from pip

## Implementation Notes
- Use `uv venv` for virtual environment
- Keep `requirements.txt` for compatibility
- Consider `pyproject.toml` for project metadata
