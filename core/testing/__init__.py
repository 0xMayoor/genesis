"""GENESIS Testing Framework.

This module provides the test harness and utilities for testing
all GENESIS components with a focus on zero-hallucination verification.
"""

from core.testing.harness import Category, Harness, Result, SuiteResult

__all__ = ["Harness", "Result", "SuiteResult", "Category"]
