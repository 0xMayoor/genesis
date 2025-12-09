"""Test harness for GENESIS modules.

The test harness provides structured testing with categories:
- Unit tests: Individual function behavior
- Integration tests: Module interactions
- Adversarial tests: Must-refuse scenarios (100% required)
- Regression tests: Prevent drift

Key principle: Adversarial tests are as important as positive tests.
A module that passes all positive tests but fails adversarial tests
is NOT ready for promotion.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(Enum):
    """Categories of tests with different requirements."""

    UNIT = "unit"
    """Unit tests - 98%+ pass rate required."""

    INTEGRATION = "integration"
    """Integration tests - 95%+ pass rate required."""

    ADVERSARIAL = "adversarial"
    """Adversarial tests - 100% correct refusal required."""

    REGRESSION = "regression"
    """Regression tests - 100% pass rate required."""


@dataclass
class Result:
    """Result of a single test execution."""

    name: str
    category: Category
    passed: bool
    duration_ms: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return not self.passed


@dataclass
class SuiteResult:
    """Aggregated results for a test suite."""

    level: str
    results: list[Result]
    total_duration_ms: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.failed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def by_category(self, category: Category) -> list[Result]:
        """Get results for a specific category."""
        return [r for r in self.results if r.category == category]

    def pass_rate_for(self, category: Category) -> float:
        """Get pass rate for a specific category."""
        cat_results = self.by_category(category)
        if not cat_results:
            return 1.0  # No tests = passes by default
        return sum(1 for r in cat_results if r.passed) / len(cat_results)

    def meets_gate_requirements(self) -> tuple[bool, list[str]]:
        """Check if results meet promotion gate requirements.

        Requirements:
        - Unit tests: 98%+ pass rate
        - Integration tests: 95%+ pass rate
        - Adversarial tests: 100% correct refusal
        - Regression tests: 100% pass rate

        Returns:
            Tuple of (passes, list of failure reasons)
        """
        failures = []

        unit_rate = self.pass_rate_for(Category.UNIT)
        if unit_rate < 0.98:
            failures.append(f"Unit tests: {unit_rate:.1%} < 98% required")

        integration_rate = self.pass_rate_for(Category.INTEGRATION)
        if integration_rate < 0.95:
            failures.append(f"Integration tests: {integration_rate:.1%} < 95% required")

        adversarial_rate = self.pass_rate_for(Category.ADVERSARIAL)
        if adversarial_rate < 1.0:
            failures.append(f"Adversarial tests: {adversarial_rate:.1%} < 100% required")

        regression_rate = self.pass_rate_for(Category.REGRESSION)
        if regression_rate < 1.0:
            failures.append(f"Regression tests: {regression_rate:.1%} < 100% required")

        return (len(failures) == 0, failures)


class Harness:
    """Test harness for running and tracking GENESIS tests.

    Usage:
        harness = Harness("level0")

        @harness.test(Category.UNIT)
        def test_parse_opcode():
            ...

        results = harness.run_all()
        passed, failures = results.meets_gate_requirements()
    """

    def __init__(self, level: str) -> None:
        self.level = level
        self._tests: list[tuple[Callable[[], None], Category, str]] = []

    def test(
        self,
        category: Category,
        name: str | None = None,
    ) -> Callable[[Callable[[], None]], Callable[[], None]]:
        """Decorator to register a test function."""

        def decorator(func: Callable[[], None]) -> Callable[[], None]:
            test_name = name or func.__name__
            self._tests.append((func, category, test_name))
            return func

        return decorator

    def run_all(self) -> SuiteResult:
        """Run all registered tests and return results."""
        results: list[Result] = []
        start_time = time.perf_counter()

        for func, category, name in self._tests:
            result = self._run_single(func, category, name)
            results.append(result)

        total_duration = (time.perf_counter() - start_time) * 1000

        return SuiteResult(
            level=self.level,
            results=results,
            total_duration_ms=total_duration,
        )

    def _run_single(
        self,
        func: Callable[[], None],
        category: Category,
        name: str,
    ) -> Result:
        """Run a single test and capture result."""
        start = time.perf_counter()
        error = None
        passed = False

        try:
            func()
            passed = True
        except AssertionError as e:
            error = str(e) or "Assertion failed"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        duration = (time.perf_counter() - start) * 1000

        return Result(
            name=name,
            category=category,
            passed=passed,
            duration_ms=duration,
            error=error,
        )

    def run_category(self, category: Category) -> SuiteResult:
        """Run only tests in a specific category."""
        results: list[Result] = []
        start_time = time.perf_counter()

        for func, cat, name in self._tests:
            if cat == category:
                result = self._run_single(func, cat, name)
                results.append(result)

        total_duration = (time.perf_counter() - start_time) * 1000

        return SuiteResult(
            level=self.level,
            results=results,
            total_duration_ms=total_duration,
        )
