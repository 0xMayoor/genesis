"""Tests for the testing harness itself.

Meta-tests to ensure our test framework works correctly.
"""

import pytest

from core.testing.harness import Category, Harness, Result, SuiteResult


class TestResult:
    """Tests for Result dataclass."""

    def test_passed_result(self) -> None:
        """Passed test result."""
        result = Result(
            name="test_something",
            category=Category.UNIT,
            passed=True,
            duration_ms=10.5,
        )
        assert result.passed is True
        assert result.failed is False
        assert result.error is None

    def test_failed_result(self) -> None:
        """Failed test result."""
        result = Result(
            name="test_something",
            category=Category.UNIT,
            passed=False,
            duration_ms=5.0,
            error="Expected 1, got 2",
        )
        assert result.passed is False
        assert result.failed is True
        assert result.error == "Expected 1, got 2"


class TestSuiteResult:
    """Tests for SuiteResult aggregation."""

    def test_empty_suite(self) -> None:
        """Empty suite should have 100% pass rate."""
        suite = SuiteResult(
            level="level0",
            results=[],
            total_duration_ms=0.0,
        )
        assert suite.total == 0
        assert suite.passed == 0
        assert suite.failed == 0
        assert suite.pass_rate == 0.0

    def test_all_passed(self) -> None:
        """All tests passed."""
        results = [
            Result("test1", Category.UNIT, True, 1.0),
            Result("test2", Category.UNIT, True, 1.0),
            Result("test3", Category.UNIT, True, 1.0),
        ]
        suite = SuiteResult("level0", results, 3.0)
        assert suite.total == 3
        assert suite.passed == 3
        assert suite.failed == 0
        assert suite.pass_rate == 1.0

    def test_some_failed(self) -> None:
        """Some tests failed."""
        results = [
            Result("test1", Category.UNIT, True, 1.0),
            Result("test2", Category.UNIT, False, 1.0, "Error"),
            Result("test3", Category.UNIT, True, 1.0),
        ]
        suite = SuiteResult("level0", results, 3.0)
        assert suite.total == 3
        assert suite.passed == 2
        assert suite.failed == 1
        assert suite.pass_rate == pytest.approx(0.666, rel=0.01)

    def test_by_category(self) -> None:
        """Filter results by category."""
        results = [
            Result("unit1", Category.UNIT, True, 1.0),
            Result("unit2", Category.UNIT, True, 1.0),
            Result("adv1", Category.ADVERSARIAL, True, 1.0),
        ]
        suite = SuiteResult("level0", results, 3.0)

        unit_results = suite.by_category(Category.UNIT)
        assert len(unit_results) == 2

        adv_results = suite.by_category(Category.ADVERSARIAL)
        assert len(adv_results) == 1

    def test_pass_rate_for_category(self) -> None:
        """Pass rate for specific category."""
        results = [
            Result("unit1", Category.UNIT, True, 1.0),
            Result("unit2", Category.UNIT, False, 1.0, "Error"),
            Result("adv1", Category.ADVERSARIAL, True, 1.0),
        ]
        suite = SuiteResult("level0", results, 3.0)

        assert suite.pass_rate_for(Category.UNIT) == 0.5
        assert suite.pass_rate_for(Category.ADVERSARIAL) == 1.0


class TestGateRequirements:
    """Tests for promotion gate requirements."""

    def test_meets_all_requirements(self) -> None:
        """Suite that meets all gate requirements."""
        results = [
            # 100% unit tests (need 98%)
            *[Result(f"unit{i}", Category.UNIT, True, 1.0) for i in range(100)],
            # 100% integration tests (need 95%)
            *[Result(f"int{i}", Category.INTEGRATION, True, 1.0) for i in range(20)],
            # 100% adversarial tests (need 100%)
            *[Result(f"adv{i}", Category.ADVERSARIAL, True, 1.0) for i in range(10)],
            # 100% regression tests (need 100%)
            *[Result(f"reg{i}", Category.REGRESSION, True, 1.0) for i in range(5)],
        ]
        suite = SuiteResult("level0", results, 100.0)

        passes, failures = suite.meets_gate_requirements()
        assert passes is True
        assert failures == []

    def test_fails_unit_requirement(self) -> None:
        """Suite that fails unit test requirement (< 98%)."""
        results = [
            # 95% unit tests - below 98% threshold
            *[Result(f"unit{i}", Category.UNIT, True, 1.0) for i in range(95)],
            *[Result(f"unit_fail{i}", Category.UNIT, False, 1.0, "Error") for i in range(5)],
        ]
        suite = SuiteResult("level0", results, 100.0)

        passes, failures = suite.meets_gate_requirements()
        assert passes is False
        assert any("Unit tests" in f and "98%" in f for f in failures)

    def test_fails_adversarial_requirement(self) -> None:
        """Suite that fails adversarial test requirement (< 100%)."""
        results = [
            # All unit tests pass
            *[Result(f"unit{i}", Category.UNIT, True, 1.0) for i in range(100)],
            # 90% adversarial - below 100% threshold
            *[Result(f"adv{i}", Category.ADVERSARIAL, True, 1.0) for i in range(9)],
            Result("adv_fail", Category.ADVERSARIAL, False, 1.0, "Should have refused"),
        ]
        suite = SuiteResult("level0", results, 100.0)

        passes, failures = suite.meets_gate_requirements()
        assert passes is False
        assert any("Adversarial tests" in f and "100%" in f for f in failures)


class TestHarness:
    """Tests for Harness class."""

    def test_register_and_run_test(self) -> None:
        """Register and run a simple test."""
        harness = Harness("test_level")

        @harness.test(Category.UNIT)
        def test_simple() -> None:
            assert 1 + 1 == 2

        results = harness.run_all()
        assert results.total == 1
        assert results.passed == 1

    def test_failing_test(self) -> None:
        """Failing test is captured correctly."""
        harness = Harness("test_level")

        @harness.test(Category.UNIT)
        def test_fails() -> None:
            assert 1 == 2, "One does not equal two"

        results = harness.run_all()
        assert results.total == 1
        assert results.failed == 1
        assert "One does not equal two" in results.results[0].error

    def test_exception_in_test(self) -> None:
        """Exception in test is captured."""
        harness = Harness("test_level")

        @harness.test(Category.UNIT)
        def test_raises() -> None:
            raise ValueError("Something went wrong")

        results = harness.run_all()
        assert results.total == 1
        assert results.failed == 1
        assert "ValueError" in results.results[0].error

    def test_run_specific_category(self) -> None:
        """Run only tests in a specific category."""
        harness = Harness("test_level")

        @harness.test(Category.UNIT)
        def test_unit() -> None:
            pass

        @harness.test(Category.ADVERSARIAL)
        def test_adversarial() -> None:
            pass

        unit_results = harness.run_category(Category.UNIT)
        assert unit_results.total == 1

        adv_results = harness.run_category(Category.ADVERSARIAL)
        assert adv_results.total == 1

    def test_custom_test_name(self) -> None:
        """Test with custom name."""
        harness = Harness("test_level")

        @harness.test(Category.UNIT, name="my_custom_test_name")
        def some_function() -> None:
            pass

        results = harness.run_all()
        assert results.results[0].name == "my_custom_test_name"
