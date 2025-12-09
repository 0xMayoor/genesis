"""Tests for training infrastructure."""

import pytest

from core.training.config import ModelConfig, TrainingConfig
from core.training.metrics import Metrics, compute_metrics


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self) -> None:
        """Default config is valid."""
        config = ModelConfig()
        assert config.use_lora is True
        assert config.lora_r == 8
        assert config.max_length == 512

    def test_estimated_params_with_lora(self) -> None:
        """LoRA reduces trainable parameters."""
        config = ModelConfig(use_lora=True, lora_r=8)
        params = config.estimated_trainable_params()

        # Should be much less than full model
        assert params < 10_000_000  # Less than 10M

    def test_estimated_params_without_lora(self) -> None:
        """Full fine-tuning has more parameters."""
        config = ModelConfig(use_lora=False)
        params = config.estimated_trainable_params()

        # Full model is ~125M
        assert params > 100_000_000


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self) -> None:
        """Default config is valid."""
        config = TrainingConfig()
        assert config.batch_size == 8
        assert config.fp16 is True
        assert config.gradient_checkpointing is True

    def test_effective_batch_size(self) -> None:
        """Effective batch size calculation."""
        config = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 32

    def test_memory_estimate_reasonable(self) -> None:
        """Memory estimate is reasonable for 16GB."""
        config = TrainingConfig()
        memory = config.estimated_memory_gb()

        # Should be under 16GB with default settings
        assert memory < 16

    def test_validation_warns_on_high_memory(self) -> None:
        """Validation warns when memory might be too high."""
        config = TrainingConfig(
            batch_size=32,
            model=ModelConfig(use_lora=False),
        )
        warnings = config.validate()

        # Should have warnings about memory
        assert len(warnings) > 0

    def test_validation_passes_for_default(self) -> None:
        """Default config should have minimal warnings."""
        config = TrainingConfig()
        warnings = config.validate()

        # Default should be safe
        assert len(warnings) == 0


class TestMetrics:
    """Tests for Metrics dataclass."""

    def test_default_metrics(self) -> None:
        """Default metrics are zero."""
        metrics = Metrics()
        assert metrics.accuracy == 0.0
        assert metrics.false_positive_rate == 0.0

    def test_meets_gate_perfect(self) -> None:
        """Perfect metrics pass gate."""
        metrics = Metrics(
            accuracy=1.0,
            adversarial_refusal_rate=1.0,
            false_positive_rate=0.0,
        )
        passes, failures = metrics.meets_gate_requirements()
        assert passes is True
        assert len(failures) == 0

    def test_meets_gate_low_accuracy(self) -> None:
        """Low accuracy fails gate."""
        metrics = Metrics(
            accuracy=0.90,
            adversarial_refusal_rate=1.0,
            false_positive_rate=0.0,
        )
        passes, failures = metrics.meets_gate_requirements()
        assert passes is False
        assert any("Accuracy" in f for f in failures)

    def test_meets_gate_adversarial_failure(self) -> None:
        """Missing adversarial refusals fails gate."""
        metrics = Metrics(
            accuracy=0.99,
            adversarial_refusal_rate=0.95,
            false_positive_rate=0.0,
        )
        passes, failures = metrics.meets_gate_requirements()
        assert passes is False
        assert any("Adversarial" in f for f in failures)

    def test_meets_gate_high_false_positive(self) -> None:
        """High false positive rate fails gate."""
        metrics = Metrics(
            accuracy=0.99,
            adversarial_refusal_rate=1.0,
            false_positive_rate=0.05,
        )
        passes, failures = metrics.meets_gate_requirements()
        assert passes is False
        assert any("False positive" in f for f in failures)

    def test_to_dict(self) -> None:
        """Metrics can be converted to dict."""
        metrics = Metrics(accuracy=0.95, loss=0.1)
        d = metrics.to_dict()

        assert d["accuracy"] == 0.95
        assert d["loss"] == 0.1
        assert "false_positive_rate" in d


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions give 100% accuracy."""
        predictions = [
            {"instructions": [{"mnemonic": "nop"}], "is_uncertain": False},
            {"instructions": [{"mnemonic": "ret"}], "is_uncertain": False},
        ]
        labels = [
            {"instructions": [{"mnemonic": "nop"}], "is_valid": True},
            {"instructions": [{"mnemonic": "ret"}], "is_valid": True},
        ]

        metrics = compute_metrics(predictions, labels)
        assert metrics.accuracy == 1.0
        assert metrics.mnemonic_accuracy == 1.0

    def test_wrong_predictions(self) -> None:
        """Wrong predictions reduce accuracy."""
        predictions = [
            {"instructions": [{"mnemonic": "nop"}], "is_uncertain": False},
            {"instructions": [{"mnemonic": "nop"}], "is_uncertain": False},  # Wrong
        ]
        labels = [
            {"instructions": [{"mnemonic": "nop"}], "is_valid": True},
            {"instructions": [{"mnemonic": "ret"}], "is_valid": True},
        ]

        metrics = compute_metrics(predictions, labels)
        assert metrics.accuracy < 1.0
        assert metrics.false_positive_rate > 0

    def test_correct_refusal(self) -> None:
        """Correct refusal on invalid input."""
        predictions = [
            {"instructions": [], "is_uncertain": True},
        ]
        labels = [
            {"instructions": [], "is_valid": False},
        ]

        metrics = compute_metrics(predictions, labels)
        assert metrics.accuracy == 1.0
        assert metrics.uncertainty_precision == 1.0

    def test_adversarial_tracking(self) -> None:
        """Adversarial samples are tracked separately."""
        predictions = [
            {"instructions": [], "is_uncertain": True},
            {"instructions": [], "is_uncertain": True},
            {"instructions": [{"mnemonic": "nop"}], "is_uncertain": False},
        ]
        labels = [
            {"instructions": [], "is_valid": False},
            {"instructions": [], "is_valid": False},
            {"instructions": [{"mnemonic": "nop"}], "is_valid": True},
        ]
        adversarial_mask = [True, True, False]

        metrics = compute_metrics(predictions, labels, adversarial_mask)
        assert metrics.adversarial_refusal_rate == 1.0

    def test_empty_inputs(self) -> None:
        """Empty inputs return default metrics."""
        metrics = compute_metrics([], [])
        assert metrics.accuracy == 0.0

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths raise error."""
        with pytest.raises(ValueError):
            compute_metrics([{}], [{}, {}])
