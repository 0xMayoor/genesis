"""Training metrics for GENESIS models.

Defines metrics for evaluating model performance,
with special attention to zero-hallucination requirements.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Metrics:
    """Metrics for a training/evaluation run."""

    # Basic metrics
    accuracy: float = 0.0
    """Overall accuracy."""

    loss: float = 0.0
    """Loss value."""

    # Per-category metrics
    mnemonic_accuracy: float = 0.0
    """Accuracy on mnemonic prediction."""

    operand_accuracy: float = 0.0
    """Accuracy on operand prediction."""

    boundary_accuracy: float = 0.0
    """Accuracy on instruction boundary detection."""

    # Zero-hallucination metrics
    uncertainty_precision: float = 0.0
    """Precision: when model says uncertain, is it correct to be uncertain?"""

    uncertainty_recall: float = 0.0
    """Recall: does model catch all cases that should be uncertain?"""

    false_positive_rate: float = 0.0
    """Rate of confident wrong answers (critical for zero-hallucination)."""

    refusal_rate: float = 0.0
    """Rate of refusals/uncertain outputs."""

    # Adversarial metrics
    adversarial_refusal_rate: float = 0.0
    """Rate of correct refusals on adversarial inputs (must be 100%)."""

    # Additional details
    details: dict[str, Any] = field(default_factory=dict)

    def meets_gate_requirements(self, level: int = 0) -> tuple[bool, list[str]]:
        """Check if metrics meet promotion gate requirements.

        Requirements:
        - Accuracy >= 98% on valid inputs
        - Adversarial refusal rate = 100%
        - False positive rate < 1%
        """
        failures = []

        if self.accuracy < 0.98:
            failures.append(f"Accuracy {self.accuracy:.1%} < 98% required")

        if self.adversarial_refusal_rate < 1.0:
            failures.append(
                f"Adversarial refusal {self.adversarial_refusal_rate:.1%} < 100% required"
            )

        if self.false_positive_rate > 0.01:
            failures.append(f"False positive rate {self.false_positive_rate:.1%} > 1% allowed")

        return (len(failures) == 0, failures)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "accuracy": self.accuracy,
            "loss": self.loss,
            "mnemonic_accuracy": self.mnemonic_accuracy,
            "operand_accuracy": self.operand_accuracy,
            "boundary_accuracy": self.boundary_accuracy,
            "uncertainty_precision": self.uncertainty_precision,
            "uncertainty_recall": self.uncertainty_recall,
            "false_positive_rate": self.false_positive_rate,
            "refusal_rate": self.refusal_rate,
            "adversarial_refusal_rate": self.adversarial_refusal_rate,
        }


def compute_metrics(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    adversarial_mask: list[bool] | None = None,
) -> Metrics:
    """Compute metrics from predictions and labels.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        adversarial_mask: Boolean mask indicating adversarial samples

    Returns:
        Computed metrics
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return Metrics()

    # Initialize counters
    total = len(predictions)
    correct = 0
    mnemonic_correct = 0
    mnemonic_total = 0

    # Zero-hallucination counters
    uncertain_correct = 0  # Model uncertain AND should be uncertain
    uncertain_total = 0  # Model said uncertain
    should_uncertain = 0  # Should be uncertain
    false_positives = 0  # Confident but wrong
    refusals = 0

    # Adversarial counters
    adversarial_total = 0
    adversarial_refused = 0

    for i, (pred, label) in enumerate(zip(predictions, labels, strict=False)):
        is_adversarial = adversarial_mask[i] if adversarial_mask else False

        pred_uncertain = pred.get("is_uncertain", False)
        label_valid = label.get("is_valid", True)

        if is_adversarial:
            adversarial_total += 1
            if pred_uncertain:
                adversarial_refused += 1

        if pred_uncertain:
            refusals += 1
            uncertain_total += 1
            if not label_valid:
                uncertain_correct += 1

        if not label_valid:
            should_uncertain += 1

        # Check correctness
        if label_valid and not pred_uncertain:
            pred_insns = pred.get("instructions", [])
            label_insns = label.get("instructions", [])

            if len(pred_insns) == len(label_insns):
                all_match = True
                for p_insn, l_insn in zip(pred_insns, label_insns, strict=False):
                    mnemonic_total += 1
                    if p_insn.get("mnemonic") == l_insn.get("mnemonic"):
                        mnemonic_correct += 1
                    else:
                        all_match = False

                if all_match:
                    correct += 1
                else:
                    false_positives += 1
            else:
                false_positives += 1
        elif not label_valid and pred_uncertain:
            correct += 1  # Correct refusal

    # Compute rates
    accuracy = correct / total if total > 0 else 0.0
    mnemonic_accuracy = mnemonic_correct / mnemonic_total if mnemonic_total > 0 else 0.0

    uncertainty_precision = uncertain_correct / uncertain_total if uncertain_total > 0 else 1.0
    uncertainty_recall = uncertain_correct / should_uncertain if should_uncertain > 0 else 1.0

    false_positive_rate = false_positives / total if total > 0 else 0.0
    refusal_rate = refusals / total if total > 0 else 0.0

    adversarial_refusal_rate = (
        adversarial_refused / adversarial_total if adversarial_total > 0 else 1.0
    )

    return Metrics(
        accuracy=accuracy,
        mnemonic_accuracy=mnemonic_accuracy,
        uncertainty_precision=uncertainty_precision,
        uncertainty_recall=uncertainty_recall,
        false_positive_rate=false_positive_rate,
        refusal_rate=refusal_rate,
        adversarial_refusal_rate=adversarial_refusal_rate,
    )
