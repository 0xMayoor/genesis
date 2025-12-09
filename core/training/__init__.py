"""GENESIS Training Infrastructure.

Provides training utilities for all levels:
- Dataset loading and preprocessing
- Model configuration
- LoRA fine-tuning setup
- Evaluation metrics
- Checkpointing
"""

from core.training.config import ModelConfig, TrainingConfig
from core.training.metrics import Metrics, compute_metrics
from core.training.trainer import Level0Trainer, train_level0

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "Metrics",
    "compute_metrics",
    "Level0Trainer",
    "train_level0",
]
