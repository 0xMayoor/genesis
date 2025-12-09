"""Training configuration for GENESIS models.

Defines configuration dataclasses for training runs.
Optimized for 16GB RAM constraint.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the base model."""

    # Model selection
    model_name: str = "microsoft/codebert-base"
    """Base model to fine-tune. Should be small (< 300M params)."""

    # Model architecture
    max_length: int = 512
    """Maximum sequence length."""

    hidden_size: int = 768
    """Hidden size (from base model)."""

    # LoRA configuration
    use_lora: bool = True
    """Whether to use LoRA for efficient fine-tuning."""

    lora_r: int = 8
    """LoRA rank. Lower = fewer parameters, less capacity."""

    lora_alpha: int = 16
    """LoRA alpha. Scaling factor."""

    lora_dropout: float = 0.1
    """Dropout for LoRA layers."""

    lora_target_modules: list[str] | None = None
    """Which modules to apply LoRA to. None = auto-detect based on model."""

    def get_lora_target_modules(self) -> list[str]:
        """Get LoRA target modules based on model architecture."""
        if self.lora_target_modules is not None:
            return self.lora_target_modules
        
        # Auto-detect based on model name
        model_lower = self.model_name.lower()
        if "gpt2" in model_lower or "gpt-2" in model_lower:
            return ["c_attn", "c_proj"]
        elif "bert" in model_lower:
            return ["query", "value"]
        elif "llama" in model_lower or "mistral" in model_lower:
            return ["q_proj", "v_proj"]
        elif "t5" in model_lower:
            return ["q", "v"]
        else:
            # Default fallback - common attention modules
            return ["q_proj", "v_proj", "query", "value"]

    def estimated_trainable_params(self) -> int:
        """Estimate trainable parameters with LoRA."""
        if not self.use_lora:
            return 125_000_000  # Full model (rough estimate)

        # LoRA params ≈ 2 * r * hidden_size * num_modules
        target_modules = self.get_lora_target_modules()
        num_modules = len(target_modules) * 12  # 12 layers typical
        return 2 * self.lora_r * self.hidden_size * num_modules


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    # Paths
    output_dir: Path = Path("models/checkpoints")
    """Where to save checkpoints."""

    dataset_path: Path = Path("genesis_datasets/level0/train.jsonl")
    """Path to training data."""

    # Training hyperparameters
    batch_size: int = 8
    """Batch size. Keep small for 16GB RAM."""

    gradient_accumulation_steps: int = 4
    """Accumulate gradients to simulate larger batch."""

    learning_rate: float = 2e-5
    """Learning rate."""

    num_epochs: int = 3
    """Number of training epochs."""

    warmup_ratio: float = 0.1
    """Warmup ratio for learning rate scheduler."""

    weight_decay: float = 0.01
    """Weight decay for regularization."""

    # Memory optimization
    fp16: bool = True
    """Use mixed precision training."""

    gradient_checkpointing: bool = True
    """Trade compute for memory."""

    # Evaluation
    eval_steps: int = 500
    """Evaluate every N steps."""

    save_steps: int = 1000
    """Save checkpoint every N steps."""

    # Logging
    logging_steps: int = 100
    """Log every N steps."""

    # Model config
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def estimated_memory_gb(self) -> float:
        """Rough estimate of GPU memory needed."""
        # Very rough: ~4 bytes per param, plus activations
        params = self.model.estimated_trainable_params()
        base_memory = params * 4 / 1e9  # GB

        # Activations scale with batch size and sequence length
        activation_memory = (
            self.batch_size * self.model.max_length * self.model.hidden_size * 4 / 1e9
        )

        # Optimizer states (Adam: 2x params)
        optimizer_memory = params * 8 / 1e9

        total = base_memory + activation_memory + optimizer_memory

        if self.fp16:
            total *= 0.6  # Rough reduction from mixed precision

        if self.gradient_checkpointing:
            total *= 0.7  # Rough reduction from checkpointing

        return total

    def validate(self) -> list[str]:
        """Validate configuration and return warnings."""
        warnings = []

        memory = self.estimated_memory_gb()
        if memory > 14:
            warnings.append(
                f"Estimated memory ({memory:.1f}GB) may exceed 16GB limit. "
                "Consider reducing batch_size or max_length."
            )

        if self.batch_size > 16:
            warnings.append("Large batch_size may cause OOM. Consider using gradient_accumulation.")

        if not self.model.use_lora:
            warnings.append("Full fine-tuning without LoRA may exceed memory limits.")

        return warnings
