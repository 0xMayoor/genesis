"""Training loop for GENESIS models.

Implements the training pipeline with:
- LoRA fine-tuning for memory efficiency
- Mixed precision training
- Gradient checkpointing
- Evaluation with zero-hallucination metrics
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from core.training.config import TrainingConfig
from core.training.metrics import Metrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class Level0TrainingSample:
    """A single training sample for Level 0."""

    input_text: str
    target_text: str
    is_valid: bool
    is_adversarial: bool


class Level0Dataset(Dataset):
    """Dataset for Level 0 training."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[Level0TrainingSample] = []

        self._load_data(data_path)

    def _load_data(self, data_path: Path) -> None:
        """Load and preprocess data from JSONL file."""
        with open(data_path) as f:
            for line in f:
                data = json.loads(line)

                # Format input: hex bytes
                input_text = f"Disassemble x86_64: {data['raw_bytes']}"

                # Format output: instruction sequence or refusal
                if data["is_valid"]:
                    instructions = data["instructions"]
                    output_parts = []
                    for insn in instructions:
                        operands = ", ".join(insn["operands"]) if insn["operands"] else ""
                        if operands:
                            output_parts.append(f"{insn['mnemonic']} {operands}")
                        else:
                            output_parts.append(insn["mnemonic"])
                    target_text = "; ".join(output_parts)
                else:
                    target_text = "[UNCERTAIN: invalid or ambiguous input]"

                self.samples.append(
                    Level0TrainingSample(
                        input_text=input_text,
                        target_text=target_text,
                        is_valid=data["is_valid"],
                        is_adversarial=data["source"] == "adversarial",
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # Tokenize input
        inputs = self.tokenizer(
            sample.input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        targets = self.tokenizer(
            sample.target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
            "is_valid": sample.is_valid,
            "is_adversarial": sample.is_adversarial,
        }


class Level0Trainer:
    """Trainer for Level 0 model.

    Uses a sequence-to-sequence approach:
    - Input: hex bytes
    - Output: disassembled instructions or uncertainty marker
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        # Validate config
        warnings = config.validate()
        for warning in warnings:
            logger.warning(warning)

        # Initialize model and tokenizer
        self._init_model()

    def _init_model(self) -> None:
        """Initialize model with LoRA if configured."""
        model_name = self.config.model.model_name

        logger.info(f"Loading model: {model_name}")

        # For Level 0, we'll use a smaller model suitable for sequence generation
        # Using a causal LM for simplicity
        from transformers import AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )

        if self.config.model.use_lora:
            logger.info("Applying LoRA configuration")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                target_modules=self.config.model.lora_target_modules,
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)

    def train(self, train_path: Path, eval_path: Path | None = None) -> Metrics:
        """Run training loop.

        Args:
            train_path: Path to training data
            eval_path: Optional path to evaluation data

        Returns:
            Final evaluation metrics
        """
        logger.info("Loading training data...")
        train_dataset = Level0Dataset(
            train_path,
            self.tokenizer,
            self.config.model.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

        global_step = 0
        best_metrics = Metrics()

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss / self.config.gradient_accumulation_steps
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()

                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                        f"Step {global_step}, Loss: {avg_loss:.4f}"
                    )

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

            # Evaluation
            if eval_path:
                metrics = self.evaluate(eval_path)
                logger.info(f"Evaluation metrics: {metrics.to_dict()}")

                if metrics.accuracy > best_metrics.accuracy:
                    best_metrics = metrics
                    self._save_checkpoint(global_step, is_best=True)

        # Final save
        self._save_checkpoint(global_step, is_final=True)

        return best_metrics

    def evaluate(self, eval_path: Path) -> Metrics:
        """Evaluate model on dataset.

        Args:
            eval_path: Path to evaluation data

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        eval_dataset = Level0Dataset(
            eval_path,
            self.tokenizer,
            self.config.model.max_length,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        predictions = []
        labels = []
        adversarial_mask = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Generate predictions
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                )

                # Decode predictions
                for i, output in enumerate(outputs):
                    pred_text = self.tokenizer.decode(output, skip_special_tokens=True)

                    # Parse prediction
                    is_uncertain = "[UNCERTAIN" in pred_text
                    pred_instructions = self._parse_output(pred_text) if not is_uncertain else []

                    predictions.append(
                        {
                            "instructions": pred_instructions,
                            "is_uncertain": is_uncertain,
                        }
                    )

                    labels.append(
                        {
                            "instructions": eval_dataset.samples[len(labels)].target_text,
                            "is_valid": batch["is_valid"][i].item(),
                        }
                    )

                    adversarial_mask.append(batch["is_adversarial"][i].item())

        return compute_metrics(predictions, labels, adversarial_mask)

    def _parse_output(self, text: str) -> list[dict[str, Any]]:
        """Parse model output into instruction list."""
        # Simple parsing - split by semicolon
        instructions = []
        parts = text.split(";")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Split mnemonic and operands
            tokens = part.split(None, 1)
            if tokens:
                mnemonic = tokens[0]
                operands = tokens[1].split(", ") if len(tokens) > 1 else []
                instructions.append(
                    {
                        "mnemonic": mnemonic,
                        "operands": operands,
                    }
                )

        return instructions

    def _save_checkpoint(
        self,
        step: int,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            save_path = output_dir / "best"
        elif is_final:
            save_path = output_dir / "final"
        else:
            save_path = output_dir / f"checkpoint-{step}"

        logger.info(f"Saving checkpoint to {save_path}")

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


def train_level0(
    config: TrainingConfig | None = None,
    train_path: Path | None = None,
) -> Metrics:
    """Convenience function to train Level 0 model.

    Args:
        config: Training configuration (uses defaults if None)
        train_path: Path to training data

    Returns:
        Final metrics
    """
    if config is None:
        config = TrainingConfig()

    if train_path is None:
        train_path = Path("genesis_datasets/level0/train.jsonl")

    trainer = Level0Trainer(config)
    return trainer.train(train_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from core.training.config import ModelConfig

    # Use a small model for testing
    config = TrainingConfig(
        model=ModelConfig(
            model_name="distilgpt2",  # Small model for testing
            max_length=256,
        ),
        batch_size=4,
        num_epochs=1,
        logging_steps=10,
    )

    metrics = train_level0(config)
    print(f"Training complete. Metrics: {metrics.to_dict()}")
