#!/usr/bin/env python3
"""Train Level 0 model.

Usage:
    python tools/train_level0.py [--config CONFIG_PATH] [--data DATA_PATH]

This script trains the Level 0 (Machine Code Patterns) model using
LoRA fine-tuning on a small base model.

Requirements:
- GPU recommended but not required (will be slow on CPU)
- ~8GB VRAM for default settings
- Training data at genesis_datasets/level0/train.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training import ModelConfig, TrainingConfig, train_level0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GENESIS Level 0 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("genesis_datasets/level0/train.jsonl"),
        help="Path to training data (JSONL format)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/level0"),
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Base model to fine-tune (default: distilgpt2)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (reduce if OOM)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )

    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )

    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without training",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Check data exists
    if not args.data.exists():
        logger.error(f"Training data not found: {args.data}")
        logger.info(
            "Generate data first with: python genesis_datasets/generators/level0_generator.py"
        )
        return 1

    # Build config
    model_config = ModelConfig(
        model_name=args.model,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
    )

    config = TrainingConfig(
        output_dir=args.output,
        dataset_path=args.data,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        fp16=not args.fp32,
        model=model_config,
    )

    # Validate
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model.model_name}")
    logger.info(f"  LoRA: {config.model.use_lora} (r={config.model.lora_r})")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Effective batch: {config.effective_batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  FP16: {config.fp16}")
    logger.info(f"  Estimated memory: {config.estimated_memory_gb():.1f} GB")

    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)

    if args.dry_run:
        logger.info("Dry run complete. Config is valid.")
        return 0

    # Train
    logger.info("Starting training...")

    try:
        metrics = train_level0(config, args.data)

        logger.info("Training complete!")
        logger.info(f"Final metrics: {metrics.to_dict()}")

        # Check gate requirements
        passes, failures = metrics.meets_gate_requirements()
        if passes:
            logger.info("✅ Model PASSES gate requirements!")
        else:
            logger.warning("❌ Model FAILS gate requirements:")
            for failure in failures:
                logger.warning(f"  - {failure}")

        return 0 if passes else 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
