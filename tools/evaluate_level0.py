#!/usr/bin/env python3
"""Evaluate Level 0 trained model against gate requirements.

Gate Requirements:
- Accuracy >= 98%
- Adversarial refusal rate = 100%
- False positive rate < 1%
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(model_path: Path):
    """Load the trained LoRA model."""
    print(f"Loading model from {model_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the output part
    if "Output:" in response:
        return response.split("Output:")[-1].strip().split("\n")[0].strip()
    return response


def evaluate_dataset(model, tokenizer, device, dataset_path: Path):
    """Evaluate model on dataset."""
    print(f"\nEvaluating on {dataset_path}...")
    
    correct = 0
    total = 0
    adversarial_refused = 0
    adversarial_total = 0
    false_positives = 0
    invalid_total = 0
    
    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(samples)}")
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(samples)}...")
        
        hex_bytes = sample.get("raw_bytes", "")
        instructions = sample.get("instructions", [])
        # Get first instruction's mnemonic if available
        expected = instructions[0]["mnemonic"].lower() if instructions else ""
        is_valid = sample.get("is_valid", True)
        source = sample.get("source", "")
        is_adversarial = source == "adversarial" or not is_valid
        
        prompt = f"Disassemble: {hex_bytes}\nOutput:"
        response = generate_response(model, tokenizer, device, prompt).lower()
        
        if is_adversarial:
            adversarial_total += 1
            # Should refuse/indicate uncertainty
            if "unknown" in response or "invalid" in response or "uncertain" in response or response == "":
                adversarial_refused += 1
        elif not is_valid:
            invalid_total += 1
            # Should not claim it's valid
            if expected in response or (response and "unknown" not in response and "invalid" not in response):
                false_positives += 1
        else:
            total += 1
            # Check if response contains expected mnemonic
            if expected and expected in response:
                correct += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    adversarial_refusal_rate = adversarial_refused / adversarial_total if adversarial_total > 0 else 1.0
    false_positive_rate = false_positives / invalid_total if invalid_total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "adversarial_refusal_rate": adversarial_refusal_rate,
        "false_positive_rate": false_positive_rate,
        "total_valid": total,
        "correct": correct,
        "adversarial_total": adversarial_total,
        "adversarial_refused": adversarial_refused,
        "invalid_total": invalid_total,
        "false_positives": false_positives,
    }


def check_gate_requirements(metrics: dict) -> tuple[bool, list[str]]:
    """Check if metrics meet gate requirements."""
    failures = []
    
    if metrics["accuracy"] < 0.98:
        failures.append(f"Accuracy {metrics['accuracy']:.2%} < 98%")
    
    if metrics["adversarial_refusal_rate"] < 1.0:
        failures.append(f"Adversarial refusal {metrics['adversarial_refusal_rate']:.2%} < 100%")
    
    if metrics["false_positive_rate"] > 0.01:
        failures.append(f"False positive rate {metrics['false_positive_rate']:.2%} > 1%")
    
    return len(failures) == 0, failures


def main():
    model_path = Path("models/level0")
    dataset_path = Path("genesis_datasets/level0/train.jsonl")
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    # Quick sanity check
    print("\n--- Sanity Check ---")
    test_prompts = [
        "Disassemble: 90\nOutput:",  # NOP
        "Disassemble: c3\nOutput:",  # RET
        "Disassemble: cc\nOutput:",  # INT3
    ]
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, device, prompt)
        print(f"  {prompt.split(chr(10))[0]} -> {response}")
    
    # Full evaluation
    metrics = evaluate_dataset(model, tokenizer, device, dataset_path)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:                {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total_valid']})")
    print(f"Adversarial Refusal:     {metrics['adversarial_refusal_rate']:.2%} ({metrics['adversarial_refused']}/{metrics['adversarial_total']})")
    print(f"False Positive Rate:     {metrics['false_positive_rate']:.2%} ({metrics['false_positives']}/{metrics['invalid_total']})")
    
    # Check gates
    print("\n" + "="*50)
    print("GATE REQUIREMENTS")
    print("="*50)
    passes, failures = check_gate_requirements(metrics)
    
    if passes:
        print("✅ ALL GATES PASSED!")
        print("🚀 Level 0 complete. Ready for Level 1 (Assembly).")
    else:
        print("❌ GATES FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nConsider: more training epochs, larger dataset, or hyperparameter tuning.")
    
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
