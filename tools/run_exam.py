#!/usr/bin/env python3
"""
Run Level 0 comprehensive exam with trained model.

Usage:
    python tools/run_exam.py [--model-path models/level0]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from tests.exam_level0 import run_exam_with_model, print_exam_results


def load_model(model_path: str):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model + LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model, tokenizer, device


def create_inference_fn(model, tokenizer, device):
    """Create inference function for exam"""
    
    def inference(hex_bytes: str) -> str:
        """Run inference on hex bytes, return predicted mnemonic"""
        # Handle adversarial cases
        if not hex_bytes or not hex_bytes.strip():
            return "INVALID"
        
        hex_bytes = hex_bytes.strip().lower()
        
        # Check for invalid hex
        try:
            # Try to parse as hex
            if any(c not in '0123456789abcdef' for c in hex_bytes):
                return "INVALID"
        except:
            return "INVALID"
        
        # Check for suspiciously long input
        if len(hex_bytes) > 30:
            return "INVALID"
        
        prompt = f"Disassemble: {hex_bytes}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract mnemonic from response
        if "Output:" in response:
            result = response.split("Output:")[-1].strip().split()[0]
            return result
        
        return "UNKNOWN"
    
    return inference


def main():
    parser = argparse.ArgumentParser(description="Run Level 0 comprehensive exam")
    parser.add_argument(
        "--model-path",
        default="models/level0",
        help="Path to trained model"
    )
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    # Create inference function
    inference_fn = create_inference_fn(model, tokenizer, device)
    
    # Run exam
    print("\nRunning comprehensive exam...")
    results = run_exam_with_model(inference_fn)
    
    # Print results
    overall_pct = print_exam_results(results)
    
    # Exit code based on pass/fail
    sys.exit(0 if overall_pct >= 98 else 1)


if __name__ == "__main__":
    main()
