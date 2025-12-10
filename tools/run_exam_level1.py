#!/usr/bin/env python3
"""
Level 1 Model Evaluation Script

Tests the trained Level 1 model against comprehensive exam cases.
Compares model output to deterministic Level1Module ground truth.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
import sys
sys.path.insert(0, ".")

from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly import Level1Module


def load_model(model_path: str):
    """Load the trained Level 1 model."""
    print(f"Loading model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully\n")
    return model, tokenizer, device


def get_model_prediction(model, tokenizer, device, mnemonic: str, operands: str) -> str:
    """Get model's prediction for an instruction."""
    prompt = f"Instruction: {mnemonic} {operands}\nEffects:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    effects = result.split("Effects:")[-1].strip() if "Effects:" in result else ""
    return effects


def get_ground_truth(module: Level1Module, mnemonic: str, operands: list[str]) -> dict:
    """Get ground truth from deterministic module."""
    instr = Instruction(
        offset=0,
        raw_bytes=b"\x90",
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )
    
    result = module.analyze(instr)
    
    return {
        "reads": [e.register for e in result.register_effects if e.operation.value == "read"],
        "writes": [e.register for e in result.register_effects if e.operation.value == "write"],
        "flags": [(e.flag, e.operation.value) for e in result.flag_effects],
        "flow": result.control_flow.flow_type.value if result.control_flow else "sequential",
        "is_certain": result.is_certain,
    }


def check_prediction(prediction: str, ground_truth: dict) -> tuple[bool, list[str]]:
    """Check if model prediction matches ground truth."""
    errors = []
    pred_lower = prediction.lower()
    
    # Check writes
    for reg in ground_truth["writes"]:
        if reg.lower() not in pred_lower:
            errors.append(f"missing write: {reg}")
    
    # Check reads
    for reg in ground_truth["reads"]:
        if reg.lower() not in pred_lower:
            errors.append(f"missing read: {reg}")
    
    # Check flow type
    flow = ground_truth["flow"]
    if flow != "sequential" and flow not in pred_lower:
        errors.append(f"missing flow: {flow}")
    
    # Check if uncertain cases are handled
    if not ground_truth["is_certain"] and "invalid" not in pred_lower:
        errors.append("should be INVALID")
    
    return len(errors) == 0, errors


# Test cases organized by category
EXAM_CASES = {
    "Data Movement": [
        ("mov", ["rax", "rbx"], "Basic register move"),
        ("mov", ["rax", "0x100"], "Move immediate"),
        ("mov", ["rax", "[rbx]"], "Move from memory"),
        ("mov", ["[rax]", "rbx"], "Move to memory"),
        ("lea", ["rax", "[rbx+rcx*4]"], "Load effective address"),
        ("push", ["rbp"], "Push register"),
        ("pop", ["rax"], "Pop register"),
        ("xchg", ["rax", "rbx"], "Exchange registers"),
    ],
    "Arithmetic": [
        ("add", ["rax", "rbx"], "Add registers"),
        ("sub", ["rax", "0x10"], "Subtract immediate"),
        ("inc", ["rcx"], "Increment"),
        ("dec", ["rdx"], "Decrement"),
        ("neg", ["rax"], "Negate"),
        ("mul", ["rbx"], "Unsigned multiply"),
        ("imul", ["rax", "rbx"], "Signed multiply"),
        ("xor", ["rax", "rax"], "XOR self (zero idiom)"),
    ],
    "Logic & Shifts": [
        ("and", ["rax", "0xff"], "AND immediate"),
        ("or", ["rax", "rbx"], "OR registers"),
        ("not", ["rax"], "Bitwise NOT"),
        ("shl", ["rax", "4"], "Shift left"),
        ("shr", ["rax", "1"], "Shift right"),
        ("test", ["rax", "rax"], "Test (sets flags)"),
        ("cmp", ["rax", "rbx"], "Compare (sets flags)"),
    ],
    "Control Flow": [
        ("jmp", ["0x1000"], "Unconditional jump"),
        ("je", ["0x2000"], "Jump if equal"),
        ("jne", ["0x3000"], "Jump if not equal"),
        ("jl", ["0x4000"], "Jump if less"),
        ("jg", ["0x5000"], "Jump if greater"),
        ("call", ["0x6000"], "Call subroutine"),
        ("ret", [], "Return"),
    ],
    "Stack Frame": [
        ("enter", ["16", "0"], "Create stack frame"),
        ("leave", [], "Destroy stack frame"),
        ("push", ["rbp"], "Save base pointer"),
        ("mov", ["rbp", "rsp"], "Setup frame pointer"),
    ],
    "Adversarial": [
        ("fakeinstr", ["rax"], "Invalid instruction"),
        ("mov", ["xyz", "abc"], "Invalid operands"),
        ("add", [], "Missing operands"),
    ],
}


def run_exam(model, tokenizer, device):
    """Run comprehensive exam on the model."""
    module = Level1Module()
    
    print("=" * 60)
    print("LEVEL 1 COMPREHENSIVE EXAM RESULTS")
    print("=" * 60)
    
    total_correct = 0
    total_cases = 0
    category_results = {}
    
    for category, cases in EXAM_CASES.items():
        correct = 0
        failures = []
        
        for mnemonic, operands, description in cases:
            total_cases += 1
            
            # Get ground truth
            ground_truth = get_ground_truth(module, mnemonic, operands)
            
            # Get model prediction
            ops_str = ", ".join(operands) if operands else ""
            prediction = get_model_prediction(model, tokenizer, device, mnemonic, ops_str)
            
            # Check correctness
            is_correct, errors = check_prediction(prediction, ground_truth)
            
            if is_correct:
                correct += 1
                total_correct += 1
            else:
                failures.append((mnemonic, operands, description, errors, prediction[:50]))
        
        # Report category results
        pct = 100 * correct / len(cases)
        status = "✓" if correct == len(cases) else "✗"
        print(f"\n{category}")
        print(f"  Score: {correct}/{len(cases)} ({pct:.1f}%) {status}")
        
        if failures:
            print("  Failures:")
            for mnem, ops, desc, errs, pred in failures[:3]:
                print(f"    - {mnem} {ops}: {', '.join(errs)}")
                print(f"      Got: {pred}...")
        
        category_results[category] = (correct, len(cases))
    
    # Overall results
    pct = 100 * total_correct / total_cases
    print("\n" + "=" * 60)
    print(f"OVERALL: {total_correct}/{total_cases} ({pct:.1f}%)")
    
    if pct >= 98:
        print("🎉 PASSED - Ready for Level 2!")
        return 0
    elif pct >= 90:
        print("⚠️  CLOSE - Minor improvements needed")
        return 1
    else:
        print("❌ FAILED - More training required")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Level 1 Model Evaluation")
    parser.add_argument("--model-path", default="models/level1", help="Path to trained model")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model_path)
    exit_code = run_exam(model, tokenizer, device)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
