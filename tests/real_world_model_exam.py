#!/usr/bin/env python3
"""
Real-World MODEL Validation

Tests the TRAINED models (not deterministic modules) against real compiled binaries.
This is the true validation - can the neural network handle real-world code?
"""

import subprocess
import tempfile
import re
from pathlib import Path
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Real C programs
REAL_PROGRAMS = {
    "simple_return": """
int main() { return 42; }
""",
    "simple_if": """
int main() {
    int x = 5;
    if (x > 0) return 1;
    return 0;
}
""",
    "while_loop": """
int main() {
    int i = 0, sum = 0;
    while (i < 10) { sum += i; i++; }
    return sum;
}
""",
    "function_call": """
int helper(int x) { return x * 2; }
int main() { return helper(5); }
""",
}


def compile_and_disassemble(name: str, code: str, tmpdir: Path) -> dict:
    """Compile C code and get objdump disassembly."""
    src = tmpdir / f"{name}.c"
    bin_path = tmpdir / name
    
    src.write_text(code)
    result = subprocess.run(
        ["gcc", "-O0", "-o", str(bin_path), str(src)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return {"error": result.stderr}
    
    # Get disassembly of main
    disasm = subprocess.run(
        ["objdump", "-d", "--disassemble=main", str(bin_path)],
        capture_output=True, text=True
    )
    
    return {
        "path": bin_path,
        "disasm": disasm.stdout,
        "name": name,
    }


def parse_objdump_instructions(disasm: str) -> list[dict]:
    """Parse objdump output into instruction list."""
    instructions = []
    
    for line in disasm.split('\n'):
        # Match lines like: "  4004e7:	55                   	push   %rbp"
        match = re.match(r'\s+([0-9a-f]+):\s+([0-9a-f ]+)\s+(\w+)\s*(.*)', line)
        if match:
            addr, bytes_hex, mnemonic, operands = match.groups()
            instructions.append({
                "offset": int(addr, 16),
                "bytes": bytes_hex.strip(),
                "mnemonic": mnemonic,
                "operands": operands.strip(),
            })
    
    return instructions


def count_control_flow(instructions: list[dict]) -> dict:
    """Analyze control flow from objdump instructions."""
    conditionals = ['je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jz', 'jnz', 'js', 'jns']
    jumps = ['jmp']
    calls = ['call', 'callq']
    returns = ['ret', 'retq']
    
    stats = {
        "total": len(instructions),
        "conditionals": 0,
        "jumps": 0,
        "calls": 0,
        "returns": 0,
    }
    
    for instr in instructions:
        m = instr["mnemonic"]
        if m in conditionals:
            stats["conditionals"] += 1
        elif m in jumps:
            stats["jumps"] += 1
        elif m in calls:
            stats["calls"] += 1
        elif m in returns:
            stats["returns"] += 1
    
    return stats


def test_level0_model(instructions: list[dict], model, tokenizer, device) -> dict:
    """Test Level 0 model on real instructions."""
    correct = 0
    total = 0
    errors = []
    
    for instr in instructions[:10]:  # Test first 10
        # Format input like training data
        bytes_hex = instr["bytes"].replace(" ", "")
        prompt = f"Bytes: {bytes_hex}\nInstruction:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = result.split("Instruction:")[-1].strip().split()[0].lower() if "Instruction:" in result else ""
        expected = instr["mnemonic"].lower()
        
        if predicted == expected:
            correct += 1
        else:
            errors.append((bytes_hex, expected, predicted))
        total += 1
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
        "errors": errors[:3],  # First 3 errors
    }


def test_level1_model(instructions: list[dict], model, tokenizer, device) -> dict:
    """Test Level 1 model on real instructions."""
    correct = 0
    total = 0
    
    for instr in instructions[:5]:  # Test first 5
        # Format like training
        ops = instr["operands"].replace("%", "").replace(",", ", ")
        prompt = f"Instruction: {instr['mnemonic']} {ops}\nSemantics:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if output contains expected keywords
        m = instr["mnemonic"].lower()
        output = result.lower()
        
        if m in ['push', 'pop'] and 'stack' in output:
            correct += 1
        elif m in ['mov', 'lea'] and ('write' in output or 'read' in output or 'register' in output):
            correct += 1
        elif m in ['add', 'sub', 'imul'] and ('arithmetic' in output or 'flag' in output or 'result' in output):
            correct += 1
        elif m in ['cmp', 'test'] and 'flag' in output:
            correct += 1
        elif m in ['je', 'jne', 'jmp', 'jg', 'jl'] and ('jump' in output or 'branch' in output or 'control' in output):
            correct += 1
        elif m in ['call', 'ret'] and ('call' in output or 'return' in output or 'control' in output):
            correct += 1
        elif m == 'nop':
            correct += 1  # nop is always correct
        else:
            # At least check it produced something
            if len(result.split("Semantics:")[-1].strip()) > 10:
                correct += 0.5
        
        total += 1
    
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0,
    }


def test_level2_model(instructions: list[dict], cf_stats: dict, model, tokenizer, device) -> dict:
    """Test Level 2 model on real instruction sequence."""
    # Format instruction sequence
    instr_strs = []
    for i in instructions[:15]:  # First 15 instructions
        m = i["mnemonic"]
        ops = i["operands"].replace("%", "").replace(",", ", ")
        
        # Add control flow annotation
        cf = ""
        if m in ['jmp']:
            cf = " [jump]"
        elif m in ['je', 'jne', 'jg', 'jl', 'jge', 'jle']:
            cf = " [conditional]"
        elif m in ['call', 'callq']:
            cf = " [call]"
        elif m in ['ret', 'retq']:
            cf = " [return]"
        
        instr_strs.append(f"{i['offset']:#x}:{m} {ops}{cf}")
    
    prompt = "Instructions:\n" + "\n".join(instr_strs) + "\nAnalysis:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    analysis = result.split("Analysis:")[-1].strip().lower()
    
    # Check if output matches expected patterns
    checks = {
        "has_blocks": "bb" in analysis or "block" in analysis,
        "has_edges": "edge" in analysis or "->" in analysis,
        "mentions_functions": "func" in analysis,
        "mentions_loops": "loop" in analysis if cf_stats["conditionals"] > 0 else True,
    }
    
    score = sum(checks.values()) / len(checks)
    
    return {
        "checks": checks,
        "score": score,
        "output_preview": analysis[:100],
    }


def run_real_model_exam():
    """Run real-world model validation."""
    print("=" * 70)
    print("REAL-WORLD MODEL VALIDATION")
    print("Testing TRAINED MODELS on REAL compiled binaries")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    models = {}
    
    print("\n📦 Loading trained models...")
    
    for level, path in [("level0", "models/level0_best"), 
                        ("level1", "models/level1_best"),
                        ("level2", "models/level2_best")]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            base = AutoModelForCausalLM.from_pretrained("distilgpt2")
            model = PeftModel.from_pretrained(base, path)
            model.to(device)
            model.eval()
            models[level] = (model, tokenizer)
            print(f"  ✓ {level} model loaded")
        except Exception as e:
            print(f"  ✗ {level} model not found: {e}")
    
    if not models:
        print("\n❌ No models found!")
        return
    
    # Compile and test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        print("\n📦 Compiling real C programs...")
        binaries = {}
        for name, code in REAL_PROGRAMS.items():
            result = compile_and_disassemble(name, code, tmpdir)
            if "error" not in result:
                result["instructions"] = parse_objdump_instructions(result["disasm"])
                result["cf_stats"] = count_control_flow(result["instructions"])
                binaries[name] = result
                print(f"  ✓ {name}: {len(result['instructions'])} instructions")
        
        # Test each level
        results = {}
        
        if "level0" in models:
            print("\n🔬 Testing Level 0 Model (Bytes → Assembly)")
            print("-" * 50)
            model, tokenizer = models["level0"]
            
            l0_results = []
            for name, binary in binaries.items():
                r = test_level0_model(binary["instructions"], model, tokenizer, device)
                l0_results.append(r)
                status = "✓" if r["accuracy"] >= 0.8 else "⚠" if r["accuracy"] >= 0.5 else "✗"
                print(f"  {status} {name}: {r['accuracy']*100:.0f}% ({r['correct']}/{r['total']})")
                if r["errors"]:
                    for b, exp, got in r["errors"][:1]:
                        print(f"      Error: {b} → expected '{exp}', got '{got}'")
            
            avg = sum(r["accuracy"] for r in l0_results) / len(l0_results)
            results["level0"] = avg
            print(f"\n  Level 0 Average: {avg*100:.1f}%")
        
        if "level1" in models:
            print("\n🔬 Testing Level 1 Model (Assembly → Semantics)")
            print("-" * 50)
            model, tokenizer = models["level1"]
            
            l1_results = []
            for name, binary in binaries.items():
                r = test_level1_model(binary["instructions"], model, tokenizer, device)
                l1_results.append(r)
                status = "✓" if r["accuracy"] >= 0.6 else "⚠" if r["accuracy"] >= 0.3 else "✗"
                print(f"  {status} {name}: {r['accuracy']*100:.0f}% ({r['correct']}/{r['total']})")
            
            avg = sum(r["accuracy"] for r in l1_results) / len(l1_results)
            results["level1"] = avg
            print(f"\n  Level 1 Average: {avg*100:.1f}%")
        
        if "level2" in models:
            print("\n🔬 Testing Level 2 Model (Instructions → CFG)")
            print("-" * 50)
            model, tokenizer = models["level2"]
            
            l2_results = []
            for name, binary in binaries.items():
                r = test_level2_model(binary["instructions"], binary["cf_stats"], model, tokenizer, device)
                l2_results.append(r)
                status = "✓" if r["score"] >= 0.75 else "⚠" if r["score"] >= 0.5 else "✗"
                checks_str = ", ".join(k for k, v in r["checks"].items() if v)
                print(f"  {status} {name}: {r['score']*100:.0f}% [{checks_str}]")
            
            avg = sum(r["score"] for r in l2_results) / len(l2_results)
            results["level2"] = avg
            print(f"\n  Level 2 Average: {avg*100:.1f}%")
        
        # Final summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS - REAL WORLD VALIDATION")
        print("=" * 70)
        
        for level, acc in results.items():
            status = "✅" if acc >= 0.8 else "⚠️" if acc >= 0.5 else "❌"
            print(f"  {status} {level}: {acc*100:.1f}%")
        
        overall = sum(results.values()) / len(results) if results else 0
        print(f"\n  Overall: {overall*100:.1f}%")
        
        if overall >= 0.75:
            print("\n🎉 Models validated on real-world binaries!")
        else:
            print("\n⚠️  Models may need improvement for real-world use")


if __name__ == "__main__":
    run_real_model_exam()
