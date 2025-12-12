#!/usr/bin/env python3
"""
STRICT REAL-WORLD GATE TEST

This is the ONLY test that matters.
- NO synthetic data
- REAL compiled binaries only
- Multiple compilers (gcc, clang)
- Multiple optimization levels (-O0 to -O3)
- Both AT&T and Intel syntax
- MUST pass 90%+ to proceed to next level

If this test fails, the model is NOT ready.
"""

import subprocess
import tempfile
import re
import random
from pathlib import Path
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# Test Programs - NEVER used in training
# ============================================================================

# These are DIFFERENT from training programs
GATE_TEST_PROGRAMS = {
    "reverse_string": """
void reverse(char* s, int len) {
    for (int i = 0; i < len/2; i++) {
        char t = s[i];
        s[i] = s[len-1-i];
        s[len-1-i] = t;
    }
}
int main() { char s[] = "hello"; reverse(s, 5); return s[0]; }
""",
    "count_ones": """
int popcount(unsigned int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
int main() { return popcount(0xFF); }
""",
    "array_max": """
int find_max(int* arr, int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max) max = arr[i];
    return max;
}
int main() { int a[] = {3,1,4,1,5,9,2,6}; return find_max(a, 8); }
""",
    "is_palindrome": """
int is_palindrome(char* s, int len) {
    for (int i = 0; i < len/2; i++)
        if (s[i] != s[len-1-i]) return 0;
    return 1;
}
int main() { return is_palindrome("radar", 5); }
""",
    "merge_sort_step": """
void merge(int* arr, int l, int m, int r) {
    int i = l, j = m + 1;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) i++;
        else {
            int t = arr[j];
            for (int k = j; k > i; k--) arr[k] = arr[k-1];
            arr[i] = t;
            i++; m++; j++;
        }
    }
}
int main() { int a[] = {1,3,2,4}; merge(a,0,1,3); return a[1]; }
""",
    "linked_math": """
int chain(int x) {
    x = x * 2 + 1;
    x = x - 3;
    x = x / 2;
    x = x ^ 0xFF;
    return x;
}
int main() { return chain(10); }
""",
    "bit_reverse": """
unsigned int reverse_bits(unsigned int n) {
    unsigned int r = 0;
    for (int i = 0; i < 32; i++) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}
int main() { return reverse_bits(0x12345678) & 0xFF; }
""",
    "collatz": """
int collatz_steps(int n) {
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) n /= 2;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}
int main() { return collatz_steps(27); }
""",
}


@dataclass
class TestBinary:
    name: str
    compiler: str
    opt_level: str
    instructions_intel: list[dict]
    instructions_att: list[dict]


def compile_test_program(name: str, code: str, compiler: str, opt: str, tmpdir: Path) -> TestBinary:
    """Compile test program."""
    src = tmpdir / f"{name}.c"
    bin_path = tmpdir / f"{name}_{compiler}_{opt}"
    
    src.write_text(code)
    
    result = subprocess.run(
        [compiler, opt, "-o", str(bin_path), str(src)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return None
    
    def parse_disasm(output):
        instructions = []
        for line in output.split('\n'):
            match = re.match(r'\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S+)\s*(.*)', line)
            if match:
                addr, bytes_hex, mnemonic, operands = match.groups()
                instructions.append({
                    "offset": int(addr, 16),
                    "bytes": bytes_hex.strip().replace(" ", ""),
                    "mnemonic": mnemonic.strip(),
                    "operands": operands.strip(),
                })
        return instructions
    
    intel = subprocess.run(
        ["objdump", "-d", "-M", "intel", str(bin_path)],
        capture_output=True, text=True
    ).stdout
    
    att = subprocess.run(
        ["objdump", "-d", str(bin_path)],
        capture_output=True, text=True
    ).stdout
    
    return TestBinary(
        name=name,
        compiler=compiler,
        opt_level=opt,
        instructions_intel=parse_disasm(intel),
        instructions_att=parse_disasm(att),
    )


def test_level0_strict(model, tokenizer, device, binaries: list[TestBinary]) -> dict:
    """Strict Level 0 test - bytes to mnemonic."""
    results = {"correct": 0, "total": 0, "errors": []}
    
    for binary in binaries:
        # Test both syntaxes
        for syntax, instructions in [("intel", binary.instructions_intel), 
                                      ("att", binary.instructions_att)]:
            # Sample random instructions
            sample = random.sample(instructions, min(10, len(instructions)))
            
            for instr in sample:
                prompt = f"Bytes: {instr['bytes']}\nInstruction:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted = result.split("Instruction:")[-1].strip().split()[0].lower() if "Instruction:" in result else ""
                expected = instr["mnemonic"].lower()
                
                # Normalize (movl -> mov, etc.)
                expected_norm = re.sub(r'[lqwb]$', '', expected)
                predicted_norm = re.sub(r'[lqwb]$', '', predicted)
                
                results["total"] += 1
                if predicted_norm == expected_norm or predicted == expected:
                    results["correct"] += 1
                else:
                    if len(results["errors"]) < 10:
                        results["errors"].append({
                            "bytes": instr["bytes"],
                            "expected": expected,
                            "predicted": predicted,
                            "binary": f"{binary.name}_{binary.compiler}_{binary.opt_level}",
                        })
    
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results


def test_level1_strict(model, tokenizer, device, binaries: list[TestBinary]) -> dict:
    """Strict Level 1 test - instruction semantics."""
    results = {"correct": 0, "total": 0, "errors": []}
    
    # Expected semantics mapping
    semantic_keywords = {
        "push": ["stack", "push", "write", "rsp", "sp"],
        "pop": ["stack", "pop", "read", "rsp", "sp"],
        "mov": ["move", "write", "read", "register", "transfer", "copy"],
        "lea": ["address", "load", "effective", "lea"],
        "add": ["add", "sum", "arithmetic", "plus", "flag"],
        "sub": ["sub", "subtract", "minus", "arithmetic", "flag"],
        "cmp": ["compare", "flag", "cmp", "sub"],
        "test": ["test", "flag", "and", "compare"],
        "jmp": ["jump", "branch", "unconditional", "control"],
        "je": ["jump", "equal", "zero", "conditional", "branch"],
        "jne": ["jump", "not", "equal", "conditional", "branch"],
        "call": ["call", "function", "push", "return"],
        "ret": ["return", "pop", "rip", "control"],
        "xor": ["xor", "exclusive", "zero", "bitwise"],
        "and": ["and", "bitwise", "mask", "flag"],
        "or": ["or", "bitwise", "flag"],
        "shl": ["shift", "left", "multiply", "bitwise"],
        "shr": ["shift", "right", "divide", "bitwise"],
        "imul": ["multiply", "mul", "signed", "arithmetic"],
        "idiv": ["divide", "div", "signed", "arithmetic"],
    }
    
    for binary in binaries:
        for syntax, instructions in [("intel", binary.instructions_intel), 
                                      ("att", binary.instructions_att)]:
            sample = random.sample(instructions, min(5, len(instructions)))
            
            for instr in sample:
                mnemonic = instr["mnemonic"].lower()
                mnemonic_base = re.sub(r'[lqwb]$', '', mnemonic)
                
                if mnemonic_base not in semantic_keywords:
                    continue
                
                prompt = f"Instruction: {instr['mnemonic']} {instr['operands']}\nSemantics:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                output = result.split("Semantics:")[-1].lower() if "Semantics:" in result else result.lower()
                
                # Check if any expected keyword is present
                expected_kws = semantic_keywords.get(mnemonic_base, [])
                found = any(kw in output for kw in expected_kws)
                
                results["total"] += 1
                if found:
                    results["correct"] += 1
                else:
                    if len(results["errors"]) < 10:
                        results["errors"].append({
                            "instruction": f"{instr['mnemonic']} {instr['operands']}",
                            "expected_keywords": expected_kws,
                            "got": output[:100],
                        })
    
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results


def test_level2_strict(model, tokenizer, device, binaries: list[TestBinary]) -> dict:
    """Strict Level 2 test - CFG analysis."""
    results = {"correct": 0, "total": 0, "errors": []}
    
    for binary in binaries:
        for syntax, instructions in [("intel", binary.instructions_intel)]:  # Intel only for L2
            if len(instructions) < 5:
                continue
            
            # Analyze ground truth
            has_loop = False
            has_conditional = False
            has_call = False
            
            addresses = [i["offset"] for i in instructions]
            
            for instr in instructions:
                m = instr["mnemonic"].lower()
                ops = instr["operands"]
                
                if m.startswith("j") and m not in ["jmp", "jmpq"]:
                    has_conditional = True
                elif m in ["jmp", "jmpq"]:
                    target_match = re.search(r'([0-9a-f]+)', ops)
                    if target_match:
                        target = int(target_match.group(1), 16)
                        if target < instr["offset"]:
                            has_loop = True
                elif m in ["call", "callq"]:
                    has_call = True
            
            # Format input
            instr_strs = []
            for i in instructions[:15]:
                m = i["mnemonic"].lower()
                cf = ""
                if m.startswith("j"):
                    cf = " [jump]" if m in ["jmp", "jmpq"] else " [conditional]"
                elif m in ["call", "callq"]:
                    cf = " [call]"
                elif m in ["ret", "retq"]:
                    cf = " [return]"
                instr_strs.append(f"{i['offset']:#x}:{i['mnemonic']} {i['operands']}{cf}")
            
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
            analysis = result.split("Analysis:")[-1].lower() if "Analysis:" in result else result.lower()
            
            # Score based on matching ground truth
            score = 0
            checks = 0
            
            # Must mention blocks
            if "bb" in analysis or "block" in analysis:
                score += 1
            checks += 1
            
            # Loop detection
            if has_loop:
                if "loop" in analysis:
                    score += 1
                checks += 1
            
            # Conditional detection
            if has_conditional:
                if "conditional" in analysis or "branch" in analysis or "edge" in analysis:
                    score += 1
                checks += 1
            
            results["total"] += checks
            results["correct"] += score
    
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    return results


def run_strict_gate():
    """Run the strict real-world gate test."""
    print("=" * 70)
    print("STRICT REAL-WORLD GATE TEST")
    print("=" * 70)
    print("This test uses ONLY real compiled binaries")
    print("Test programs are NEVER used in training")
    print("MUST pass 90%+ to proceed")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Check compilers
    compilers = []
    for c in ["gcc", "clang"]:
        try:
            subprocess.run([c, "--version"], capture_output=True, check=True)
            compilers.append(c)
        except:
            pass
    
    print(f"Compilers: {compilers}")
    opt_levels = ["-O0", "-O1", "-O2", "-O3"]
    
    # Compile test binaries
    print("\n📦 Compiling test binaries (NOT used in training)...")
    binaries = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        for name, code in GATE_TEST_PROGRAMS.items():
            for compiler in compilers:
                for opt in opt_levels:
                    binary = compile_test_program(name, code, compiler, opt, tmpdir)
                    if binary and len(binary.instructions_intel) > 5:
                        binaries.append(binary)
        
        print(f"  Compiled {len(binaries)} test binaries")
        
        # Load and test each model
        results = {}
        
        for level, path in [("level0", "models/level0_ultimate"),
                            ("level1", "models/level1_v5"),
                            ("level2", "models/level2_v5")]:
            print(f"\n🔬 Testing {level.upper()} Model")
            print("-" * 50)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                base = AutoModelForCausalLM.from_pretrained("distilgpt2")
                model = PeftModel.from_pretrained(base, path)
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"  ❌ Could not load model: {e}")
                results[level] = {"accuracy": 0, "status": "NOT LOADED"}
                continue
            
            if level == "level0":
                r = test_level0_strict(model, tokenizer, device, binaries)
            elif level == "level1":
                r = test_level1_strict(model, tokenizer, device, binaries)
            else:
                r = test_level2_strict(model, tokenizer, device, binaries)
            
            acc = r["accuracy"]
            status = "✅ PASS" if acc >= 0.9 else "⚠️ MARGINAL" if acc >= 0.7 else "❌ FAIL"
            
            print(f"  Accuracy: {acc*100:.1f}% ({r['correct']}/{r['total']})")
            print(f"  Status: {status}")
            
            if r.get("errors"):
                print(f"  Sample errors:")
                for err in r["errors"][:3]:
                    print(f"    - {err}")
            
            results[level] = {"accuracy": acc, "status": status, "details": r}
        
        # Final verdict
        print("\n" + "=" * 70)
        print("GATE TEST RESULTS")
        print("=" * 70)
        
        all_pass = True
        for level, r in results.items():
            acc = r["accuracy"]
            status = r["status"]
            icon = "✅" if acc >= 0.9 else "⚠️" if acc >= 0.7 else "❌"
            print(f"  {icon} {level}: {acc*100:.1f}% - {status}")
            if acc < 0.9:
                all_pass = False
        
        print()
        if all_pass:
            print("🎉 ALL MODELS PASSED THE GATE!")
            print("   Ready for next level development")
        else:
            print("❌ MODELS DID NOT PASS THE GATE")
            print("   Need retraining with real binary data")
        
        return results


if __name__ == "__main__":
    run_strict_gate()
