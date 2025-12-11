#!/usr/bin/env python3
"""Verify Level 2 model against comprehensive exam."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_PATH = "models/level2_best"
BASE_MODEL = "distilgpt2"

# Test cases from exam
TEST_CASES = [
    # Linear code
    ("linear_simple", "Instructions:\n0x0:push rbp\n0x1:mov rbp,rsp\n0x2:ret [return]", 
     ["BB0", "return", "func"]),
    
    # Unconditional jump
    ("unconditional_jump", "Instructions:\n0x0:jmp 0x10 [jump->0x10]\n0x10:ret [return]",
     ["BB0", "BB1", "unconditional"]),
    
    # Simple if
    ("simple_if", "Instructions:\n0x0:cmp rax,0\n0x1:je 0x10 [conditional->0x10]\n0x2:mov rax,1\n0x3:ret [return]\n0x10:mov rax,0\n0x11:ret [return]",
     ["BB0", "BB1", "conditional"]),
    
    # If-else merge (diamond)
    ("if_else_merge", "Instructions:\n0x0:cmp rax,0\n0x1:je 0x10 [conditional->0x10]\n0x2:mov rax,1\n0x3:jmp 0x20 [jump->0x20]\n0x10:mov rax,2\n0x11:jmp 0x20 [jump->0x20]\n0x20:ret [return]",
     ["BB0", "BB1", "BB2", "edges"]),
    
    # While loop
    ("while_loop", "Instructions:\n0x0:cmp rcx,0\n0x1:je 0x20 [conditional->0x20]\n0x2:dec rcx\n0x3:jmp 0x0 [jump->0x0]\n0x20:ret [return]",
     ["BB0", "BB1", "loop"]),
    
    # Do-while loop  
    ("do_while", "Instructions:\n0x0:dec rcx\n0x1:cmp rcx,0\n0x2:jne 0x0 [conditional->0x0]\n0x3:ret [return]",
     ["BB0", "loop"]),
    
    # Function with call
    ("function_call", "Instructions:\n0x0:push rbp\n0x1:call 0x100 [call->0x100]\n0x2:pop rbp\n0x3:ret [return]\n0x100:mov rax,1\n0x101:ret [return]",
     ["BB0", "func", "0x100"]),
    
    # Nested if
    ("nested_if", "Instructions:\n0x0:cmp rax,0\n0x1:je 0x20 [conditional->0x20]\n0x2:cmp rbx,0\n0x3:je 0x10 [conditional->0x10]\n0x4:mov rcx,1\n0x5:jmp 0x20 [jump->0x20]\n0x10:mov rcx,2\n0x11:jmp 0x20 [jump->0x20]\n0x20:ret [return]",
     ["BB0", "BB1", "BB2", "conditional"]),
]

def main():
    print("Loading Level 2 model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Device: {device}\n")
    print("=" * 60)
    print("LEVEL 2 MODEL VERIFICATION")
    print("=" * 60)
    
    passed = 0
    failed = []
    
    for name, prompt, expected_tokens in TEST_CASES:
        full_prompt = f"{prompt}\nAnalysis:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = result.split("Analysis:")[-1].strip()
        
        # Check if expected tokens are in output
        all_found = all(tok.lower() in analysis.lower() for tok in expected_tokens)
        
        if all_found:
            passed += 1
            status = "✓"
        else:
            failed.append((name, expected_tokens, analysis[:60]))
            status = "✗"
        
        print(f"\n{status} {name}")
        print(f"  Output: {analysis[:80]}...")
    
    print("\n" + "=" * 60)
    pct = 100 * passed / len(TEST_CASES)
    print(f"RESULT: {passed}/{len(TEST_CASES)} ({pct:.0f}%)")
    
    if failed:
        print("\nFailed cases:")
        for name, expected, got in failed:
            print(f"  {name}: expected {expected}")
            print(f"    got: {got}...")
    
    if pct >= 90:
        print("\n🎉 Level 2 model PASSED!")
    else:
        print("\n⚠️  Model needs more training")
    
    return passed, len(TEST_CASES)

if __name__ == "__main__":
    main()
