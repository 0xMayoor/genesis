#!/usr/bin/env python3
"""
Level 2 Dataset Generator

Generates training data for control flow analysis:
- Basic block detection
- CFG construction
- Function boundaries
- Loop detection
"""

import json
import random
from pathlib import Path
from dataclasses import asdict
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from levels.level0_machine.types import Instruction, InstructionCategory
from levels.level1_assembly.types import (
    Level1Output,
    ControlFlowEffect,
    ControlFlowType,
)
from levels.level2_ir import (
    Level2Module,
    Level2Input,
    Level2Output,
)


# ============================================================================
# Instruction Templates
# ============================================================================

def make_instr(
    offset: int,
    mnemonic: str,
    operands: list[str] = None,
    cf_type: ControlFlowType = ControlFlowType.SEQUENTIAL,
    target: str = None,
    condition: str = None
) -> Level1Output:
    """Create a Level1Output."""
    operands = operands or []
    instr = Instruction(
        offset=offset,
        raw_bytes=b"\x90",
        mnemonic=mnemonic,
        operands=tuple(operands),
        size=1,
        category=InstructionCategory.DATA_TRANSFER,
    )
    
    cf = None
    if cf_type != ControlFlowType.SEQUENTIAL or target:
        cf = ControlFlowEffect(
            type=cf_type,
            target_expr=target,
            condition=condition,
        )
    
    return Level1Output(
        instruction=instr,
        register_effects=[],
        memory_effects=[],
        flag_effects=[],
        control_flow=cf,
        is_uncertain=False,
    )


# ============================================================================
# Pattern Templates
# ============================================================================

def generate_linear_function(base_offset: int, length: int) -> list[Level1Output]:
    """Generate a linear function with no branches."""
    regs = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi"]
    ops = ["mov", "add", "sub", "xor", "and", "or"]
    
    instructions = [
        make_instr(base_offset, "push", ["rbp"]),
        make_instr(base_offset + 1, "mov", ["rbp", "rsp"]),
    ]
    
    for i in range(length - 4):
        op = random.choice(ops)
        r1, r2 = random.sample(regs, 2)
        instructions.append(make_instr(base_offset + 2 + i, op, [r1, r2]))
    
    instructions.extend([
        make_instr(base_offset + length - 2, "pop", ["rbp"]),
        make_instr(base_offset + length - 1, "ret", [], cf_type=ControlFlowType.RETURN),
    ])
    
    return instructions


def generate_if_pattern(base_offset: int) -> list[Level1Output]:
    """Generate if-else pattern."""
    then_target = base_offset + 0x10
    merge_target = base_offset + 0x20
    
    return [
        make_instr(base_offset, "cmp", ["rax", "0"]),
        make_instr(base_offset + 1, "je", [hex(then_target)], 
                   cf_type=ControlFlowType.CONDITIONAL, target=hex(then_target)),
        # Else block
        make_instr(base_offset + 2, "mov", ["rbx", "1"]),
        make_instr(base_offset + 3, "jmp", [hex(merge_target)],
                   cf_type=ControlFlowType.JUMP, target=hex(merge_target)),
        # Then block
        make_instr(then_target, "mov", ["rbx", "0"]),
        make_instr(then_target + 1, "jmp", [hex(merge_target)],
                   cf_type=ControlFlowType.JUMP, target=hex(merge_target)),
        # Merge
        make_instr(merge_target, "ret", [], cf_type=ControlFlowType.RETURN),
    ]


def generate_while_loop(base_offset: int, body_size: int = 3) -> list[Level1Output]:
    """Generate while loop pattern."""
    exit_target = base_offset + 0x20
    
    instructions = [
        # Header
        make_instr(base_offset, "cmp", ["rcx", "0"]),
        make_instr(base_offset + 1, "je", [hex(exit_target)],
                   cf_type=ControlFlowType.CONDITIONAL, target=hex(exit_target)),
    ]
    
    # Body
    for i in range(body_size):
        instructions.append(make_instr(base_offset + 2 + i, "dec", ["rcx"]))
    
    # Back edge
    instructions.append(make_instr(base_offset + 2 + body_size, "jmp", [hex(base_offset)],
                                   cf_type=ControlFlowType.JUMP, target=hex(base_offset)))
    
    # Exit
    instructions.append(make_instr(exit_target, "ret", [], cf_type=ControlFlowType.RETURN))
    
    return instructions


def generate_do_while_loop(base_offset: int, body_size: int = 2) -> list[Level1Output]:
    """Generate do-while loop pattern."""
    instructions = []
    
    # Body
    for i in range(body_size):
        instructions.append(make_instr(base_offset + i, "dec", ["rcx"]))
    
    # Condition and back edge
    instructions.extend([
        make_instr(base_offset + body_size, "cmp", ["rcx", "0"]),
        make_instr(base_offset + body_size + 1, "jne", [hex(base_offset)],
                   cf_type=ControlFlowType.CONDITIONAL, target=hex(base_offset)),
        make_instr(base_offset + body_size + 2, "ret", [], cf_type=ControlFlowType.RETURN),
    ])
    
    return instructions


def generate_nested_loops(base_offset: int) -> list[Level1Output]:
    """Generate nested loop pattern."""
    inner_header = base_offset + 0x10
    outer_continue = base_offset + 0x20
    exit_target = base_offset + 0x30
    
    return [
        # Outer header
        make_instr(base_offset, "cmp", ["rcx", "0"]),
        make_instr(base_offset + 1, "je", [hex(exit_target)],
                   cf_type=ControlFlowType.CONDITIONAL, target=hex(exit_target)),
        make_instr(base_offset + 2, "jmp", [hex(inner_header)],
                   cf_type=ControlFlowType.JUMP, target=hex(inner_header)),
        # Inner header
        make_instr(inner_header, "cmp", ["rdx", "0"]),
        make_instr(inner_header + 1, "je", [hex(outer_continue)],
                   cf_type=ControlFlowType.CONDITIONAL, target=hex(outer_continue)),
        # Inner body
        make_instr(inner_header + 2, "dec", ["rdx"]),
        make_instr(inner_header + 3, "jmp", [hex(inner_header)],
                   cf_type=ControlFlowType.JUMP, target=hex(inner_header)),
        # Outer continue
        make_instr(outer_continue, "dec", ["rcx"]),
        make_instr(outer_continue + 1, "jmp", [hex(base_offset)],
                   cf_type=ControlFlowType.JUMP, target=hex(base_offset)),
        # Exit
        make_instr(exit_target, "ret", [], cf_type=ControlFlowType.RETURN),
    ]


def generate_function_call(base_offset: int, callee_offset: int) -> list[Level1Output]:
    """Generate function with call."""
    return_point = base_offset + 0x10
    
    return [
        # Caller
        make_instr(base_offset, "push", ["rbp"]),
        make_instr(base_offset + 1, "mov", ["rdi", "1"]),
        make_instr(base_offset + 2, "call", [hex(callee_offset)],
                   cf_type=ControlFlowType.CALL, target=hex(callee_offset)),
        make_instr(base_offset + 3, "pop", ["rbp"]),
        make_instr(base_offset + 4, "ret", [], cf_type=ControlFlowType.RETURN),
        # Callee
        make_instr(callee_offset, "mov", ["rax", "rdi"]),
        make_instr(callee_offset + 1, "ret", [], cf_type=ControlFlowType.RETURN),
    ]


def generate_switch_pattern(base_offset: int, num_cases: int = 3) -> list[Level1Output]:
    """Generate switch-like pattern."""
    instructions = []
    case_targets = []
    merge_target = base_offset + 0x100
    
    offset = base_offset
    
    # Generate case comparisons
    for i in range(num_cases):
        case_target = base_offset + 0x20 + i * 0x10
        case_targets.append(case_target)
        instructions.extend([
            make_instr(offset, "cmp", ["rax", str(i)]),
            make_instr(offset + 1, "je", [hex(case_target)],
                       cf_type=ControlFlowType.CONDITIONAL, target=hex(case_target)),
        ])
        offset += 2
    
    # Default case
    instructions.extend([
        make_instr(offset, "mov", ["rbx", "-1"]),
        make_instr(offset + 1, "jmp", [hex(merge_target)],
                   cf_type=ControlFlowType.JUMP, target=hex(merge_target)),
    ])
    
    # Case blocks
    for i, target in enumerate(case_targets):
        instructions.extend([
            make_instr(target, "mov", ["rbx", str(i)]),
            make_instr(target + 1, "jmp", [hex(merge_target)],
                       cf_type=ControlFlowType.JUMP, target=hex(merge_target)),
        ])
    
    # Merge
    instructions.append(make_instr(merge_target, "ret", [], cf_type=ControlFlowType.RETURN))
    
    return instructions


# ============================================================================
# Sample Generation
# ============================================================================

def instruction_to_dict(instr: Level1Output) -> dict:
    """Convert instruction to serializable dict."""
    return {
        "offset": instr.instruction.offset,
        "mnemonic": instr.instruction.mnemonic,
        "operands": list(instr.instruction.operands),
        "control_flow": {
            "type": instr.control_flow.type.value,
            "target": instr.control_flow.target_expr,
            "condition": instr.control_flow.condition,
        } if instr.control_flow else None,
    }


def output_to_dict(output: Level2Output) -> dict:
    """Convert Level2Output to serializable dict."""
    return {
        "basic_blocks": [
            {
                "id": b.id,
                "start": b.start_offset,
                "end": b.end_offset,
                "entry_type": b.entry_type.value,
                "exit_type": b.exit_type.value,
                "size": b.size,
            }
            for b in output.basic_blocks
        ],
        "cfg_edges": [
            {
                "source": e.source_block,
                "target": e.target_block,
                "type": e.edge_type.value,
            }
            for e in output.cfg_edges
        ],
        "functions": [
            {
                "entry": f.entry_offset,
                "blocks": f.blocks,
                "exits": f.exit_blocks,
            }
            for f in output.functions
        ],
        "loops": [
            {
                "header": l.header_block,
                "back_edge": l.back_edge_block,
                "body": l.body_blocks,
                "type": l.loop_type.value,
            }
            for l in output.loops
        ],
    }


def generate_sample(module: Level2Module, instructions: list[Level1Output], pattern_name: str) -> dict:
    """Generate a single training sample."""
    input_data = Level2Input(instructions=instructions, entry_point=instructions[0].instruction.offset)
    output = module.analyze(input_data)
    
    return {
        "pattern": pattern_name,
        "instructions": [instruction_to_dict(i) for i in instructions],
        "entry_point": input_data.entry_point,
        "output": output_to_dict(output),
    }


def generate_dataset(num_samples: int = 5000) -> list[dict]:
    """Generate complete training dataset."""
    module = Level2Module()
    samples = []
    
    patterns = [
        ("linear_short", lambda: generate_linear_function(0, random.randint(4, 6))),
        ("linear_medium", lambda: generate_linear_function(0, random.randint(7, 12))),
        ("linear_long", lambda: generate_linear_function(0, random.randint(13, 20))),
        ("if_else", lambda: generate_if_pattern(0)),
        ("while_loop_small", lambda: generate_while_loop(0, random.randint(1, 3))),
        ("while_loop_large", lambda: generate_while_loop(0, random.randint(4, 8))),
        ("do_while", lambda: generate_do_while_loop(0, random.randint(1, 4))),
        ("nested_loops", lambda: generate_nested_loops(0)),
        ("function_call", lambda: generate_function_call(0, 0x100)),
        ("switch_2", lambda: generate_switch_pattern(0, 2)),
        ("switch_3", lambda: generate_switch_pattern(0, 3)),
        ("switch_4", lambda: generate_switch_pattern(0, 4)),
    ]
    
    # Generate samples for each pattern
    samples_per_pattern = num_samples // len(patterns)
    
    for pattern_name, generator in patterns:
        for _ in range(samples_per_pattern):
            try:
                instructions = generator()
                sample = generate_sample(module, instructions, pattern_name)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to generate {pattern_name}: {e}")
    
    # Fill remaining with random patterns
    while len(samples) < num_samples:
        pattern_name, generator = random.choice(patterns)
        try:
            instructions = generator()
            sample = generate_sample(module, instructions, pattern_name)
            samples.append(sample)
        except Exception:
            pass
    
    random.shuffle(samples)
    return samples


def save_dataset(samples: list[dict], output_path: Path):
    """Save dataset to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def print_stats(samples: list[dict]):
    """Print dataset statistics."""
    patterns = {}
    for s in samples:
        p = s["pattern"]
        patterns[p] = patterns.get(p, 0) + 1
    
    print("\n" + "=" * 50)
    print("LEVEL 2 DATASET STATISTICS")
    print("=" * 50)
    print(f"\nTotal samples: {len(samples)}")
    print(f"\nPatterns:")
    for pattern, count in sorted(patterns.items()):
        print(f"  {pattern}: {count}")


def main():
    print("Generating Level 2 dataset...")
    
    samples = generate_dataset(5000)
    
    output_path = Path(__file__).parent.parent / "level2" / "train.jsonl"
    save_dataset(samples, output_path)
    
    print(f"Dataset saved to {output_path}")
    print_stats(samples)


if __name__ == "__main__":
    main()
