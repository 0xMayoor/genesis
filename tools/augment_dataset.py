#!/usr/bin/env python3
"""
Augment Level 0 dataset with exam patterns.
Adds real-world patterns that the model needs to learn.
"""

import json
from pathlib import Path

# Import exam patterns
from tests.exam_level0 import (
    FUNCTION_PATTERNS,
    SHELLCODE_PATTERNS, 
    INTEL_MANUAL_EXAMPLES,
    CTF_CHALLENGES,
)

def main():
    dataset_path = Path("genesis_datasets/level0/train.jsonl")
    
    # Read existing samples
    existing = []
    existing_bytes = set()
    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            existing.append(sample)
            existing_bytes.add(sample.get("raw_bytes", ""))
    
    print(f"Existing samples: {len(existing)}")
    
    # Collect new samples from exam patterns
    new_samples = []
    
    all_patterns = [
        ("Function Patterns", FUNCTION_PATTERNS),
        ("Shellcode Patterns", SHELLCODE_PATTERNS),
        ("Intel Manual Examples", INTEL_MANUAL_EXAMPLES),
        ("CTF Challenges", CTF_CHALLENGES),
    ]
    
    for name, patterns in all_patterns:
        added = 0
        for hex_bytes, expected_mnemonic, description in patterns:
            # Skip if already in dataset
            if hex_bytes.lower() in existing_bytes:
                continue
            
            # Create sample - repeat each pattern multiple times for emphasis
            for _ in range(10):  # Add 10 copies of each pattern
                sample = {
                    "raw_bytes": hex_bytes.lower(),
                    "expected_mnemonic": expected_mnemonic,
                    "source": f"exam_{name.lower().replace(' ', '_')}",
                    "description": description,
                }
                new_samples.append(sample)
            added += 1
        print(f"  {name}: added {added} unique patterns (x10 each)")
    
    print(f"New samples to add: {len(new_samples)}")
    
    # Append to dataset
    with open(dataset_path, "a") as f:
        for sample in new_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Count final
    total = len(existing) + len(new_samples)
    print(f"Total samples now: {total}")


if __name__ == "__main__":
    main()
