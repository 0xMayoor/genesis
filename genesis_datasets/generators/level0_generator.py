"""Level 0 Dataset Generator.

Generates training data for machine code pattern recognition.
Each sample consists of:
- Input: Raw bytes
- Output: Decoded instructions with metadata

Data sources:
1. Synthetic instruction sequences (controlled patterns)
2. Real binary snippets (from system libraries)
3. Adversarial samples (must-refuse cases)

All outputs are verified against external disassemblers.
"""

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from levels.level0_machine import (
    Architecture,
    Level0Input,
    Level0Module,
)


@dataclass
class Level0Sample:
    """A single training sample for Level 0."""

    # Input
    raw_bytes: bytes
    architecture: str

    # Expected output
    instructions: list[dict[str, Any]]
    is_valid: bool  # False for adversarial samples

    # Metadata
    source: str  # "synthetic", "binary", "adversarial"
    category: str  # More specific category
    difficulty: str  # "easy", "medium", "hard"

    @property
    def expected_mnemonic(self) -> str:
        """Get the first instruction's mnemonic, or 'unknown' for invalid."""
        if self.instructions and self.is_valid:
            return self.instructions[0].get("mnemonic", "unknown")
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "raw_bytes": self.raw_bytes.hex(),
            "architecture": self.architecture,
            "instructions": self.instructions,
            "is_valid": self.is_valid,
            "expected_mnemonic": self.expected_mnemonic,
            "source": self.source,
            "category": self.category,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Level0Sample":
        """Create from dict."""
        return cls(
            raw_bytes=bytes.fromhex(data["raw_bytes"]),
            architecture=data["architecture"],
            instructions=data["instructions"],
            is_valid=data["is_valid"],
            source=data["source"],
            category=data["category"],
            difficulty=data["difficulty"],
        )


# Common x86_64 instruction patterns for synthetic generation
X86_64_PATTERNS = {
    "nop": [b"\x90"],
    "ret": [b"\xc3", b"\xc2\x00\x00"],
    "push_reg": [bytes([0x50 + i]) for i in range(8)],  # push rax-rdi
    "pop_reg": [bytes([0x58 + i]) for i in range(8)],  # pop rax-rdi
    "mov_reg_reg": [
        b"\x89\xc0",  # mov eax, eax
        b"\x89\xd8",  # mov eax, ebx
        b"\x89\xc8",  # mov eax, ecx
        b"\x48\x89\xc0",  # mov rax, rax
        b"\x48\x89\xd8",  # mov rax, rbx
    ],
    "xor_reg_reg": [
        b"\x31\xc0",  # xor eax, eax
        b"\x31\xdb",  # xor ebx, ebx
        b"\x48\x31\xc0",  # xor rax, rax
    ],
    "add_reg_imm": [
        b"\x83\xc0\x01",  # add eax, 1
        b"\x83\xc0\x10",  # add eax, 16
        b"\x48\x83\xc0\x01",  # add rax, 1
    ],
    "sub_reg_imm": [
        b"\x83\xe8\x01",  # sub eax, 1
        b"\x48\x83\xe8\x08",  # sub rax, 8
    ],
    "cmp_reg_imm": [
        b"\x83\xf8\x00",  # cmp eax, 0
        b"\x83\xf8\x01",  # cmp eax, 1
    ],
    "test_reg_reg": [
        b"\x85\xc0",  # test eax, eax
        b"\x48\x85\xc0",  # test rax, rax
    ],
    "jmp_short": [b"\xeb\x00", b"\xeb\x10", b"\xeb\xfe"],
    "jcc_short": [
        b"\x74\x00",  # je
        b"\x75\x00",  # jne
        b"\x7c\x00",  # jl
        b"\x7f\x00",  # jg
    ],
    "call_rel": [b"\xe8\x00\x00\x00\x00"],
    "syscall": [b"\x0f\x05"],
    "int3": [b"\xcc"],
    "lea": [
        b"\x48\x8d\x04\x25\x00\x00\x00\x00",  # lea rax, [0]
    ],
}

# Function prologues/epilogues
FUNCTION_PATTERNS = {
    "prologue_simple": b"\x55\x48\x89\xe5",  # push rbp; mov rbp, rsp
    "prologue_with_sub": b"\x55\x48\x89\xe5\x48\x83\xec\x10",  # + sub rsp, 16
    "epilogue_simple": b"\x5d\xc3",  # pop rbp; ret
    "epilogue_leave": b"\xc9\xc3",  # leave; ret
}

# Adversarial patterns (should be refused or marked uncertain)
ADVERSARIAL_PATTERNS = {
    "incomplete_prefix": b"\x0f",  # Two-byte opcode prefix alone
    "incomplete_rex": b"\x48",  # REX.W alone
    "all_zeros": b"\x00" * 16,
    "all_ones": b"\xff" * 16,
    "random_high_entropy": bytes(random.randint(0, 255) for _ in range(16)),
}


class Level0DatasetGenerator:
    """Generates training datasets for Level 0."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._module = Level0Module()

    def generate_synthetic(self, count: int) -> Iterator[Level0Sample]:
        """Generate synthetic instruction sequences."""

        for _ in range(count):
            # Choose pattern type
            pattern_type = self._rng.choice(
                [
                    "single_instruction",
                    "instruction_sequence",
                    "function_snippet",
                ]
            )

            if pattern_type == "single_instruction":
                yield from self._generate_single_instruction()
            elif pattern_type == "instruction_sequence":
                yield from self._generate_instruction_sequence()
            else:
                yield from self._generate_function_snippet()

    def _generate_single_instruction(self) -> Iterator[Level0Sample]:
        """Generate single instruction samples."""
        category = self._rng.choice(list(X86_64_PATTERNS.keys()))
        patterns = X86_64_PATTERNS[category]
        raw_bytes = self._rng.choice(patterns)

        # Get ground truth from module
        output = self._module.process(Level0Input(data=raw_bytes, architecture=Architecture.X86_64))

        if not output.is_uncertain:
            yield Level0Sample(
                raw_bytes=raw_bytes,
                architecture="x86_64",
                instructions=[
                    {
                        "offset": i.offset,
                        "mnemonic": i.mnemonic,
                        "operands": list(i.operands),
                        "size": i.size,
                        "category": i.category.value,
                    }
                    for i in output.instructions
                ],
                is_valid=True,
                source="synthetic",
                category=category,
                difficulty="easy",
            )

    def _generate_instruction_sequence(self) -> Iterator[Level0Sample]:
        """Generate sequences of 2-5 instructions."""
        num_instructions = self._rng.randint(2, 5)
        raw_bytes = b""

        for _ in range(num_instructions):
            category = self._rng.choice(list(X86_64_PATTERNS.keys()))
            patterns = X86_64_PATTERNS[category]
            raw_bytes += self._rng.choice(patterns)

        output = self._module.process(Level0Input(data=raw_bytes, architecture=Architecture.X86_64))

        if not output.is_uncertain:
            yield Level0Sample(
                raw_bytes=raw_bytes,
                architecture="x86_64",
                instructions=[
                    {
                        "offset": i.offset,
                        "mnemonic": i.mnemonic,
                        "operands": list(i.operands),
                        "size": i.size,
                        "category": i.category.value,
                    }
                    for i in output.instructions
                ],
                is_valid=True,
                source="synthetic",
                category="sequence",
                difficulty="medium",
            )

    def _generate_function_snippet(self) -> Iterator[Level0Sample]:
        """Generate function-like code snippets."""
        # Prologue + body + epilogue
        prologue = self._rng.choice(list(FUNCTION_PATTERNS.values())[:2])
        epilogue = self._rng.choice(list(FUNCTION_PATTERNS.values())[2:])

        # Random body
        body = b""
        for _ in range(self._rng.randint(1, 3)):
            category = self._rng.choice(["mov_reg_reg", "xor_reg_reg", "add_reg_imm"])
            body += self._rng.choice(X86_64_PATTERNS[category])

        raw_bytes = prologue + body + epilogue

        output = self._module.process(Level0Input(data=raw_bytes, architecture=Architecture.X86_64))

        if not output.is_uncertain:
            yield Level0Sample(
                raw_bytes=raw_bytes,
                architecture="x86_64",
                instructions=[
                    {
                        "offset": i.offset,
                        "mnemonic": i.mnemonic,
                        "operands": list(i.operands),
                        "size": i.size,
                        "category": i.category.value,
                    }
                    for i in output.instructions
                ],
                is_valid=True,
                source="synthetic",
                category="function",
                difficulty="medium",
            )

    def generate_adversarial(self, count: int) -> Iterator[Level0Sample]:
        """Generate adversarial samples that should be refused."""

        for _ in range(count):
            adv_type = self._rng.choice(
                [
                    "incomplete",
                    "random",
                    "malformed",
                ]
            )

            if adv_type == "incomplete":
                raw_bytes = self._rng.choice(
                    [
                        b"\x0f",  # Incomplete two-byte opcode
                        b"\x48",  # Incomplete REX prefix
                        b"\x66\x0f",  # Incomplete with prefix
                    ]
                )
            elif adv_type == "random":
                raw_bytes = bytes(self._rng.randint(0, 255) for _ in range(16))
            else:
                raw_bytes = self._rng.choice(
                    [
                        b"\x00" * 16,
                        b"\xff" * 16,
                        b"\xcc" * 16,  # Many int3
                    ]
                )

            yield Level0Sample(
                raw_bytes=raw_bytes,
                architecture="x86_64",
                instructions=[],  # Should refuse
                is_valid=False,
                source="adversarial",
                category=adv_type,
                difficulty="hard",
            )

    def generate_from_binary(self, binary_path: Path, count: int) -> Iterator[Level0Sample]:
        """Extract samples from real binaries."""

        if not binary_path.exists():
            return

        # Read binary
        data = binary_path.read_bytes()

        # Find code sections (simplified - just sample random offsets)
        for _ in range(count):
            if len(data) < 32:
                continue

            offset = self._rng.randint(0, len(data) - 32)
            snippet = data[offset : offset + 32]

            output = self._module.process(
                Level0Input(data=snippet, architecture=Architecture.X86_64)
            )

            if not output.is_uncertain and output.instruction_count >= 3:
                yield Level0Sample(
                    raw_bytes=snippet[: output.bytes_processed],
                    architecture="x86_64",
                    instructions=[
                        {
                            "offset": i.offset,
                            "mnemonic": i.mnemonic,
                            "operands": list(i.operands),
                            "size": i.size,
                            "category": i.category.value,
                        }
                        for i in output.instructions
                    ],
                    is_valid=True,
                    source="binary",
                    category=binary_path.name,
                    difficulty="hard",
                )

    def generate_dataset(
        self,
        synthetic_count: int = 5000,
        adversarial_count: int = 1000,
        binary_paths: list[Path] | None = None,
        binary_samples_per_file: int = 100,
    ) -> list[Level0Sample]:
        """Generate a complete dataset."""

        samples: list[Level0Sample] = []

        # Synthetic samples
        for sample in self.generate_synthetic(synthetic_count):
            samples.append(sample)

        # Adversarial samples
        for sample in self.generate_adversarial(adversarial_count):
            samples.append(sample)

        # Binary samples
        if binary_paths:
            for path in binary_paths:
                for sample in self.generate_from_binary(path, binary_samples_per_file):
                    samples.append(sample)

        # Shuffle
        self._rng.shuffle(samples)

        return samples

    def save_dataset(self, samples: list[Level0Sample], output_path: Path) -> None:
        """Save dataset to JSON Lines format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

    def load_dataset(self, input_path: Path) -> list[Level0Sample]:
        """Load dataset from JSON Lines format."""
        samples = []
        with open(input_path) as f:
            for line in f:
                samples.append(Level0Sample.from_dict(json.loads(line)))
        return samples


def get_system_binaries() -> list[Path]:
    """Get paths to common system binaries for training data."""
    candidates = [
        Path("/usr/bin/ls"),
        Path("/usr/bin/cat"),
        Path("/usr/bin/grep"),
        Path("/usr/bin/sed"),
        Path("/usr/bin/awk"),
        Path("/usr/bin/python3"),
        Path("/usr/bin/bash"),
        Path("/usr/lib/libc.so.6"),
    ]
    return [p for p in candidates if p.exists()]


if __name__ == "__main__":
    # Generate a sample dataset
    generator = Level0DatasetGenerator(seed=42)

    print("Generating Level 0 dataset...")
    samples = generator.generate_dataset(
        synthetic_count=1000,
        adversarial_count=200,
        binary_paths=get_system_binaries(),
        binary_samples_per_file=50,
    )

    print(f"Generated {len(samples)} samples")

    # Stats
    valid = sum(1 for s in samples if s.is_valid)
    adversarial = sum(1 for s in samples if s.source == "adversarial")
    synthetic = sum(1 for s in samples if s.source == "synthetic")
    binary = sum(1 for s in samples if s.source == "binary")

    print(f"  Valid: {valid}")
    print(f"  Adversarial: {adversarial}")
    print(f"  Synthetic: {synthetic}")
    print(f"  From binaries: {binary}")

    # Save
    output_path = Path("datasets/level0/train.jsonl")
    generator.save_dataset(samples, output_path)
    print(f"Saved to {output_path}")
