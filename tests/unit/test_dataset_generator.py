"""Tests for Level 0 dataset generator."""

import json
import tempfile
from pathlib import Path

import pytest

from genesis_datasets.generators.level0_generator import (
    Level0DatasetGenerator,
    Level0Sample,
    get_system_binaries,
)


class TestLevel0Sample:
    """Tests for Level0Sample dataclass."""

    def test_to_dict_and_back(self) -> None:
        """Sample can be serialized and deserialized."""
        sample = Level0Sample(
            raw_bytes=b"\x90\x90\xc3",
            architecture="x86_64",
            instructions=[
                {"offset": 0, "mnemonic": "nop", "operands": [], "size": 1, "category": "system"},
                {"offset": 1, "mnemonic": "nop", "operands": [], "size": 1, "category": "system"},
                {
                    "offset": 2,
                    "mnemonic": "ret",
                    "operands": [],
                    "size": 1,
                    "category": "control_flow",
                },
            ],
            is_valid=True,
            source="synthetic",
            category="sequence",
            difficulty="easy",
        )

        # Round-trip
        data = sample.to_dict()
        restored = Level0Sample.from_dict(data)

        assert restored.raw_bytes == sample.raw_bytes
        assert restored.architecture == sample.architecture
        assert restored.instructions == sample.instructions
        assert restored.is_valid == sample.is_valid

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict output can be JSON serialized."""
        sample = Level0Sample(
            raw_bytes=b"\xff\x00\xab",
            architecture="x86_64",
            instructions=[],
            is_valid=False,
            source="adversarial",
            category="random",
            difficulty="hard",
        )

        # Should not raise
        json_str = json.dumps(sample.to_dict())
        assert isinstance(json_str, str)


class TestLevel0DatasetGenerator:
    """Tests for Level0DatasetGenerator."""

    @pytest.fixture
    def generator(self) -> Level0DatasetGenerator:
        return Level0DatasetGenerator(seed=42)

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same samples."""
        gen1 = Level0DatasetGenerator(seed=123)
        gen2 = Level0DatasetGenerator(seed=123)

        samples1 = list(gen1.generate_synthetic(10))
        samples2 = list(gen2.generate_synthetic(10))

        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2, strict=False):
            assert s1.raw_bytes == s2.raw_bytes

    def test_generate_synthetic_produces_valid_samples(
        self, generator: Level0DatasetGenerator
    ) -> None:
        """Synthetic samples are valid."""
        samples = list(generator.generate_synthetic(50))

        assert len(samples) > 0
        for sample in samples:
            assert sample.is_valid is True
            assert sample.source == "synthetic"
            assert len(sample.instructions) > 0
            assert len(sample.raw_bytes) > 0

    def test_generate_adversarial_produces_invalid_samples(
        self, generator: Level0DatasetGenerator
    ) -> None:
        """Adversarial samples are marked invalid."""
        samples = list(generator.generate_adversarial(50))

        assert len(samples) == 50
        for sample in samples:
            assert sample.is_valid is False
            assert sample.source == "adversarial"
            assert sample.instructions == []

    def test_generate_dataset_includes_all_types(self, generator: Level0DatasetGenerator) -> None:
        """Dataset includes synthetic and adversarial samples."""
        samples = generator.generate_dataset(
            synthetic_count=100,
            adversarial_count=20,
            binary_paths=None,
        )

        sources = {s.source for s in samples}
        assert "synthetic" in sources
        assert "adversarial" in sources

        # Check counts (synthetic may produce fewer due to filtering)
        adversarial_count = sum(1 for s in samples if s.source == "adversarial")
        assert adversarial_count == 20

    def test_save_and_load_dataset(self, generator: Level0DatasetGenerator) -> None:
        """Dataset can be saved and loaded."""
        samples = generator.generate_dataset(
            synthetic_count=20,
            adversarial_count=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            generator.save_dataset(samples, path)

            assert path.exists()

            loaded = generator.load_dataset(path)
            assert len(loaded) == len(samples)

            for orig, load in zip(samples, loaded, strict=False):
                assert orig.raw_bytes == load.raw_bytes
                assert orig.is_valid == load.is_valid

    def test_samples_have_required_fields(self, generator: Level0DatasetGenerator) -> None:
        """All samples have required fields."""
        samples = generator.generate_dataset(
            synthetic_count=50,
            adversarial_count=10,
        )

        for sample in samples:
            assert isinstance(sample.raw_bytes, bytes)
            assert sample.architecture in ["x86_64", "x86_32", "arm64", "arm32"]
            assert isinstance(sample.instructions, list)
            assert isinstance(sample.is_valid, bool)
            assert sample.source in ["synthetic", "adversarial", "binary"]
            assert sample.difficulty in ["easy", "medium", "hard"]


class TestSystemBinaries:
    """Tests for system binary discovery."""

    def test_get_system_binaries_returns_existing_paths(self) -> None:
        """Only returns paths that exist."""
        binaries = get_system_binaries()

        for path in binaries:
            assert path.exists(), f"{path} should exist"

    def test_get_system_binaries_returns_list(self) -> None:
        """Returns a list."""
        binaries = get_system_binaries()
        assert isinstance(binaries, list)


class TestDatasetQuality:
    """Tests for dataset quality properties."""

    @pytest.fixture
    def generator(self) -> Level0DatasetGenerator:
        return Level0DatasetGenerator(seed=42)

    def test_instruction_offsets_are_valid(self, generator: Level0DatasetGenerator) -> None:
        """Instruction offsets are within bounds."""
        samples = list(generator.generate_synthetic(50))

        for sample in samples:
            if sample.is_valid:
                for insn in sample.instructions:
                    assert insn["offset"] >= 0
                    assert insn["offset"] < len(sample.raw_bytes)

    def test_instruction_sizes_sum_correctly(self, generator: Level0DatasetGenerator) -> None:
        """Instruction sizes sum to total bytes."""
        samples = list(generator.generate_synthetic(50))

        for sample in samples:
            if sample.is_valid and sample.instructions:
                total_size = sum(i["size"] for i in sample.instructions)
                assert total_size == len(sample.raw_bytes)

    def test_adversarial_ratio_is_reasonable(self, generator: Level0DatasetGenerator) -> None:
        """Adversarial samples are ~15-20% of dataset."""
        samples = generator.generate_dataset(
            synthetic_count=1000,
            adversarial_count=200,
        )

        adversarial = sum(1 for s in samples if not s.is_valid)
        ratio = adversarial / len(samples)

        # Should be around 15-20%
        assert 0.10 <= ratio <= 0.25
