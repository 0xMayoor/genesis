"""Level 0 Module: Machine Code Pattern Recognition.

This module uses the Capstone disassembly engine as the foundation,
with additional pattern recognition and uncertainty handling.
"""

from capstone import (
    CS_ARCH_ARM,
    CS_ARCH_ARM64,
    CS_ARCH_X86,
    CS_MODE_32,
    CS_MODE_64,
    CS_MODE_ARM,
    Cs,
    CsError,
)

from core.types import UncertaintyReason, VerificationResult
from levels.level0_machine.types import (
    Architecture,
    Instruction,
    InstructionCategory,
    Level0Input,
    Level0Output,
)

# Mapping from mnemonic prefixes to categories
MNEMONIC_CATEGORIES: dict[str, InstructionCategory] = {
    # Data transfer
    "mov": InstructionCategory.DATA_TRANSFER,
    "lea": InstructionCategory.DATA_TRANSFER,
    "xchg": InstructionCategory.DATA_TRANSFER,
    "cmov": InstructionCategory.DATA_TRANSFER,
    # Stack
    "push": InstructionCategory.STACK,
    "pop": InstructionCategory.STACK,
    "enter": InstructionCategory.STACK,
    "leave": InstructionCategory.STACK,
    # Arithmetic
    "add": InstructionCategory.ARITHMETIC,
    "sub": InstructionCategory.ARITHMETIC,
    "mul": InstructionCategory.ARITHMETIC,
    "imul": InstructionCategory.ARITHMETIC,
    "div": InstructionCategory.ARITHMETIC,
    "idiv": InstructionCategory.ARITHMETIC,
    "inc": InstructionCategory.ARITHMETIC,
    "dec": InstructionCategory.ARITHMETIC,
    "neg": InstructionCategory.ARITHMETIC,
    # Logic
    "and": InstructionCategory.LOGIC,
    "or": InstructionCategory.LOGIC,
    "xor": InstructionCategory.LOGIC,
    "not": InstructionCategory.LOGIC,
    "shl": InstructionCategory.LOGIC,
    "shr": InstructionCategory.LOGIC,
    "sal": InstructionCategory.LOGIC,
    "sar": InstructionCategory.LOGIC,
    "rol": InstructionCategory.LOGIC,
    "ror": InstructionCategory.LOGIC,
    # Control flow
    "jmp": InstructionCategory.CONTROL_FLOW,
    "je": InstructionCategory.CONTROL_FLOW,
    "jne": InstructionCategory.CONTROL_FLOW,
    "jz": InstructionCategory.CONTROL_FLOW,
    "jnz": InstructionCategory.CONTROL_FLOW,
    "jg": InstructionCategory.CONTROL_FLOW,
    "jl": InstructionCategory.CONTROL_FLOW,
    "jge": InstructionCategory.CONTROL_FLOW,
    "jle": InstructionCategory.CONTROL_FLOW,
    "ja": InstructionCategory.CONTROL_FLOW,
    "jb": InstructionCategory.CONTROL_FLOW,
    "jae": InstructionCategory.CONTROL_FLOW,
    "jbe": InstructionCategory.CONTROL_FLOW,
    "call": InstructionCategory.CONTROL_FLOW,
    "ret": InstructionCategory.CONTROL_FLOW,
    "loop": InstructionCategory.CONTROL_FLOW,
    # Comparison
    "cmp": InstructionCategory.COMPARISON,
    "test": InstructionCategory.COMPARISON,
    # System
    "syscall": InstructionCategory.SYSTEM,
    "int": InstructionCategory.SYSTEM,
    "nop": InstructionCategory.SYSTEM,
    "hlt": InstructionCategory.SYSTEM,
    "cpuid": InstructionCategory.SYSTEM,
    # String
    "movs": InstructionCategory.STRING,
    "cmps": InstructionCategory.STRING,
    "scas": InstructionCategory.STRING,
    "lods": InstructionCategory.STRING,
    "stos": InstructionCategory.STRING,
    "rep": InstructionCategory.STRING,
}


def _get_category(mnemonic: str) -> InstructionCategory:
    """Get instruction category from mnemonic."""
    mnemonic_lower = mnemonic.lower()

    # Check exact match first
    if mnemonic_lower in MNEMONIC_CATEGORIES:
        return MNEMONIC_CATEGORIES[mnemonic_lower]

    # Check prefix match
    for prefix, category in MNEMONIC_CATEGORIES.items():
        if mnemonic_lower.startswith(prefix):
            return category

    # SIMD detection
    if any(mnemonic_lower.startswith(p) for p in ["v", "p", "movd", "movq"]):
        return InstructionCategory.SIMD

    return InstructionCategory.UNKNOWN


def _get_capstone_config(arch: Architecture) -> tuple[int, int]:
    """Get Capstone arch and mode for architecture."""
    configs = {
        Architecture.X86_64: (CS_ARCH_X86, CS_MODE_64),
        Architecture.X86_32: (CS_ARCH_X86, CS_MODE_32),
        Architecture.ARM64: (CS_ARCH_ARM64, CS_MODE_ARM),
        Architecture.ARM32: (CS_ARCH_ARM, CS_MODE_ARM),
    }
    if arch not in configs:
        raise ValueError(f"Unsupported architecture: {arch}")
    return configs[arch]


class Level0Module:
    """Level 0: Machine Code Pattern Recognition.

    This module decodes raw bytes into machine instructions.
    It uses Capstone as the disassembly backend with additional
    uncertainty handling and verification.

    Zero-hallucination principles:
    - Returns uncertain when data doesn't look like valid code
    - Verifies instruction boundaries
    - Reports confidence based on decode success rate
    """

    # Minimum confidence threshold
    CONFIDENCE_THRESHOLD = 0.85

    # Minimum valid instruction ratio to consider data as code
    MIN_VALID_RATIO = 0.7

    def __init__(self, default_arch: Architecture = Architecture.X86_64) -> None:
        self._default_arch = default_arch

    def process(self, input_data: Level0Input) -> Level0Output:
        """Process binary data and extract instructions.

        Args:
            input_data: Level0Input with binary data

        Returns:
            Level0Output with decoded instructions or uncertainty
        """
        # Determine architecture
        arch = input_data.architecture or self._default_arch
        if arch == Architecture.UNKNOWN:
            arch = self._detect_architecture(input_data.data)
            if arch == Architecture.UNKNOWN:
                return Level0Output.uncertain(
                    UncertaintyReason.AMBIGUOUS_INPUT,
                    details="Cannot determine architecture",
                )

        # Decode instructions
        try:
            instructions, bytes_processed = self._decode(
                input_data.data,
                arch,
                input_data.base_address,
                input_data.max_instructions,
            )
        except CsError as e:
            return Level0Output.uncertain(
                UncertaintyReason.MALFORMED_INPUT,
                architecture=arch,
                details=f"Disassembly error: {e}",
            )

        # Check if this looks like valid code
        if len(instructions) == 0:
            return Level0Output.uncertain(
                UncertaintyReason.MALFORMED_INPUT,
                architecture=arch,
                details="No valid instructions found",
            )

        # Calculate confidence based on coverage and validity
        coverage = bytes_processed / len(input_data.data)
        unknown_ratio = sum(
            1 for i in instructions if i.category == InstructionCategory.UNKNOWN
        ) / len(instructions)

        confidence = coverage * (1 - unknown_ratio * 0.3)

        # Check if confidence is too low
        if confidence < self.MIN_VALID_RATIO:
            return Level0Output.uncertain(
                UncertaintyReason.LOW_CONFIDENCE,
                architecture=arch,
                details=f"Low confidence: {confidence:.2%}",
            )

        # Verify the output
        verification = self._verify(instructions, input_data.data, bytes_processed)

        return Level0Output.success(
            instructions=tuple(instructions),
            architecture=arch,
            bytes_processed=bytes_processed,
            confidence=min(confidence, 0.99),
            verification=verification,
        )

    def _decode(
        self,
        data: bytes,
        arch: Architecture,
        base_address: int,
        max_instructions: int | None,
    ) -> tuple[list[Instruction], int]:
        """Decode bytes into instructions using Capstone."""
        cs_arch, cs_mode = _get_capstone_config(arch)
        md = Cs(cs_arch, cs_mode)
        md.detail = False  # We don't need detailed info yet

        instructions: list[Instruction] = []
        bytes_processed = 0

        for insn in md.disasm(data, base_address):
            instruction = Instruction(
                offset=insn.address - base_address,
                raw_bytes=bytes(insn.bytes),
                mnemonic=insn.mnemonic,
                operands=tuple(insn.op_str.split(", ")) if insn.op_str else (),
                size=insn.size,
                category=_get_category(insn.mnemonic),
            )
            instructions.append(instruction)
            bytes_processed += insn.size

            if max_instructions and len(instructions) >= max_instructions:
                break

        return instructions, bytes_processed

    def _detect_architecture(self, data: bytes) -> Architecture:
        """Attempt to detect architecture from byte patterns.

        This is a simple heuristic - returns UNKNOWN if uncertain.
        """
        # Try x86_64 first (most common)
        try:
            cs_arch, cs_mode = _get_capstone_config(Architecture.X86_64)
            md = Cs(cs_arch, cs_mode)

            count = sum(1 for _ in md.disasm(data[:64], 0))
            if count > 5:  # At least 5 valid instructions
                return Architecture.X86_64
        except CsError:
            pass

        return Architecture.UNKNOWN

    def _verify(
        self,
        instructions: list[Instruction],
        original_data: bytes,
        bytes_processed: int,
    ) -> VerificationResult:
        """Verify decoded instructions against original data."""
        # Reconstruct bytes from instructions
        reconstructed = b"".join(i.raw_bytes for i in instructions)

        # Check if reconstructed matches original
        if reconstructed == original_data[:bytes_processed]:
            return VerificationResult(
                passed=True,
                method="byte_reconstruction",
                details=f"All {len(instructions)} instructions verified",
            )
        else:
            return VerificationResult(
                passed=False,
                method="byte_reconstruction",
                errors=["Reconstructed bytes don't match original"],
            )

    def can_handle(self, input_data: Level0Input) -> bool:
        """Check if this module can handle the input."""
        # We can attempt to handle any binary data
        return len(input_data.data) > 0
