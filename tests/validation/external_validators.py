"""External validation tools.

These wrap external disassemblers to provide ground truth
for comparing against GENESIS Level 0 output.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExternalInstruction:
    """Instruction from external disassembler."""

    offset: int
    raw_bytes: bytes
    mnemonic: str
    operands: str

    @property
    def assembly(self) -> str:
        if self.operands:
            return f"{self.mnemonic} {self.operands}"
        return self.mnemonic


class ExternalValidator:
    """Base class for external validators."""

    def __init__(self) -> None:
        self._available: bool | None = None

    @property
    def name(self) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if the external tool is available."""
        raise NotImplementedError

    def disassemble(self, data: bytes, arch: str = "x86_64") -> list[ExternalInstruction]:
        """Disassemble bytes and return instructions."""
        raise NotImplementedError


class ObjdumpValidator(ExternalValidator):
    """Validator using GNU objdump."""

    @property
    def name(self) -> str:
        return "objdump"

    def is_available(self) -> bool:
        if self._available is None:
            try:
                result = subprocess.run(
                    ["objdump", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._available = False
        return self._available

    def disassemble(self, data: bytes, arch: str = "x86_64") -> list[ExternalInstruction]:
        """Disassemble using objdump."""
        if not self.is_available():
            raise RuntimeError("objdump not available")

        # Map architecture names
        arch_map = {
            "x86_64": "i386:x86-64",
            "x86_32": "i386",
            "arm64": "aarch64",
            "arm32": "arm",
        }
        objdump_arch = arch_map.get(arch, arch)

        # Write bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                [
                    "objdump",
                    "-D",
                    "-b",
                    "binary",
                    "-m",
                    objdump_arch,
                    str(temp_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return self._parse_objdump_output(result.stdout)
        finally:
            temp_path.unlink(missing_ok=True)

    def _parse_objdump_output(self, output: str) -> list[ExternalInstruction]:
        """Parse objdump output into instructions."""
        instructions = []

        # Pattern: "   0:   89 d8                   mov    %ebx,%eax"
        # objdump uses tabs between columns
        pattern = re.compile(
            r"^\s*([0-9a-f]+):\s+"  # offset with colon
            r"((?:[0-9a-f]{2}\s)+)\s*"  # hex bytes (pairs separated by spaces)
            r"(\S+)"  # mnemonic
            r"(?:\s+(.*))?$",  # operands (optional)
            re.IGNORECASE,
        )

        for line in output.split("\n"):
            match = pattern.match(line)
            if match:
                offset = int(match.group(1), 16)
                raw_bytes = bytes.fromhex(match.group(2).replace(" ", ""))
                mnemonic = match.group(3)
                operands = match.group(4) or ""

                instructions.append(
                    ExternalInstruction(
                        offset=offset,
                        raw_bytes=raw_bytes,
                        mnemonic=mnemonic,
                        operands=operands.strip(),
                    )
                )

        return instructions


class Radare2Validator(ExternalValidator):
    """Validator using radare2."""

    @property
    def name(self) -> str:
        return "radare2"

    def is_available(self) -> bool:
        if self._available is None:
            try:
                result = subprocess.run(
                    ["r2", "-v"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._available = False
        return self._available

    def disassemble(self, data: bytes, arch: str = "x86_64") -> list[ExternalInstruction]:
        """Disassemble using radare2."""
        if not self.is_available():
            raise RuntimeError("radare2 not available")

        # Map architecture names
        arch_map = {
            "x86_64": "x86",
            "x86_32": "x86",
            "arm64": "arm",
            "arm32": "arm",
        }
        bits_map = {
            "x86_64": "64",
            "x86_32": "32",
            "arm64": "64",
            "arm32": "32",
        }
        r2_arch = arch_map.get(arch, "x86")
        r2_bits = bits_map.get(arch, "64")

        # Write bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            # Use r2 with commands: set arch, set bits, analyze, print disassembly
            result = subprocess.run(
                [
                    "r2",
                    "-q",  # quiet
                    "-c",
                    f"e asm.arch={r2_arch}; e asm.bits={r2_bits}; pd {len(data)}",
                    str(temp_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return self._parse_r2_output(result.stdout)
        finally:
            temp_path.unlink(missing_ok=True)

    def _parse_r2_output(self, output: str) -> list[ExternalInstruction]:
        """Parse radare2 output into instructions."""
        instructions = []

        # Pattern: "0x00000000      55             push rbp"
        pattern = re.compile(
            r"^\s*0x([0-9a-f]+)\s+"  # offset
            r"([0-9a-f]+)\s+"  # bytes
            r"(\S+)"  # mnemonic
            r"(?:\s+(.*))?$",  # operands (optional)
            re.IGNORECASE,
        )

        for line in output.split("\n"):
            match = pattern.match(line)
            if match:
                offset = int(match.group(1), 16)
                raw_bytes = bytes.fromhex(match.group(2))
                mnemonic = match.group(3)
                operands = match.group(4) or ""

                instructions.append(
                    ExternalInstruction(
                        offset=offset,
                        raw_bytes=raw_bytes,
                        mnemonic=mnemonic,
                        operands=operands.strip(),
                    )
                )

        return instructions


class NdisasmValidator(ExternalValidator):
    """Validator using NASM ndisasm."""

    @property
    def name(self) -> str:
        return "ndisasm"

    def is_available(self) -> bool:
        if self._available is None:
            try:
                subprocess.run(
                    ["ndisasm", "-h"],
                    capture_output=True,
                    timeout=5,
                )
                # ndisasm returns 1 for -h but still works
                self._available = True
            except FileNotFoundError:
                self._available = False
            except subprocess.TimeoutExpired:
                self._available = False
        return self._available

    def disassemble(self, data: bytes, arch: str = "x86_64") -> list[ExternalInstruction]:
        """Disassemble using ndisasm."""
        if not self.is_available():
            raise RuntimeError("ndisasm not available")

        # Map architecture to bits
        bits_map = {
            "x86_64": "64",
            "x86_32": "32",
        }
        bits = bits_map.get(arch)
        if bits is None:
            raise ValueError(f"ndisasm doesn't support {arch}")

        # Write bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                ["ndisasm", f"-b{bits}", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return self._parse_ndisasm_output(result.stdout)
        finally:
            temp_path.unlink(missing_ok=True)

    def _parse_ndisasm_output(self, output: str) -> list[ExternalInstruction]:
        """Parse ndisasm output into instructions."""
        instructions = []

        # Pattern: "00000000  55                push rbp"
        pattern = re.compile(
            r"^([0-9A-F]+)\s+"  # offset
            r"([0-9A-F]+)\s+"  # bytes
            r"(\S+)"  # mnemonic
            r"(?:\s+(.*))?$",  # operands (optional)
            re.IGNORECASE,
        )

        for line in output.split("\n"):
            match = pattern.match(line)
            if match:
                offset = int(match.group(1), 16)
                raw_bytes = bytes.fromhex(match.group(2))
                mnemonic = match.group(3).lower()
                operands = match.group(4) or ""

                instructions.append(
                    ExternalInstruction(
                        offset=offset,
                        raw_bytes=raw_bytes,
                        mnemonic=mnemonic,
                        operands=operands.strip(),
                    )
                )

        return instructions


def get_available_validators() -> list[ExternalValidator]:
    """Get list of available external validators."""
    validators = [
        ObjdumpValidator(),
        Radare2Validator(),
        NdisasmValidator(),
    ]
    return [v for v in validators if v.is_available()]
