"""Level 1 Module: Assembly Semantics.

This module provides deterministic semantic analysis of x86_64 instructions.
It serves as ground truth for training the Level 1 model.
"""

from levels.level0_machine.types import Instruction
from levels.level1_assembly.types import (
    Level1Input,
    Level1Output,
    RegisterEffect,
    MemoryEffect,
    FlagEffect,
    ControlFlowEffect,
    EffectOperation,
    FlagOperation,
    ControlFlowType,
    X86_64_REGISTERS,
    X86_64_FLAGS,
)


class Level1Module:
    """Deterministic assembly semantics analyzer."""
    
    def __init__(self) -> None:
        self._init_semantics()
    
    def _init_semantics(self) -> None:
        """Initialize semantic rules for instructions."""
        # Map mnemonic -> handler function
        self._handlers = {
            # Data movement
            "mov": self._sem_mov,
            "movzx": self._sem_mov,
            "movsx": self._sem_mov,
            "movsxd": self._sem_mov,
            "lea": self._sem_lea,
            "xchg": self._sem_xchg,
            "push": self._sem_push,
            "pop": self._sem_pop,
            
            # Arithmetic
            "add": self._sem_arithmetic,
            "sub": self._sem_arithmetic,
            "inc": self._sem_inc_dec,
            "dec": self._sem_inc_dec,
            "neg": self._sem_neg,
            "mul": self._sem_mul,
            "imul": self._sem_imul,
            "div": self._sem_div,
            "idiv": self._sem_div,
            
            # Logic
            "and": self._sem_logic,
            "or": self._sem_logic,
            "xor": self._sem_logic,
            "not": self._sem_not,
            "shl": self._sem_shift,
            "shr": self._sem_shift,
            "sar": self._sem_shift,
            "rol": self._sem_rotate,
            "ror": self._sem_rotate,
            
            # Comparison
            "cmp": self._sem_cmp,
            "test": self._sem_test,
            
            # Control flow
            "jmp": self._sem_jmp,
            "je": self._sem_jcc, "jz": self._sem_jcc,
            "jne": self._sem_jcc, "jnz": self._sem_jcc,
            "jl": self._sem_jcc, "jnge": self._sem_jcc,
            "jle": self._sem_jcc, "jng": self._sem_jcc,
            "jg": self._sem_jcc, "jnle": self._sem_jcc,
            "jge": self._sem_jcc, "jnl": self._sem_jcc,
            "jb": self._sem_jcc, "jc": self._sem_jcc, "jnae": self._sem_jcc,
            "jbe": self._sem_jcc, "jna": self._sem_jcc,
            "ja": self._sem_jcc, "jnbe": self._sem_jcc,
            "jae": self._sem_jcc, "jnc": self._sem_jcc, "jnb": self._sem_jcc,
            "js": self._sem_jcc,
            "jns": self._sem_jcc,
            "call": self._sem_call,
            "ret": self._sem_ret,
            
            # System
            "nop": self._sem_nop,
            "syscall": self._sem_syscall,
            "int": self._sem_int,
            "int3": self._sem_int,
            "hlt": self._sem_nop,
            
            # Stack frame
            "leave": self._sem_leave,
            "enter": self._sem_enter,
            
            # Flags
            "pushf": self._sem_pushf,
            "popf": self._sem_popf,
            "cld": self._sem_flag_clear,
            "std": self._sem_flag_set,
            
            # String ops (basic)
            "movsb": self._sem_string_mov,
            "movsw": self._sem_string_mov,
            "movsd": self._sem_string_mov,
            "movsq": self._sem_string_mov,
            "stosb": self._sem_string_sto,
            "stosw": self._sem_string_sto,
            "stosd": self._sem_string_sto,
            "stosq": self._sem_string_sto,
            "lodsb": self._sem_string_lod,
            "lodsw": self._sem_string_lod,
            "lodsd": self._sem_string_lod,
            "lodsq": self._sem_string_lod,
            "scasb": self._sem_string_sca,
            "scasw": self._sem_string_sca,
            "scasd": self._sem_string_sca,
            "scasq": self._sem_string_sca,
            "rep": self._sem_rep,
            "repne": self._sem_rep,
            
            # Bit manipulation
            "bsf": self._sem_bit_scan,
            "bsr": self._sem_bit_scan,
            "bswap": self._sem_bswap,
            
            # Set byte on condition
            "sete": self._sem_setcc, "setz": self._sem_setcc,
            "setne": self._sem_setcc, "setnz": self._sem_setcc,
            "setl": self._sem_setcc,
            "setg": self._sem_setcc,
            "setb": self._sem_setcc,
            "setae": self._sem_setcc,
            
            # Conditional move
            "cmove": self._sem_cmov, "cmovz": self._sem_cmov,
            "cmovne": self._sem_cmov, "cmovnz": self._sem_cmov,
            "cmovl": self._sem_cmov,
            "cmovg": self._sem_cmov,
            
            # Misc
            "cpuid": self._sem_cpuid,
            "rdtsc": self._sem_rdtsc,
        }
    
    def analyze(self, input: Level1Input) -> Level1Output:
        """Analyze instruction semantics."""
        instr = input.instruction
        mnemonic = instr.mnemonic.lower()
        
        # Check for unknown mnemonic
        if mnemonic not in self._handlers:
            return Level1Output(
                instruction=instr,
                is_uncertain=True,
                uncertainty_reason=f"Unknown instruction: {mnemonic}",
                confidence=0.0,
            )
        
        # Get handler and execute
        handler = self._handlers[mnemonic]
        try:
            return handler(instr)
        except Exception as e:
            return Level1Output(
                instruction=instr,
                is_uncertain=True,
                uncertainty_reason=f"Analysis error: {e}",
                confidence=0.0,
            )
    
    # === Semantic Handlers ===
    
    def _sem_mov(self, instr: Instruction) -> Level1Output:
        """MOV dst, src - Move data."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, "MOV requires 2 operands")
        
        dst, src = ops[0], ops[1]
        effects = []
        mem_effects = []
        
        # Destination
        if self._is_register(dst):
            effects.append(RegisterEffect(dst, EffectOperation.WRITE, src))
        elif self._is_memory(dst):
            mem_effects.append(MemoryEffect(EffectOperation.WRITE, self._parse_mem(dst), self._mem_size(dst)))
        
        # Source
        if self._is_register(src):
            effects.append(RegisterEffect(src, EffectOperation.READ))
        elif self._is_memory(src):
            mem_effects.append(MemoryEffect(EffectOperation.READ, self._parse_mem(src), self._mem_size(src)))
        
        return Level1Output(
            instruction=instr,
            register_effects=effects,
            memory_effects=mem_effects,
            flag_effects=[],  # MOV doesn't affect flags
        )
    
    def _sem_lea(self, instr: Instruction) -> Level1Output:
        """LEA dst, [src] - Load effective address."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, "LEA requires 2 operands")
        
        dst, src = ops[0], ops[1]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.WRITE, f"&{src}"),
            ],
            flag_effects=[],  # LEA doesn't affect flags
        )
    
    def _sem_xchg(self, instr: Instruction) -> Level1Output:
        """XCHG a, b - Exchange values."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, "XCHG requires 2 operands")
        
        a, b = ops[0], ops[1]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(a, EffectOperation.READ_WRITE, b),
                RegisterEffect(b, EffectOperation.READ_WRITE, a),
            ],
            flag_effects=[],
        )
    
    def _sem_push(self, instr: Instruction) -> Level1Output:
        """PUSH src - Push to stack."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "PUSH requires 1 operand")
        
        src = ops[0]
        size = 8  # 64-bit mode default
        
        effects = [
            RegisterEffect("rsp", EffectOperation.READ_WRITE, f"rsp - {size}"),
        ]
        if self._is_register(src):
            effects.append(RegisterEffect(src, EffectOperation.READ))
        
        return Level1Output(
            instruction=instr,
            register_effects=effects,
            memory_effects=[
                MemoryEffect(EffectOperation.WRITE, "rsp", size),
            ],
            flag_effects=[],
        )
    
    def _sem_pop(self, instr: Instruction) -> Level1Output:
        """POP dst - Pop from stack."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "POP requires 1 operand")
        
        dst = ops[0]
        size = 8
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.WRITE, "[rsp]"),
                RegisterEffect("rsp", EffectOperation.READ_WRITE, f"rsp + {size}"),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rsp", size),
            ],
            flag_effects=[],
        )
    
    def _sem_arithmetic(self, instr: Instruction) -> Level1Output:
        """ADD/SUB dst, src - Arithmetic with flags."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, f"{instr.mnemonic} requires 2 operands")
        
        dst, src = ops[0], ops[1]
        op = "+" if instr.mnemonic.lower() == "add" else "-"
        
        effects = [
            RegisterEffect(dst, EffectOperation.READ_WRITE, f"{dst} {op} {src}"),
        ]
        if self._is_register(src):
            effects.append(RegisterEffect(src, EffectOperation.READ))
        
        return Level1Output(
            instruction=instr,
            register_effects=effects,
            flag_effects=self._arithmetic_flags(),
        )
    
    def _sem_inc_dec(self, instr: Instruction) -> Level1Output:
        """INC/DEC dst - Increment/decrement (doesn't affect CF!)."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, f"{instr.mnemonic} requires 1 operand")
        
        dst = ops[0]
        op = "+ 1" if instr.mnemonic.lower() == "inc" else "- 1"
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"{dst} {op}"),
            ],
            flag_effects=[
                # Note: INC/DEC don't affect CF!
                FlagEffect("OF", FlagOperation.MODIFIED, "signed overflow"),
                FlagEffect("SF", FlagOperation.MODIFIED, "result < 0"),
                FlagEffect("ZF", FlagOperation.MODIFIED, "result == 0"),
                FlagEffect("AF", FlagOperation.MODIFIED, "aux carry"),
                FlagEffect("PF", FlagOperation.MODIFIED, "parity"),
            ],
        )
    
    def _sem_neg(self, instr: Instruction) -> Level1Output:
        """NEG dst - Two's complement negation."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "NEG requires 1 operand")
        
        dst = ops[0]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"-{dst}"),
            ],
            flag_effects=self._arithmetic_flags(),
        )
    
    def _sem_mul(self, instr: Instruction) -> Level1Output:
        """MUL src - Unsigned multiply (uses implicit rax, rdx)."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "MUL requires 1 operand")
        
        src = ops[0]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rax", EffectOperation.READ_WRITE, f"rax * {src}"),
                RegisterEffect("rdx", EffectOperation.WRITE, "high bits"),
                RegisterEffect(src, EffectOperation.READ) if self._is_register(src) else None,
            ],
            flag_effects=[
                FlagEffect("CF", FlagOperation.MODIFIED, "high bits != 0"),
                FlagEffect("OF", FlagOperation.MODIFIED, "high bits != 0"),
            ],
        )
    
    def _sem_imul(self, instr: Instruction) -> Level1Output:
        """IMUL - Signed multiply (multiple forms)."""
        ops = list(instr.operands)
        
        if len(ops) == 1:
            # One operand form: rdx:rax = rax * src
            return self._sem_mul(instr)
        elif len(ops) == 2:
            # Two operand form: dst = dst * src
            dst, src = ops[0], ops[1]
            return Level1Output(
                instruction=instr,
                register_effects=[
                    RegisterEffect(dst, EffectOperation.READ_WRITE, f"{dst} * {src}"),
                ],
                flag_effects=[
                    FlagEffect("CF", FlagOperation.MODIFIED),
                    FlagEffect("OF", FlagOperation.MODIFIED),
                ],
            )
        else:
            return self._uncertain(instr, "IMUL: invalid operand count")
    
    def _sem_div(self, instr: Instruction) -> Level1Output:
        """DIV/IDIV src - Division (uses implicit rdx:rax)."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "DIV requires 1 operand")
        
        src = ops[0]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rax", EffectOperation.READ_WRITE, f"rdx:rax / {src}"),
                RegisterEffect("rdx", EffectOperation.READ_WRITE, f"rdx:rax % {src}"),
            ],
            flag_effects=[],  # DIV flags are undefined
        )
    
    def _sem_logic(self, instr: Instruction) -> Level1Output:
        """AND/OR/XOR dst, src - Logical operations."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, f"{instr.mnemonic} requires 2 operands")
        
        dst, src = ops[0], ops[1]
        op_map = {"and": "&", "or": "|", "xor": "^"}
        op = op_map.get(instr.mnemonic.lower(), "?")
        
        effects = [
            RegisterEffect(dst, EffectOperation.READ_WRITE, f"{dst} {op} {src}"),
        ]
        if self._is_register(src):
            effects.append(RegisterEffect(src, EffectOperation.READ))
        
        return Level1Output(
            instruction=instr,
            register_effects=effects,
            flag_effects=[
                FlagEffect("CF", FlagOperation.CLEAR),  # Always cleared
                FlagEffect("OF", FlagOperation.CLEAR),  # Always cleared
                FlagEffect("SF", FlagOperation.MODIFIED, "result < 0"),
                FlagEffect("ZF", FlagOperation.MODIFIED, "result == 0"),
                FlagEffect("PF", FlagOperation.MODIFIED, "parity"),
            ],
        )
    
    def _sem_not(self, instr: Instruction) -> Level1Output:
        """NOT dst - Bitwise NOT."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "NOT requires 1 operand")
        
        dst = ops[0]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"~{dst}"),
            ],
            flag_effects=[],  # NOT doesn't affect flags
        )
    
    def _sem_shift(self, instr: Instruction) -> Level1Output:
        """SHL/SHR/SAR dst, count - Shift operations."""
        ops = list(instr.operands)
        if len(ops) < 1:
            return self._uncertain(instr, f"{instr.mnemonic} requires operands")
        
        dst = ops[0]
        count = ops[1] if len(ops) > 1 else "1"
        op_map = {"shl": "<<", "shr": ">>", "sar": ">>>"}
        op = op_map.get(instr.mnemonic.lower(), "?")
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"{dst} {op} {count}"),
            ],
            flag_effects=[
                FlagEffect("CF", FlagOperation.MODIFIED, "last bit shifted out"),
                FlagEffect("OF", FlagOperation.MODIFIED, "if count == 1"),
                FlagEffect("SF", FlagOperation.MODIFIED),
                FlagEffect("ZF", FlagOperation.MODIFIED),
            ],
        )
    
    def _sem_rotate(self, instr: Instruction) -> Level1Output:
        """ROL/ROR dst, count - Rotate operations."""
        ops = list(instr.operands)
        if len(ops) < 1:
            return self._uncertain(instr, f"{instr.mnemonic} requires operands")
        
        dst = ops[0]
        count = ops[1] if len(ops) > 1 else "1"
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"rotate({dst}, {count})"),
            ],
            flag_effects=[
                FlagEffect("CF", FlagOperation.MODIFIED),
                FlagEffect("OF", FlagOperation.MODIFIED, "if count == 1"),
            ],
        )
    
    def _sem_cmp(self, instr: Instruction) -> Level1Output:
        """CMP a, b - Compare (sets flags like SUB but doesn't store)."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, "CMP requires 2 operands")
        
        a, b = ops[0], ops[1]
        effects = []
        if self._is_register(a):
            effects.append(RegisterEffect(a, EffectOperation.READ))
        if self._is_register(b):
            effects.append(RegisterEffect(b, EffectOperation.READ))
        
        return Level1Output(
            instruction=instr,
            register_effects=effects,
            flag_effects=self._arithmetic_flags(),
        )
    
    def _sem_test(self, instr: Instruction) -> Level1Output:
        """TEST a, b - Test (sets flags like AND but doesn't store)."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, "TEST requires 2 operands")
        
        a, b = ops[0], ops[1]
        effects = []
        if self._is_register(a):
            effects.append(RegisterEffect(a, EffectOperation.READ))
        if self._is_register(b):
            effects.append(RegisterEffect(b, EffectOperation.READ))
        
        return Level1Output(
            instruction=instr,
            register_effects=effects,
            flag_effects=[
                FlagEffect("CF", FlagOperation.CLEAR),
                FlagEffect("OF", FlagOperation.CLEAR),
                FlagEffect("SF", FlagOperation.MODIFIED),
                FlagEffect("ZF", FlagOperation.MODIFIED),
                FlagEffect("PF", FlagOperation.MODIFIED),
            ],
        )
    
    def _sem_jmp(self, instr: Instruction) -> Level1Output:
        """JMP target - Unconditional jump."""
        ops = list(instr.operands)
        target = ops[0] if ops else "?"
        
        return Level1Output(
            instruction=instr,
            control_flow=ControlFlowEffect(
                type=ControlFlowType.JUMP,
                target_expr=str(target),
            ),
        )
    
    def _sem_jcc(self, instr: Instruction) -> Level1Output:
        """Jcc target - Conditional jump."""
        ops = list(instr.operands)
        target = ops[0] if ops else "?"
        
        # Map mnemonic to condition
        conditions = {
            "je": "ZF == 1", "jz": "ZF == 1",
            "jne": "ZF == 0", "jnz": "ZF == 0",
            "jl": "SF != OF", "jnge": "SF != OF",
            "jle": "ZF == 1 or SF != OF", "jng": "ZF == 1 or SF != OF",
            "jg": "ZF == 0 and SF == OF", "jnle": "ZF == 0 and SF == OF",
            "jge": "SF == OF", "jnl": "SF == OF",
            "jb": "CF == 1", "jc": "CF == 1", "jnae": "CF == 1",
            "jbe": "CF == 1 or ZF == 1", "jna": "CF == 1 or ZF == 1",
            "ja": "CF == 0 and ZF == 0", "jnbe": "CF == 0 and ZF == 0",
            "jae": "CF == 0", "jnc": "CF == 0", "jnb": "CF == 0",
            "js": "SF == 1",
            "jns": "SF == 0",
        }
        
        cond = conditions.get(instr.mnemonic.lower(), "?")
        
        return Level1Output(
            instruction=instr,
            control_flow=ControlFlowEffect(
                type=ControlFlowType.CONDITIONAL,
                target_expr=str(target),
                condition=cond,
            ),
        )
    
    def _sem_call(self, instr: Instruction) -> Level1Output:
        """CALL target - Function call."""
        ops = list(instr.operands)
        target = ops[0] if ops else "?"
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rsp", EffectOperation.READ_WRITE, "rsp - 8"),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.WRITE, "rsp", 8),  # Return address
            ],
            control_flow=ControlFlowEffect(
                type=ControlFlowType.CALL,
                target_expr=str(target),
            ),
        )
    
    def _sem_ret(self, instr: Instruction) -> Level1Output:
        """RET - Return from function."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rsp", EffectOperation.READ_WRITE, "rsp + 8"),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rsp", 8),  # Return address
            ],
            control_flow=ControlFlowEffect(
                type=ControlFlowType.RETURN,
                target_expr="[rsp]",
            ),
        )
    
    def _sem_nop(self, instr: Instruction) -> Level1Output:
        """NOP - No operation."""
        return Level1Output(instruction=instr)
    
    def _sem_syscall(self, instr: Instruction) -> Level1Output:
        """SYSCALL - System call."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rcx", EffectOperation.WRITE, "rip"),
                RegisterEffect("r11", EffectOperation.WRITE, "rflags"),
                RegisterEffect("rax", EffectOperation.READ_WRITE, "syscall result"),
            ],
            control_flow=ControlFlowEffect(type=ControlFlowType.INTERRUPT),
        )
    
    def _sem_int(self, instr: Instruction) -> Level1Output:
        """INT n - Software interrupt."""
        return Level1Output(
            instruction=instr,
            control_flow=ControlFlowEffect(type=ControlFlowType.INTERRUPT),
        )
    
    def _sem_leave(self, instr: Instruction) -> Level1Output:
        """LEAVE - Equivalent to mov rsp, rbp; pop rbp."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rsp", EffectOperation.WRITE, "rbp"),
                RegisterEffect("rbp", EffectOperation.READ_WRITE, "[rbp]"),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rbp", 8),
            ],
        )
    
    def _sem_enter(self, instr: Instruction) -> Level1Output:
        """ENTER size, nesting - Create stack frame."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rbp", EffectOperation.READ_WRITE),
                RegisterEffect("rsp", EffectOperation.READ_WRITE),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.WRITE, "rsp", 8),
            ],
        )
    
    def _sem_pushf(self, instr: Instruction) -> Level1Output:
        """PUSHF - Push flags."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rsp", EffectOperation.READ_WRITE, "rsp - 8"),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.WRITE, "rsp", 8),
            ],
        )
    
    def _sem_popf(self, instr: Instruction) -> Level1Output:
        """POPF - Pop flags."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rsp", EffectOperation.READ_WRITE, "rsp + 8"),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rsp", 8),
            ],
            flag_effects=[FlagEffect(f, FlagOperation.MODIFIED) for f in X86_64_FLAGS],
        )
    
    def _sem_flag_clear(self, instr: Instruction) -> Level1Output:
        """CLD - Clear direction flag."""
        return Level1Output(
            instruction=instr,
            flag_effects=[FlagEffect("DF", FlagOperation.CLEAR)],
        )
    
    def _sem_flag_set(self, instr: Instruction) -> Level1Output:
        """STD - Set direction flag."""
        return Level1Output(
            instruction=instr,
            flag_effects=[FlagEffect("DF", FlagOperation.SET)],
        )
    
    def _sem_string_mov(self, instr: Instruction) -> Level1Output:
        """MOVS - Move string."""
        size = {"movsb": 1, "movsw": 2, "movsd": 4, "movsq": 8}.get(instr.mnemonic.lower(), 1)
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rsi", EffectOperation.READ_WRITE),
                RegisterEffect("rdi", EffectOperation.READ_WRITE),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rsi", size),
                MemoryEffect(EffectOperation.WRITE, "rdi", size),
            ],
        )
    
    def _sem_string_sto(self, instr: Instruction) -> Level1Output:
        """STOS - Store string."""
        size = {"stosb": 1, "stosw": 2, "stosd": 4, "stosq": 8}.get(instr.mnemonic.lower(), 1)
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rax", EffectOperation.READ),
                RegisterEffect("rdi", EffectOperation.READ_WRITE),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.WRITE, "rdi", size),
            ],
        )
    
    def _sem_string_lod(self, instr: Instruction) -> Level1Output:
        """LODS - Load string."""
        size = {"lodsb": 1, "lodsw": 2, "lodsd": 4, "lodsq": 8}.get(instr.mnemonic.lower(), 1)
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rax", EffectOperation.WRITE),
                RegisterEffect("rsi", EffectOperation.READ_WRITE),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rsi", size),
            ],
        )
    
    def _sem_string_sca(self, instr: Instruction) -> Level1Output:
        """SCAS - Scan string."""
        size = {"scasb": 1, "scasw": 2, "scasd": 4, "scasq": 8}.get(instr.mnemonic.lower(), 1)
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rax", EffectOperation.READ),
                RegisterEffect("rdi", EffectOperation.READ_WRITE),
            ],
            memory_effects=[
                MemoryEffect(EffectOperation.READ, "rdi", size),
            ],
            flag_effects=self._arithmetic_flags(),
        )
    
    def _sem_rep(self, instr: Instruction) -> Level1Output:
        """REP prefix - Repeat string operation."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("rcx", EffectOperation.READ_WRITE, "rcx - 1"),
            ],
        )
    
    def _sem_bit_scan(self, instr: Instruction) -> Level1Output:
        """BSF/BSR - Bit scan."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, f"{instr.mnemonic} requires 2 operands")
        
        dst, src = ops[0], ops[1]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.WRITE),
                RegisterEffect(src, EffectOperation.READ) if self._is_register(src) else None,
            ],
            flag_effects=[
                FlagEffect("ZF", FlagOperation.MODIFIED, "src == 0"),
            ],
        )
    
    def _sem_bswap(self, instr: Instruction) -> Level1Output:
        """BSWAP - Byte swap."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, "BSWAP requires 1 operand")
        
        dst = ops[0]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"bswap({dst})"),
            ],
        )
    
    def _sem_setcc(self, instr: Instruction) -> Level1Output:
        """SETcc - Set byte on condition."""
        ops = list(instr.operands)
        if len(ops) != 1:
            return self._uncertain(instr, f"{instr.mnemonic} requires 1 operand")
        
        dst = ops[0]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.WRITE, "condition ? 1 : 0"),
            ],
        )
    
    def _sem_cmov(self, instr: Instruction) -> Level1Output:
        """CMOVcc - Conditional move."""
        ops = list(instr.operands)
        if len(ops) != 2:
            return self._uncertain(instr, f"{instr.mnemonic} requires 2 operands")
        
        dst, src = ops[0], ops[1]
        
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect(dst, EffectOperation.READ_WRITE, f"condition ? {src} : {dst}"),
                RegisterEffect(src, EffectOperation.READ) if self._is_register(src) else None,
            ],
        )
    
    def _sem_cpuid(self, instr: Instruction) -> Level1Output:
        """CPUID - CPU identification."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("eax", EffectOperation.READ_WRITE),
                RegisterEffect("ebx", EffectOperation.WRITE),
                RegisterEffect("ecx", EffectOperation.READ_WRITE),
                RegisterEffect("edx", EffectOperation.WRITE),
            ],
        )
    
    def _sem_rdtsc(self, instr: Instruction) -> Level1Output:
        """RDTSC - Read timestamp counter."""
        return Level1Output(
            instruction=instr,
            register_effects=[
                RegisterEffect("eax", EffectOperation.WRITE, "tsc_low"),
                RegisterEffect("edx", EffectOperation.WRITE, "tsc_high"),
            ],
        )
    
    # === Helpers ===
    
    def _uncertain(self, instr: Instruction, reason: str) -> Level1Output:
        """Return uncertain output."""
        return Level1Output(
            instruction=instr,
            is_uncertain=True,
            uncertainty_reason=reason,
            confidence=0.0,
        )
    
    def _is_register(self, operand: str) -> bool:
        """Check if operand is a register."""
        return operand.lower() in X86_64_REGISTERS
    
    def _is_memory(self, operand: str) -> bool:
        """Check if operand is a memory reference."""
        return "[" in operand or operand.startswith("ptr")
    
    def _parse_mem(self, operand: str) -> str:
        """Parse memory operand to address expression."""
        # Simple extraction of address from [addr] format
        if "[" in operand and "]" in operand:
            start = operand.index("[") + 1
            end = operand.index("]")
            return operand[start:end]
        return operand
    
    def _mem_size(self, operand: str) -> int:
        """Determine memory operand size."""
        op_lower = operand.lower()
        if "byte" in op_lower:
            return 1
        elif "word" in op_lower:
            return 2
        elif "dword" in op_lower:
            return 4
        elif "qword" in op_lower:
            return 8
        return 8  # Default to 64-bit
    
    def _arithmetic_flags(self) -> list[FlagEffect]:
        """Standard arithmetic flag effects."""
        return [
            FlagEffect("CF", FlagOperation.MODIFIED, "unsigned overflow"),
            FlagEffect("OF", FlagOperation.MODIFIED, "signed overflow"),
            FlagEffect("SF", FlagOperation.MODIFIED, "result < 0"),
            FlagEffect("ZF", FlagOperation.MODIFIED, "result == 0"),
            FlagEffect("AF", FlagOperation.MODIFIED, "aux carry"),
            FlagEffect("PF", FlagOperation.MODIFIED, "parity"),
        ]
