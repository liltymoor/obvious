import enum

from isa import ArgType, Instruction, Opcode


class HaltReachedError(Exception):
    pass

class UnexpectedOpcodeError(Exception):
    def __init__(self, opcode):
        super().__init__(f"Unexpected opcode {opcode}")

class ALU:
    def __init__(self) -> None:
        self.N = False
        self.Z = True
        self.result = 0

    def compute(self, op: Opcode, lhs: int, rhs: int) -> None:
        operations = {
            Opcode.ADD: lambda: lhs + rhs,
            Opcode.SUB: lambda: lhs - rhs,
            Opcode.MUL: lambda: rhs * lhs,
            Opcode.DIV: lambda: lhs // rhs,
            Opcode.MOD: lambda: lhs % rhs,
            Opcode.CMP: lambda: lhs - rhs,
            Opcode.SETE: lambda: int(self.Z == 1 and self.N == 0),
            Opcode.SETG: lambda: int(self.Z == 0 and self.N == 0),
            Opcode.SETL: lambda: int(self.Z == 0 and self.N == 1),
            Opcode.AND: lambda: lhs & rhs,
            Opcode.OR: lambda: lhs | rhs,
            Opcode.NOT: lambda: int(lhs == 0),
            Opcode.NEG: lambda: -lhs
        }

        self.result = operations[op]()
        self.set_flags(self.result)

    def set_flags(self, result) -> None:
        if result == 0:
            self.Z = True
        else:
            self.Z = False

        if result < 0:
            self.N = True
        else:
            self.N = False

    def get_flags(self) -> tuple[bool, bool]:
        return self.N, self.Z

class MUX(enum.Enum):
    RHS_FROM_DR = enum.auto()
    RHS_FROM_CR = enum.auto()

    AR_FROM_DR = enum.auto()
    AR_FROM_CR = enum.auto()

    PC_ONE = enum.auto()
    PC_FROM_BR = enum.auto()
    PC_FROM_CR = enum.auto()
    PC_FROM_DR = enum.auto()
    PC_INC = enum.auto()

    ACC_FROM_IN = enum.auto()
    ACC_FROM_ALU = enum.auto()
    ACC_FROM_DR = enum.auto()
    ACC_FROM_CR = enum.auto()

    RAM_R_FROM_PC = enum.auto()
    RAM_R_FROM_AR = enum.auto()

class DataPath:
    def __init__(self, mem_size: int, program: list[Instruction]):
        self.mem_size = mem_size
        self.acc = 0
        self.dr = 0
        self.cr = 0
        self.ar = 0
        self.br = 0
        self.pc = 0

        self.input = ""
        self.output = ""

        self.left_op = 0
        self.right_op = 0
        self.memory = [0] * mem_size

        for k, v in enumerate(program):
            self.memory[k] = v.get_coded()

        self.alu = ALU()

    def latch_dr(self, path: MUX) -> None:
        self.dr = self.oe(path)

    def latch_cr(self, path: MUX) -> None:
        self.cr = self.oe(path)

    def get_cr_opcode(self):
        return (self.cr & 0xFC000000) >> 27

    def get_cr_optype(self):
        return (self.cr & 0x06000000) >> 25

    def latch_br(self) -> None:
        self.br = self.pc

    def latch_pc(self, path: MUX) -> None:
        if path is MUX.PC_ONE:
            self.pc = 1
        if path is MUX.PC_INC:
            self.pc += 1
        if path is MUX.PC_FROM_BR:
            self.pc = self.br
        if path is MUX.PC_FROM_CR:
            self.pc = self.cr & 0x01FFFFFF

    def latch_ar(self, path: MUX) -> None:
        if path is MUX.AR_FROM_CR:
            self.ar = self.cr & 0x01FFFFFF
            return
        if path is MUX.AR_FROM_DR:
            self.ar = self.dr
            return

    def latch_acc(self, path: MUX) -> None:
        if path is MUX.ACC_FROM_IN:
            self.acc = ord(self.input)
            return
        if path is MUX.ACC_FROM_ALU:
            self.acc = self.alu.result
            return
        if path is MUX.ACC_FROM_CR:
            self.acc = self.cr & 0x01FFFFFF
            return
        if path is MUX.ACC_FROM_DR:
            self.acc = self.dr

    def latch_alu_lhs(self) -> None:
        self.left_op = self.acc

    def latch_alu_rhs(self, path: MUX) -> None:
        if path is MUX.RHS_FROM_DR:
            self.right_op = self.dr
            return

        if path is MUX.RHS_FROM_CR:
            self.right_op = self.cr & 0x01FFFFFF
            return

    def oe(self, path: MUX) -> int:
        if path is MUX.RAM_R_FROM_AR:
            return self.memory[self.ar]
        if path is MUX.RAM_R_FROM_PC:
            return self.memory[self.pc]
        return 0

    def wr(self) -> None:
        self.memory[self.ar] = self.acc

    def latch_out(self):
        self.output = self.acc

    def snapshot_mem(self, start_addr: int, snap_size: int = 16, decode: str = "hex") -> str:
        mem_data = []
        for i in range(snap_size):
            mem_data.append(self.memory[start_addr + i])
        if decode == "ascii":
            mapped_mem_data = map(chr, mem_data)
            return f"{hex(start_addr)}: [{",".join(mapped_mem_data)}]"
        mapped_mem_data = map(hex, mem_data)
        return f"{hex(start_addr)}: [{",".join(mapped_mem_data)}]"

class ControlUnit:
    def __init__(self, data_path: DataPath,
                 in_queue: list[tuple[int, str]],
                 can_be_interrupted: bool = False,
                 collect_trace: bool = False,
                 ) -> None:
        self.dp = data_path
        self.tick = 0

        self.input = in_queue
        self.output:list[tuple[int, str]] = []

        self.interrupted = False
        self.ilock = not can_be_interrupted

        self.collect_trace = collect_trace
        self.trace:list[str] = []

    def pass_output(self):
        self.output.append((self.tick, int(self.dp.output)))

    def instr_fetch(self) -> tuple[Opcode, int]:
        self.dp.latch_cr(MUX.RAM_R_FROM_PC) # CR = RAM[PC]
        self.dp.latch_pc(MUX.PC_INC)
        self.next_tick()
        return Opcode(self.dp.get_cr_opcode()), self.dp.get_cr_optype()


    def operand_fetch(self, argtype: ArgType):
        if argtype is ArgType.IMMEDIATE:
            return
        if argtype is ArgType.DIRECT:
            self.dp.latch_ar(MUX.AR_FROM_CR)
            self.dp.latch_dr(MUX.RAM_R_FROM_AR) # DR = RAM[AR = CR[7:32]]
            self.next_tick()
            return
        if argtype is ArgType.INDIRECT:
            self.dp.latch_ar(MUX.AR_FROM_CR)
            self.dp.latch_dr(MUX.RAM_R_FROM_AR) # DR = RAM[AR = CR[7:32]]
            self.next_tick()
            self.dp.latch_ar(MUX.AR_FROM_DR)
            self.dp.latch_dr(MUX.RAM_R_FROM_AR) # DR = RAM[AR = DR]
            self.next_tick()
            return

    def execute(self, opcode, arg_type: ArgType): # noqa: C901
        match opcode:
            case Opcode.ADD | Opcode.SUB | Opcode.MUL | Opcode.DIV | Opcode.MOD | Opcode.AND | Opcode.OR:
                self._execute_binary_op(opcode, arg_type)
            case Opcode.LD:
                self._execute_load(arg_type)
            case Opcode.ST:
                self._execute_store(arg_type)
            case Opcode.OUT:
                self.dp.latch_out()
                self.pass_output()
            case Opcode.IN:
                self.dp.latch_acc(MUX.ACC_FROM_IN)
            case Opcode.JNZ | Opcode.JZ | Opcode.JMP:
                self._execute_jump(opcode, arg_type)
            case Opcode.SETG | Opcode.SETL | Opcode.SETE:
                self._execute_set_flags(opcode)
            case Opcode.NEG | Opcode.NOT:
                self._execute_unary_op(opcode)
            case Opcode.CMP:
                self._execute_compare(arg_type)
            case Opcode.IRET:
                self._execute_iret()
            case Opcode.ILOCK:
                self.ilock = True
            case Opcode.HALT:
                raise HaltReachedError
            case _:
                raise UnexpectedOpcodeError(opcode)
        self.next_tick()

    def _get_path_for_arg_type(self, arg_type, immediate_path, direct_path):
        return immediate_path if arg_type is ArgType.IMMEDIATE else direct_path

    def _execute_binary_op(self, opcode, arg_type):
        path = self._get_path_for_arg_type(arg_type, MUX.RHS_FROM_CR, MUX.RHS_FROM_DR)
        self.dp.latch_alu_lhs()
        self.dp.latch_alu_rhs(path)
        self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
        self.dp.latch_acc(MUX.ACC_FROM_ALU)

    def _execute_load(self, arg_type):
        path = self._get_path_for_arg_type(arg_type, MUX.ACC_FROM_CR, MUX.ACC_FROM_DR)
        self.dp.latch_acc(path)
        self.dp.alu.set_flags(self.dp.acc)

    def _execute_store(self, arg_type):
        path = self._get_path_for_arg_type(arg_type, MUX.AR_FROM_CR, MUX.AR_FROM_DR)
        self.dp.latch_ar(path)
        self.dp.wr()

    def _execute_jump(self, opcode, arg_type):
        n, z = self.dp.alu.get_flags()
        should_jump = (opcode == Opcode.JMP or
                       (opcode == Opcode.JNZ and not z) or
                       (opcode == Opcode.JZ and z))
        if should_jump:
            path = self._get_path_for_arg_type(arg_type, MUX.PC_FROM_CR, MUX.PC_FROM_DR)
            self.dp.latch_pc(path)

    def _execute_set_flags(self, opcode):
        self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
        self.dp.latch_acc(MUX.ACC_FROM_ALU)

    def _execute_unary_op(self, opcode):
        self.dp.latch_alu_lhs()
        self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
        self.dp.latch_acc(MUX.ACC_FROM_ALU)

    def _execute_compare(self, arg_type):
        path = self._get_path_for_arg_type(arg_type, MUX.RHS_FROM_CR, MUX.RHS_FROM_DR)
        self.dp.latch_alu_lhs()
        self.dp.latch_alu_rhs(path)
        self.dp.alu.compute(Opcode.CMP, self.dp.left_op, self.dp.right_op)

    def _execute_iret(self):
        self.ilock = False
        self.interrupted = False
        self.dp.latch_pc(MUX.PC_FROM_BR)

    def icheck(self):
        if len(self.input) == 0 or self.ilock:
            return False

        trigger_tick, selected_input = self.input[0]
        if trigger_tick <= self.tick:
            return True
        return False

    def interrupt(self):
        trigger_tick, selected_input = self.input[0]
        self.input = self.input[1:]
        self.interrupted = True

        self.dp.input = selected_input
        self.dp.latch_br()

        self.dp.latch_pc(MUX.PC_ONE)

    def run_next(self) -> None:
        opc, opt = self.instr_fetch()
        arg_type = ArgType(opt)

        non_addr_command = [
            Opcode.IN,
            Opcode.SETL,
            Opcode.SETE,
            Opcode.SETG,
            Opcode.OUT,
            Opcode.NOT,
            Opcode.NEG,
            Opcode.HALT,
        ]
        if opc not in non_addr_command:
            self.operand_fetch(arg_type)

        if self.collect_trace:
            self.trace.append(self.snapshot_cu())

        self.execute(opc, arg_type)
        if self.icheck():
            self.interrupt()

    def next_tick(self):
        self.tick += 1

    def snapshot_cu(self) -> str:
        return (F"TICK{"(I)" if self.interrupted else ""}: {self.tick:8} |"
                F" Command: {Opcode(self.dp.get_cr_opcode()):13} |"
                F" PC: {self.dp.pc:4} |"
                F" CR: {hex(self.dp.cr):15}"
                F" ACC: {hex(self.dp.acc):10} |"
                F" DR: {hex(self.dp.dr):10} |"
                F" AR: {hex(self.dp.ar):10}"
                F" FLAGS: {self.dp.alu.get_flags()!s:4}")

    def memory_snapshot(self, addr: int, snap_sz: int, decode: str = "ascii") -> str:
        return self.dp.snapshot_mem(addr, decode=decode, snap_size=snap_sz)

    def get_output(self):
        return self.output if len(self.output) > 0 else []


