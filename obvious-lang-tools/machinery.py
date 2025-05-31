import enum
from isa import Opcode, Instruction, ArgType

class HaltReached(Exception):
    pass

class ALU:
    def __init__(self) -> None:
        self.N = False
        self.Z = True
        self.result = 0

    def compute(self, op: Opcode, lhs: int, rhs: int) -> None:
        result = 0
        match op:
            case Opcode.ADD:
                result = lhs + rhs
            case Opcode.SUB:
                result = lhs - rhs
            case Opcode.MUL:
                result = rhs * lhs
            case Opcode.DIV:
                result = lhs // rhs
            case Opcode.MOD:
                result = lhs % rhs
            case Opcode.CMP:
                result = lhs - rhs
            case Opcode.SETE:
                if self.Z == 1 and self.N == 0:
                    result = 1
                else:
                    result = 0
            case Opcode.SETG:
                if self.Z == 0 and self.N == 0:
                    result = 1
                else:
                    result = 0
            case Opcode.SETL:
                if self.Z == 0 and self.N == 1:
                    result = 1
                else:
                    result = 0
            case Opcode.AND:
                result = lhs & rhs
            case Opcode.OR:
                result = lhs | rhs
            case Opcode.NOT:
                if lhs == 0:
                    result = 1
                if lhs != 0:
                    result = 0
            case Opcode.NEG:
                result = -lhs
        self.result = result
        self.set_flags(result)

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

    PC_ZERO = enum.auto()
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
    def __init__(self, mem_size: int):
        self.mem_size = mem_size
        self.acc = 0
        self.dr = 0
        self.cr = 0
        self.ar = 0
        self.br = 0
        self.pc = 0

        self.input = 0
        self.output = 0

        self.left_op = 0
        self.right_op = 0

        self.memory = [0] * mem_size
        self.alu = ALU()

    def latch_dr(self, path: MUX) -> None:
        self.dr = self.oe(path)

    def latch_cr(self, path: MUX) -> None:
        self.cr = self.oe(path)

    def get_cr_opcode(self):
        return self.cr & 0xFC000000

    def get_cr_optype(self):
        return self.cr & 0x03000000

    def latch_br(self) -> None:
        self.br = self.pc

    def latch_pc(self, path: MUX) -> None:
        if path is MUX.PC_ZERO:
            self.pc = 0
        if path is MUX.PC_INC:
            self.pc += 1
        if path is MUX.PC_FROM_BR:
            self.pc = self.br

    def latch_ar(self, path: MUX) -> None:
        if path is MUX.AR_FROM_CR:
            self.ar = self.cr & 0x01FFFFFF
            return
        if path is MUX.AR_FROM_DR:
            self.ar = self.dr
            return

    def latch_acc(self, path: MUX) -> None:
        if path is MUX.ACC_FROM_IN:
            self.acc = self.input
            return
        if path is MUX.ACC_FROM_ALU:
            self.acc = self.alu.result
            return

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
        self.dr = self.memory[self.ar]

    def latch_out(self):
        self.output = self.acc

class ControlUnit:
    def __init__(self, translated_program: list[Instruction], data_path: DataPath):
        # must be inserted into mem
        self.code = translated_program
        self.dp = data_path
        self.tick = 0

    def instr_fetch(self) -> tuple[int, int]:
        self.dp.latch_cr(MUX.RAM_R_FROM_PC) # CR = RAM[PC]
        self.dp.latch_pc(MUX.PC_INC)
        self.next_tick()
        return self.dp.get_cr_opcode(), self.dp.get_cr_optype()

    def operand_fetch(self, argtype: ArgType):
        if argtype is ArgType.IMMEDIATE:
            #self.dp.latch_alu_rhs(MUX.RHS_FROM_CR) # ALU_RHS = CR[7:32]
            return
        if argtype is ArgType.DIRECT:
            self.dp.latch_dr(MUX.AR_FROM_CR) # DR = RAM[AR = CR[7:32]]
            #self.dp.latch_alu_rhs(MUX.RHS_FROM_DR) # ALU_RHS = DR
            self.next_tick()
            return
        if argtype is ArgType.INDIRECT:
            self.dp.latch_dr(MUX.RHS_FROM_CR) # DR = RAM[AR = CR[7:32]]
            self.next_tick()
            self.dp.latch_dr(MUX.RHS_FROM_DR) # DR = RAM[AR = DR]
            self.next_tick()
            #self.dp.latch_alu_rhs(MUX.RHS_FROM_DR)
            return

    def execute(self, opcode, arg_type: ArgType):
        match opcode:
            case Opcode.ADD | Opcode.SUB | Opcode.MUL | Opcode.DIV | Opcode.MOD | Opcode.AND | Opcode.OR | Opcode.MOD:
                if arg_type is ArgType.IMMEDIATE:
                    path = MUX.RHS_FROM_CR
                else:
                    path = MUX.RHS_FROM_DR
                self.dp.latch_alu_lhs()
                self.dp.latch_alu_rhs(path)
                self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
                self.dp.latch_acc(MUX.ACC_FROM_ALU)
            case Opcode.LD:
                if arg_type is ArgType.IMMEDIATE:
                    path = MUX.ACC_FROM_CR
                else:
                    path = MUX.ACC_FROM_DR
                self.dp.latch_acc(path)
            case Opcode.ST:
                if arg_type is ArgType.IMMEDIATE:
                    path = MUX.AR_FROM_CR
                else:
                    path = MUX.AR_FROM_DR
                self.dp.latch_ar(path)
                self.dp.wr()
            case Opcode.OUT:
                self.dp.latch_out()
            case Opcode.IN:
                self.dp.latch_acc(MUX.ACC_FROM_IN)
            case Opcode.JNZ | Opcode.JZ | Opcode.JMP:
                if arg_type is ArgType.IMMEDIATE:
                    path = MUX.PC_FROM_CR
                else:
                    path = MUX.PC_FROM_DR
                N, Z = self.dp.alu.get_flags()
                if opcode == Opcode.JNZ and not Z:
                    self.dp.latch_pc(path)
                if opcode == Opcode.JZ and Z:
                    self.dp.latch_pc(path)
                if opcode == Opcode.JMP:
                    self.dp.latch_pc(path)
            case Opcode.SETG | Opcode.SETL | Opcode.SETE:
                self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
                self.dp.latch_acc(MUX.ACC_FROM_ALU)
            case Opcode.NEG | Opcode.NOT:
                self.dp.latch_alu_lhs()
                self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
                self.dp.latch_acc(MUX.ACC_FROM_ALU)
            case Opcode.CMP:
                if arg_type is ArgType.IMMEDIATE:
                    path = MUX.RHS_FROM_CR
                else:
                    path = MUX.RHS_FROM_DR
                self.dp.latch_alu_lhs()
                self.dp.latch_alu_rhs(path)
                self.dp.alu.compute(opcode, self.dp.left_op, self.dp.right_op)
            case Opcode.HALT:
                raise HaltReached
            case _:
                raise RuntimeError(f"Invalid opcode {opcode}")
        self.next_tick()

    def interrupt(self):
        pass

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

        self.execute(opc, arg_type)


    def next_tick(self):
        self.tick += 1

    def snapshot_cu(self) -> str:
        return "TICK: {:5} | Command: {:10} | IP {:10} | ACC: {:10} | DR {: 5} | PC: {:3} | AR: {:10}".format(
            self.tick,
            self.dp.get_cr_opcode(),
            self.dp.pc,
            self.dp.acc,
            self.dp.dr,
            self.dp.pc,
            self.dp.ar,
        )


