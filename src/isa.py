from __future__ import annotations

import enum
import struct


class OpcodeOverflowError(Exception):
    def __init__(self, opcode):
        super().__init__(f"Opcode {hex(opcode)} must fit in 5 bits")

class ArgTypeOverflowError(Exception):
    def __init__(self, arg):
        super().__init__(f"Arg {hex(arg)} must fit in 2 bits")

WORD_SIZE = 32

@enum.unique
class Opcode(enum.Enum):
    # isa length 24
    LD = enum.auto()
    ST = enum.auto()

    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    MOD = enum.auto()

    NEG = enum.auto()

    # bitwise logical operations
    AND = enum.auto()
    OR = enum.auto()

    # logical not
    NOT = enum.auto()

    CMP = enum.auto()

    SETG = enum.auto()
    SETE = enum.auto()
    SETL = enum.auto()

    JMP = enum.auto()

    JZ = enum.auto()
    JNZ = enum.auto()

    HALT = enum.auto()

    IN = enum.auto() # pass in reg to acc
    OUT = enum.auto() # pass acc to out reg

    IRET = enum.auto()
    ILOCK = enum.auto()


@enum.unique
class ArgType(enum.Enum):
    IMMEDIATE = 0
    DIRECT = 1
    INDIRECT = 2


class Instruction:
    def __init__(
        self, opcode: Opcode, arg: int | None = None, arg_type: ArgType = ArgType.DIRECT) -> None:
        self.opcode = opcode
        self.arg = arg
        self.arg_type = arg_type

    def __str__(self) -> str:
        return f"{self.opcode.name:<4} " + (
            f"{'[' + self.arg_type.name + ']':<11} {hex(self.arg)}" if self.arg is not None else ""
        )

    def get_raw_instruction(self) -> bytes:
        #B (unsigned char), b (signed char), ? (bool), i (int32)
        opcode_value = self.opcode.value
        arg_type_value = self.arg_type.value
        has_arg = 0 if self.arg is None else 1

        data = struct.pack("BBB", opcode_value, arg_type_value, has_arg)

        if has_arg:
            data += struct.pack("i", self.arg)

        return data

    @staticmethod
    def from_raw_instruction(data:bytes, offset:int = 0) -> tuple[Instruction, int]:
        # returns the actual instruction and new offset
        opcode_value, arg_type_value, has_arg = struct.unpack_from("BBB", data, offset)
        offset += 3

        if has_arg:
            arg = struct.unpack_from("i", data, offset)[0]
            offset += 4
            return Instruction(Opcode(opcode_value), arg, ArgType(arg_type_value)), offset
        return Instruction(Opcode(opcode_value), None, ArgType.IMMEDIATE), offset

    @staticmethod
    def parse(line: str) -> Instruction:
        arr = line.split()
        try:
            if len(arr) == 1:
                return Instruction(Opcode[arr[0]])
            if len(arr) == 3:
                return Instruction(Opcode[arr[0]], int(arr[2]), ArgType[arr[1][1:-1]])
        except Exception as e:
            raise ValueError() from e
        raise ValueError()

    def get_coded(self) -> int:
        if not (0 <= self.opcode.value < (1 << 5)):
            raise OpcodeOverflowError(self.opcode)

        if not (0 <= self.arg_type.value < (1 << 2)):
            raise ArgTypeOverflowError(self.arg_type.value)

        result = (self.opcode.value & 0x1F) << 27  # 5 бит opcode
        result |= (self.arg_type.value & 0x03) << 25  # 2 бита arg_type
        if self.arg is not None:
            result |= (self.arg & 0x1FFFFFF)  # 25 бит аргумента
        return result


class BranchMark:
    def __init__(self) -> None:
        self.position:None|int = None

    def set_position(self, position: int) -> None:
        self.position = position


class MarkedInstruction:
    def __init__(self, instruction: Instruction, mark: BranchMark) -> None:
        self.instruction = instruction
        self.mark = mark

    def __str__(self) -> str:
        return str(self.instruction) + " | MARKED to " + str(self.mark.position)

    def get_raw_instruction(self) -> bytes:
        return self.instruction.get_raw_instruction()

def init(ptr: int, arg_type: ArgType, value: int = 0) -> list[Instruction]:
    instructions = []
    instructions.append(Instruction(Opcode.LD, value, ArgType.IMMEDIATE))
    instructions.append(Instruction(Opcode.ST, ptr, arg_type))
    return instructions

def inc(ptr: int, arg_type: ArgType) -> list[Instruction]:
    instructions = []
    instructions.append(Instruction(Opcode.LD, ptr, arg_type))
    instructions.append(Instruction(Opcode.ADD, 1, ArgType.IMMEDIATE))
    if arg_type != arg_type.IMMEDIATE:
        arg_type = ArgType(arg_type.value - 1)
    instructions.append(Instruction(Opcode.ST, ptr, arg_type))
    return instructions

def dec(ptr: int, arg_type: ArgType) -> list[Instruction]:
    instructions = []
    instructions.append(Instruction(Opcode.LD, ptr, arg_type))
    instructions.append(Instruction(Opcode.SUB, 1, ArgType.IMMEDIATE))
    if arg_type != arg_type.IMMEDIATE:
        arg_type = ArgType(arg_type.value - 1)
    instructions.append(Instruction(Opcode.ST, ptr, arg_type))
    return instructions
