from __future__ import annotations

import enum

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
    REM = enum.auto()
    NEG = enum.auto()
    MOD = enum.auto()

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
class ArgType(enum.StrEnum):
    IMMEDIATE = "IMMEDIATE"
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"


class Instruction:
    def __init__(
        self, opcode: Opcode, arg: int | None = None, arg_type: ArgType = ArgType.DIRECT, line: int | None = None
    ) -> None:
        self.opcode = opcode
        self.arg = arg
        self.arg_type = arg_type
        self.line = line

    def __str__(self) -> str:
        return f"{self.opcode.name:<4} " + (
            f"{'[' + self.arg_type.name + ']':<11} {self.arg}" if self.arg is not None else ""
        )

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
