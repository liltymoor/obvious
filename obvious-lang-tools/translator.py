from isa import Instruction, Opcode, ArgType, WORD_SIZE
from ast import AstExpr

class TranslateError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"TranslateError occured: {message}")


class Translator:
    def __init__(self) -> None:
        self.output_port = 0x0000001
        self.input_port = 0x0000010

        self.data_start = 0x2000000
        self.data_end = 0x2FFFFFF
        self.free_data = self.data_start

        self.mem_start = 0x3000000
        self.mem_end = 0x3FFFFFF
        self.mem_cur = 0x3000000
        self.mem_free_list: list[int] = []

        self.preload: list[Instruction] = []

        self.var_table: dict[str, int] = {}

    def allocate_data(self, size: int, val: list[int]) -> int:
        assert size >= len(val)

        if self.free_data + size >= self.data_end:
            raise TranslateError(message="Data memory exhausted")
        for i, v in enumerate(val):
            if v < 0 or v >= 2 ** (WORD_SIZE):
                raise TranslateError(message="Store immediate argument bigger than word")
            self.preload.append(Instruction(Opcode.LD, v, ArgType.IMMEDIATE))
            self.preload.append(Instruction(Opcode.ST, self.free_data + i, ArgType.IMMEDIATE))
        res = self.free_data
        self.free_data += size
        return res

    def _allocate_prog_mem(self) -> int:
        if self.mem_free_list:
            return self.mem_free_list.pop()
        if self.mem_cur < self.mem_end:
            self.mem_cur += 1
            return self.mem_cur - 1
        raise TranslateError(message="Program memory exhausted")

    def _free_prog_mem(self, ptr: int) -> None:
        self.mem_free_list.append(ptr)

    def allocate_for_tmp_expr(self) -> int:
        return self._allocate_prog_mem()

    def free_tmp_expr(self, ptr: int) -> None:
        self._free_prog_mem(ptr)

    def allocate_var(self, name: str) -> int:
        addr = self._allocate_prog_mem()
        self.var_table[name] = addr
        return addr

    def get_var_addr(self, name: str) -> int:
        return self.var_table[name]

    def translate(self, ast_expressions: list[AstExpr]) -> list[Instruction]:
        instructions: list[Instruction] = []
        for ast_expr in ast_expressions:
            instructions += ast_expr.translate(self)
        return instructions

