from isa import Instruction, Opcode, ArgType, WORD_SIZE

class TranslateError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(f"TranslateError occured: {message}")

class Translator:
    def __init__(self) -> None:
        self.output_port = 0x0000001
        self.input_port = 0x0000010

        self.mem_start = 0x1000000
        self.mem_end = 0x2000000
        self.mem_cur = self.mem_start
        self.mem_free_list: list[int] = []

        self.preload: list[Instruction] = []

        self.var_table: dict[str, int] = {}

    def allocate_data(self, size: int, val: list[int]) -> int:

        assert size >= len(val)

        if self.mem_cur + size >= self.mem_end:
            raise TranslateError(message="Memory exhausted")
        for i, v in enumerate(val):
            if v < 0 or v >= 2 ** (WORD_SIZE):
                raise TranslateError(message="Store immediate argument bigger than word")
            self.preload.append(Instruction(Opcode.LD, v, ArgType.IMMEDIATE))
            self.preload.append(Instruction(Opcode.ST, self.mem_cur + i, ArgType.IMMEDIATE))
        res = self.mem_cur
        self.mem_cur += size
        return res

    def allocate_memory(self, size: int) -> int:
        if self.mem_cur + size >= self.mem_end:
            raise TranslateError(message="Memory exhausted")
        res = self.mem_cur
        self.mem_cur += size
        return res

    def _allocate_prog_mem(self) -> int:
        if self.mem_free_list:
            return self.mem_free_list.pop()
        if self.mem_cur < self.mem_end:
            self.mem_cur += 1
            return self.mem_cur - 1
        raise TranslateError(message="Memory exhausted")

    def _free_prog_mem(self, ptr: int) -> None:
        self.mem_free_list.append(ptr)

    def allocate_for_tmp_expr(self) -> int:
        return self._allocate_prog_mem()

    def free_tmp_expr(self, ptr: int) -> None:
        self._free_prog_mem(ptr)

    def allocate_var(self, name: str) -> int:
        if name in self.var_table.keys():
            return self.get_var_addr(name)
        addr = self._allocate_prog_mem()
        self.var_table[name] = addr
        return addr

    def get_var_addr(self, name: str) -> int:
        return self.var_table[name]