import base64
import os
import struct
import sys

from ast_tools import AstBuilder, AstInterrupt, resolve_marks
from isa import ArgType, Instruction, Opcode
from tokenizer import Tokenizer
from translator import Translator


def build(src=type[str], target=type[str]):
    with open(src) as text:
        tokenizer = Tokenizer(text)
        builder = AstBuilder(tokenizer)
        translator = Translator()

        ast_tree = builder.build()
        interrupt_instructions = list()
        instructions = list()

        for ast_node in ast_tree:
            if isinstance(ast_node, AstInterrupt):
                interrupt_instructions = ast_node.translate(translator)
                continue
            instructions += ast_node.translate(translator)

        instructions = translator.preload + instructions
        instructions = resolve_marks(interrupt_instructions + instructions)
        instructions.append(Instruction(Opcode.HALT, None, ArgType.IMMEDIATE))

        code = instructions
        print("LoC:", tokenizer.line)
        print("Instructions:", len(instructions))

        # Убедимся, что каталог назначения существует
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)

        with open(target + ".bin", "wb") as binary:
            binary.write(struct.pack("?", len(interrupt_instructions) > 0))
            for i in code:
                raw = i.get_raw_instruction()
                binary.write(raw)

        with open(target + ".hex", "w") as target_hex:
            for k,v in enumerate(code):
                target_hex.write(f"{hex(k * 4):4} - {hex(v.get_coded()):11} - {v.opcode.name:5} {hex(v.arg * 4) if v.arg is not None else "":10} | [{v.arg_type.name}]\n")

        with open(target + ".base64", "w") as f:
            f.write(base64.b64encode(struct.pack("?", len(interrupt_instructions) > 0)).decode("utf-8"))
            for i in code:
                raw = i.get_raw_instruction()
                f.write(base64.b64encode(raw).decode("utf-8"))

if __name__ == "__main__":
    _, src, target = sys.argv
    build(src, target)
