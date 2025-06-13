import struct
import sys

from tokenizer import Tokenizer
from ast_tools import AstBuilder, AstInterrupt, resolve_marks
from translator import Translator
from isa import Instruction, Opcode, ArgType

# with open(src, 'rb') as binary:
#     instructions = list()
#
#     serialized = binary.read()
#     offset = 0
#     while offset < len(serialized):
#         i, offset = Instruction.from_raw_instruction(serialized, offset)
#         instructions.append(i)

def build(src=type[str], target=type[str]):
    assert target.endswith(".bin"), "Target must end with .bin"
    with open(src, 'r') as text:
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
        for i_num, i in enumerate(code):
            print( f"{str(i_num):<2}: {str(i)}")

        with open(target, 'wb') as binary:
            binary.write(struct.pack('?', len(interrupt_instructions) > 0))
            for i in code:
                raw = i.get_raw_instruction()
                binary.write(raw)


if __name__ == '__main__':
    _, src, target = sys.argv
    build(src, target)