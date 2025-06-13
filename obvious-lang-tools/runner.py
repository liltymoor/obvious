import struct
import sys
from isa import Instruction
from machinery import ControlUnit, DataPath

def run(target:str):
    assert str.endswith(target, ".bin"), "Runnable target must end with .bin"
    with open(target, 'rb') as binary:
        instructions = list()
        data = binary.read()
        offset = 0

        can_be_interrupted = struct.unpack_from('?', data, offset)[0]
        offset += 1

        instructions = []
        while offset < len(data):
            i, offset = Instruction.from_raw_instruction(data, offset)
            instructions.append(i)

    dp = DataPath(2**25, instructions)
    cu = ControlUnit(dp, [(1, 'h'), (10, 'e'), (20, 'l'), (25, 'l'), (103, 'o')], can_be_interrupted=can_be_interrupted)

    iter_counter = 0
    while iter_counter <= 2**32:
        cu.run_next()
        iter_counter += 1


if __name__ == '__main__':
    _, target = sys.argv
    run(target)