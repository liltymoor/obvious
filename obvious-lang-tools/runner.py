import sys
from isa import Instruction
from machinery import ControlUnit, DataPath

def run(target:str):
    assert str.endswith(target, ".bin"), "Runnable target must end with .bin"
    with open(target, 'rb') as binary:
        instructions = list()

        serialized = binary.read()
        offset = 0
        while offset < len(serialized):
            i, offset = Instruction.from_raw_instruction(serialized, offset)
            instructions.append(i)
    #todo runner
    dp = DataPath(2**25, instructions)
    cu = ControlUnit(dp)

    iter_counter = 0
    while iter_counter <= 2**32:
        cu.run_next()
        iter_counter += 1


if __name__ == '__main__':
    _, target = sys.argv
    run(target)