import argparse
import os
import struct

from isa import Instruction
from machinery import ControlUnit, DataPath, HaltReachedError


def parse_int_or_hex(value):
    try:
        if value.lower().startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        raise RuntimeError()


def run(target:str, verbose: bool=False, input_stream: list[tuple[int, str]] = []):
    assert str.endswith(target, ".bin"), "Runnable target must end with .bin"
    with open(target, "rb") as binary:
        instructions:list[Instruction] = list()
        data = binary.read()
        offset = 0

        can_be_interrupted = struct.unpack_from("?", data, offset)[0]
        offset += 1

        instructions = []
        while offset < len(data):
            i, offset = Instruction.from_raw_instruction(data, offset)
            instructions.append(i)

    dp = DataPath(2**25, instructions)
    cu = ControlUnit(
        dp,
        input_stream,
        can_be_interrupted=can_be_interrupted,
        collect_trace=verbose
    )

    iter_counter = 0
    try:
        while iter_counter <= 2**32:
            cu.run_next()
            iter_counter += 1
    except HaltReachedError:
        if verbose:
            os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
            with open(os.path.dirname(os.path.abspath(target)) + "/journal.txt", "w") as journal:
                for trace in cu.trace:
                    journal.write(f"{trace}\n")
                journal.write("Output format (TICK, DECODED TO INT OUT)\n")
                journal.write(",".join(cu.get_output()))
        print("Total ticks:", cu.tick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary runner for Obvious-binary files")
    parser.add_argument("target", help="Specify Obvious-binary target")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable trace logging")

    #TODO Realizr memory snapshot
    parser.add_argument("--memsnap", nargs=2,
                        type=parse_int_or_hex,
                        metavar=("HEX", "INT"),
                        help="Tuple of: first — hex (for example, 0x1a), second — int",
                        default=(-1, -1))

    args = parser.parse_args()
    run(args.target, args.verbose)
