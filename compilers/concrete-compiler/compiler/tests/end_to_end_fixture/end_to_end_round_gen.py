import argparse
from platform import mac_ver

import numpy as np

from end_to_end_linalg_leveled_gen import P_ERROR


def round(val, p_start, p_end, signed=False):
    p_delta = p_start - p_end
    carry_mask = 1 << (p_delta - 1)
    if val & carry_mask != 0:
        val += carry_mask << 1
    output = val >> p_delta
    if signed:
        if output >= (1 << (p_end - 1)):
            output = -output
    return output


def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    # unsigned_unsigned
    for from_p in args.bitwidth:
        for to_p in range(2, from_p):
            max_value = (2 ** from_p) - 1
            print(f"description: unsigned_round_{from_p}to{to_p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: !FHE.eint<{from_p}>) -> !FHE.eint<{to_p}> {{")
            print(f"    %1 = \"FHE.round\"(%arg0) : (!FHE.eint<{from_p}>) -> !FHE.eint<{to_p}>")
            print(f"    return %1: !FHE.eint<{to_p}>")
            print("  }")
            print(f"p-error: {P_ERROR}")
            print("tests:")
            for i in range(8):
                val = np.random.randint(max_value)
                print("  - inputs:")
                print(f"    - scalar: {val}")
                print("    outputs:")
                print(f"    - scalar: {round(val, from_p, to_p)}")
            print("---")
    # signed_signed
    for from_p in args.bitwidth:
        for to_p in range(2, from_p):
            min_value = -(2 ** (from_p - 1))
            max_value = abs(min_value) - 1
            print(f"description: signed_round_from_{from_p}to{to_p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: !FHE.esint<{from_p}>) -> !FHE.esint<{to_p}> {{")
            print(f"    %1 = \"FHE.round\"(%arg0) : (!FHE.esint<{from_p}>) -> !FHE.esint<{to_p}>")
            print(f"    return %1: !FHE.esint<{to_p}>")
            print("  }")
            print(f"p-error: {P_ERROR}")
            print("tests:")
            for i in range(8):
                val = np.random.randint(min_value, max_value)
                print("  - inputs:")
                print(f"    - scalar: {val}")
                print(f"      signed: true")
                print("    outputs:")
                print(f"    - scalar: {round(val, from_p, to_p, True)}")
                print(f"      signed: true")
            print("---")

if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--bitwidth",
        help="Specify the list of bitwidth to generate",
        nargs="+",
        type=int,
        default=list(range(3,9)),
    )
    generate(CLI.parse_args())
