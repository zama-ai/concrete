from platform import mac_ver
import numpy as np
import argparse


MIN_PRECISON = 1
MAX_PRECISION = 16

PRECISION_FORCE_CRT = 9

def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    for p in args.bitwidth:
        max_value = (2 ** p) - 1
        random_lut = np.random.randint(max_value+1, size=2**p)
        print(f"description: apply_lookup_table_{p}bits")
        print("program: |")
        print(
            f"  func.func @main(%arg0: !FHE.eint<{p}>) -> !FHE.eint<{p}> {{")
        print(f"    %tlu = arith.constant dense<[{','.join(map(str, random_lut))}]> : tensor<{2**p}xi64>")
        print(
            f"    %1 = \"FHE.apply_lookup_table\"(%arg0, %tlu): (!FHE.eint<{p}>, tensor<{2**p}xi64>) -> (!FHE.eint<{p}>)")
        print(f"    return %1: !FHE.eint<{p}>")
        print("  }")
        if p >= PRECISION_FORCE_CRT:
            print("encoding: crt")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: 0")
        print("    outputs:")
        print(f"    - scalar: {random_lut[0]}")
        print("  - inputs:")
        random_i = np.random.randint(max_value)
        print(f"    - scalar: {random_i}")
        print("    outputs:")
        print(f"    - scalar: {random_lut[random_i]}")
        print("  - inputs:")
        print(f"    - scalar: {max_value}")
        print("    outputs:")
        print(f"    - scalar: {random_lut[max_value]}")
        print("---")

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--bitwidth",
    help="Specify the list of bitwidth to generate",
    nargs="+",
    type=int,
    default=list(range(1,16)),
)
generate(CLI.parse_args())
