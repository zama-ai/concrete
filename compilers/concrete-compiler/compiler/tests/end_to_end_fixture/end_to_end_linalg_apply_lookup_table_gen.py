import argparse
from platform import mac_ver

import numpy as np

from end_to_end_linalg_leveled_gen import P_ERROR

PRECISION_FORCE_CRT = 9

def get_lut_integer_type(p):
    if p <= 8:
        return "i8"
    if p <= 16:
        return "i16"
    if p <= 32:
        return "i32"
    if p <= 64:
        return "i64"
    else:
        raise Exception("Unexpected precision") 

def generate(args):
    print("# /!\\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    for n_ct in args.n_ct:
        for p in args.bitwidth:
            for n_lut in args.n_lut:
                max_value = (2 ** p) - 1
                random_lut = np.random.randint(max_value+1, size=2**p)
                itype = get_lut_integer_type(p)
                iprec = itype.replace("i", "")
                # identity_apply_lookup_table
                print(f"description: apply_lookup_table_{p}bits_{n_ct}ct_{n_lut}layer")
                print("program: |")
                print(
                    f"  func.func @main(%0: tensor<{n_ct}x!FHE.eint<{p}>>, %tlu: tensor<{2**p}x{itype}>) -> tensor<{n_ct}x!FHE.eint<{p}>> {{")
                for i in range(0, n_lut):
                    print(f"    %{i+1} = \"FHELinalg.apply_lookup_table\"(%{i}, %tlu):")
                    print(f"        (tensor<{n_ct}x!FHE.eint<{p}>>, tensor<{2**p}x{itype}>) -> (tensor<{n_ct}x!FHE.eint<{p}>>)")
                print(f"    return %{n_lut}: tensor<{n_ct}x!FHE.eint<{p}>>")
                print("  }")
                if p >= PRECISION_FORCE_CRT:
                    print("encoding: crt")
                print(f"p-error: {P_ERROR}")
                random_input = np.random.randint(max_value+1, size=n_ct)
                print("tests:")
                print("  - inputs:")
                print(f"    - tensor: [{','.join(map(str, random_input))}]")
                print(f"      shape: [{n_ct}]")
                print(f"    - tensor: [{','.join(map(str, random_lut))}]")
                print(f"      shape: [{2**p}]")
                print(f"      width: {iprec}")
                outputs = random_input
                for i in range(0, n_lut):
                    outputs = [random_lut[v] for v in outputs]
                print("    outputs:")
                print(f"    - tensor: [{','.join(map(str, outputs))}]")
                print(f"      shape: [{n_ct}]")
                print("---")


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--bitwidth",
        help="Specify the list of bitwidth to generate",
        nargs="+",
        type=int,
        default=list(range(1,17)),
    )
    CLI.add_argument(
        "--n-ct",
        help="Specify the tensor sizes to generate",
        nargs="+",
        type=int,
        default=[4],
    )
    CLI.add_argument(
        "--n-lut",
        help="Specify the number of FHELinalg.apply_lookup_table layers to generate",
        nargs="+",
        type=int,
        default=[1,2],
    )
    CLI.add_argument(
        "--minimal",
        help="Specify whether to generate minimal tests only",
        type=bool,
        default=False,
    )
    generate(CLI.parse_args())
