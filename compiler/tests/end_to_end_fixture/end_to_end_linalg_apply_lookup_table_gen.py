from platform import mac_ver
import numpy as np
import argparse


def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    for n_ct in args.n_ct:
        for p in args.bitwidth:
            for n_lut in args.n_lut:
                max_value = (2 ** p) - 1
                random_lut = np.random.randint(max_value+1, size=2**p)
                # identity_apply_lookup_table
                print(f"description: apply_lookup_table_{p}bits_{n_ct}ct_{n_lut}layer")
                print("program: |")
                print(
                    f"  func.func @main(%0: tensor<{n_ct}x!FHE.eint<{p}>>) -> tensor<{n_ct}x!FHE.eint<{p}>> {{")
                print(f"    %tlu = arith.constant dense<[{','.join(map(str, random_lut))}]> : tensor<{2**p}xi64>")
                for i in range(0, n_lut):
                    print(f"    %{i+1} = \"FHELinalg.apply_lookup_table\"(%{i}, %tlu):")
                    print(f"        (tensor<{n_ct}x!FHE.eint<{p}>>, tensor<{2**p}xi64>) -> (tensor<{n_ct}x!FHE.eint<{p}>>)")
                print(f"    return %{n_lut}: tensor<{n_ct}x!FHE.eint<{p}>>")
                print("  }")
                random_input = np.random.randint(max_value+1, size=n_ct)
                print("tests:")
                print("  - inputs:")
                print(f"    - tensor: [{','.join(map(str, random_input))}]")
                print(f"      shape: [{n_ct}]")
                outputs = random_input
                for i in range(0, n_lut):
                    outputs = [random_lut[v] for v in outputs]
                print("    outputs:")
                print(f"    - tensor: [{','.join(map(str, outputs))}]")
                print(f"      shape: [{n_ct}]")
                print("---")


CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--bitwidth",
    help="Specify the list of bitwidth to generate",
    nargs="+",
    type=int,
    default=list(range(1,16)),
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
generate(CLI.parse_args())
