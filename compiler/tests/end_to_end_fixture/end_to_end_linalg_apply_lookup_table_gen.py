from platform import mac_ver
import numpy as np
import argparse


def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED THANKS THE end_to_end_levelled_gen.py scripts")
    np.random.seed(0)
    for n_ct in args.n_ct:
        for p in range(args.min_bitwidth, args.max_bitwidth+1):
            max_value = (2 ** p) - 1
            random_lut = np.random.randint(max_value+1, size=2**p)
            # identity_apply_lookup_table
            print(
                "description: apply_lookup_table_{0}bits_{1}ct".format(p, n_ct))
            print("program: |")
            print(
                "  func.func @main(%0: tensor<{1}x!FHE.eint<{0}>>) -> tensor<{1}x!FHE.eint<{0}>> {{".format(p, n_ct))
            print("    %tlu = arith.constant dense<[{0}]> : tensor<{1}xi64>".format(
                ','.join(map(str, random_lut)), 2**p))
            for i in range(0, args.n_lut):
                print(
                    "    %{4} = \"FHELinalg.apply_lookup_table\"(%{3}, %tlu): (tensor<{2}x!FHE.eint<{0}>>, tensor<{1}xi64>) -> (tensor<{2}x!FHE.eint<{0}>>)".format(p, 2**p, n_ct, i, i+1))
            print("    return %{2}: tensor<{1}x!FHE.eint<{0}>>".format(
                p, n_ct, args.n_lut))
            print("  }")
            random_input = np.random.randint(max_value+1, size=n_ct)
            print("tests:")
            print("  - inputs:")
            print(
                "    - tensor: [{0}]".format(','.join(map(str, random_input))))
            print("      shape: [{0}]".format(n_ct))
            outputs = random_input
            for i in range(0, args.n_lut):
                outputs = [random_lut[v] for v in outputs]

            print("    outputs:")
            print("    - tensor: [{0}]".format(','.join(map(str, outputs))))
            print("      shape: [{0}]".format(n_ct))
            print("---")


CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--min-bitwidth",
    type=int,
    default=1,
)
CLI.add_argument(
    "--max-bitwidth",
    type=int,
    default=16,
)
CLI.add_argument(
    "--n-ct",
    nargs="+",
    type=int,
    default=[1, 64, 128, 1024],
)
CLI.add_argument(
    "--n-lut",
    type=int,
    default=1,
)
generate(CLI.parse_args())
