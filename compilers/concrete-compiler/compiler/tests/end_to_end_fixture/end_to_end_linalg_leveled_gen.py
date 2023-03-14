import argparse
import numpy as np

PRECISIONS_TO_BENCH = [1, 2, 5, 8, 9, 12, 16, 24, 32, 40, 48, 56]
N_CT = [100, 1000, 100000]
P_ERROR = 1.0 - 0.999936657516


def main():
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    for p in PRECISIONS_TO_BENCH:
        for n_ct in N_CT:
            max_value = (2 ** p) - 1
            integer_bitwidth = p + 1
            random_cst = np.random.randint(max_value+1, size=n_ct)
            random_input = np.random.randint(max_value+1, size=n_ct)
            # add_eint_int_cst
            print(
                "description: add_eint_int_cst_{0}bits_{1}ct".format(p, n_ct))
            print("program: |")
            print(
                "  func.func @main(%arg0: tensor<{1}x!FHE.eint<{0}>>) -> tensor<{1}x!FHE.eint<{0}>> {{".format(p, n_ct))
            print("    %0 = arith.constant dense<[{0}]> : tensor<{1}xi{2}>".format(
                ','.join(map(str, random_cst)), n_ct, integer_bitwidth))
            print(
                "    %1 = \"FHELinalg.add_eint_int\"(%arg0, %0): (tensor<{1}x!FHE.eint<{0}>>, tensor<{1}xi{2}>) -> (tensor<{1}x!FHE.eint<{0}>>)".format(p, n_ct, integer_bitwidth))
            print("    return %1: tensor<{1}x!FHE.eint<{0}>>".format(p, n_ct))
            print("  }")
            print(f"p-error: {P_ERROR}")
            print("tests:")
            print("  - inputs:")
            print(
                "    - tensor: [{0}]".format(','.join(map(str, random_input))))
            print("      shape: [{0}]".format(n_ct))
            print("---")
            # add_eint
            print(
                "description: add_eint_{0}bits_{1}ct".format(p, n_ct))
            print("program: |")
            print(
                "  func.func @main(%arg0: tensor<{1}x!FHE.eint<{0}>>, %arg1: tensor<{1}x!FHE.eint<{0}>>) -> tensor<{1}x!FHE.eint<{0}>> {{".format(p, n_ct))
            print(
                "    %1 = \"FHELinalg.add_eint\"(%arg0, %arg1): (tensor<{1}x!FHE.eint<{0}>>, tensor<{1}x!FHE.eint<{0}>>) -> (tensor<{1}x!FHE.eint<{0}>>)".format(p, n_ct, integer_bitwidth))
            print("    return %1: tensor<{1}x!FHE.eint<{0}>>".format(p, n_ct))
            print("  }")
            print(f"p-error: {P_ERROR}")
            print("tests:")
            print("  - inputs:")
            print(
                "    - tensor: [{0}]".format(','.join(map(str, random_input))))
            print("      shape: [{0}]".format(n_ct))
            print(
                "    - tensor: [{0}]".format(','.join(map(str, random_input))))
            print("      shape: [{0}]".format(n_ct))
            print("---")
            # mul_eint_int
            print(
                "description: mul_eint_int_{0}bits_{1}ct".format(p, n_ct))
            print("program: |")
            print(
                "  func.func @main(%arg0: tensor<{1}x!FHE.eint<{0}>>) -> tensor<{1}x!FHE.eint<{0}>> {{".format(p, n_ct))
            print("    %0 = arith.constant dense<[2]> : tensor<1xi{0}>".format(
                integer_bitwidth))
            print(
                "    %1 = \"FHELinalg.mul_eint_int\"(%arg0, %0): (tensor<{1}x!FHE.eint<{0}>>, tensor<1xi{2}>) -> (tensor<{1}x!FHE.eint<{0}>>)".format(p, n_ct, integer_bitwidth))
            print("    return %1: tensor<{1}x!FHE.eint<{0}>>".format(
                p, n_ct, integer_bitwidth))
            print("  }")
            print(f"p-error: {P_ERROR}")
            print("tests:")
            print("  - inputs:")
            print(
                "    - tensor: [{0}]".format(','.join(map(str, random_input))))
            print("      shape: [{0}]".format(n_ct))
            print("---")

if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--minimal",
        help="Specify whether to generate minimal tests only",
        type=bool,
        default=False,
    )
    main()
