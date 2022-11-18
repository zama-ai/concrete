from platform import mac_ver
import numpy as np

MIN_PRECISON = 1
MAX_PRECISION = 16
N_CT = [1, 64, 128, 1024]


def main():
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED THANKS THE end_to_end_levelled_gen.py scripts")
    np.random.seed(0)
    for n_ct in N_CT:
        for p in range(MIN_PRECISON, MAX_PRECISION+1):
            max_value = (2 ** p) - 1
            random_lut = np.random.randint(max_value+1, size=2**p)
            # identity_apply_lookup_table
            print(
                "description: apply_lookup_table_{0}bits_{1}ct".format(p, n_ct))
            print("program: |")
            print(
                "  func.func @main(%arg0: tensor<{1}x!FHE.eint<{0}>>) -> tensor<{1}x!FHE.eint<{0}>> {{".format(p, n_ct))
            print("    %tlu = arith.constant dense<[{0}]> : tensor<{1}xi64>".format(
                ','.join(map(str, random_lut)), 2**p))
            print(
                "    %1 = \"FHELinalg.apply_lookup_table\"(%arg0, %tlu): (tensor<{2}x!FHE.eint<{0}>>, tensor<{1}xi64>) -> (tensor<{2}x!FHE.eint<{0}>>)".format(p, 2**p, n_ct))
            print("    return %1: tensor<{1}x!FHE.eint<{0}>>".format(p, n_ct))
            print("  }")
            random_input = np.random.randint(max_value+1, size=n_ct)
            print("tests:")
            print("  - inputs:")
            print(
                "    - tensor: [{0}]".format(','.join(map(str, random_input))))
            print("      shape: [{0}]".format(n_ct))
            outputs = np.vectorize(lambda i: random_lut[i])(random_input)
            print("    outputs:")
            print("    - tensor: [{0}]".format(','.join(map(str, outputs))))
            print("      shape: [{0}]".format(n_ct))
            print("---")


main()
