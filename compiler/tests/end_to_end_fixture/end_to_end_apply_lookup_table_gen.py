from platform import mac_ver
import numpy as np

MIN_PRECISON = 1
MAX_PRECISION = 16


def main():
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    for p in range(MIN_PRECISON, MAX_PRECISION+1):
        if p != 1:
            print("---")
        max_value = (2 ** p) - 1
        random_lut = np.random.randint(max_value+1, size=2**p)
        print("description: apply_lookup_table_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print("    %tlu = arith.constant dense<[{0}]> : tensor<{1}xi64>".format(
            ','.join(map(str, random_lut)), 2**p))
        print(
            "    %1 = \"FHE.apply_lookup_table\"(%arg0, %tlu): (!FHE.eint<{0}>, tensor<{1}xi64>) -> (!FHE.eint<{0}>)".format(p, 2**p))
        print("    return %1: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: 0")
        print("    outputs:")
        print("    - scalar: {0}".format(random_lut[0]))
        print("  - inputs:")
        random_i = np.random.randint(max_value)
        print("    - scalar: {0}".format(random_i))
        print("    outputs:")
        print("    - scalar: {0}".format(random_lut[random_i]))
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value))
        print("    outputs:")
        print("    - scalar: {0}".format(random_lut[max_value]))


main()
