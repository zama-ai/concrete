MIN_PRECISON = 1
MAX_PRECISION = 57


def main():
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED THANKS THE end_to_end_levelled_gen.py scripts")
    print("# This reference file aims to test all levelled ops with all bitwidth than we known that the compiler/optimizer support.\n\n")
    for p in range(MIN_PRECISON, MAX_PRECISION+1):
        if p != 1:
            print("---")
        max_value = (2 ** p) - 1
        integer_bitwidth = p + 1
        max_constant = min((2 ** (57-p)) - 1, max_value)

        # identity
        print("description: identity_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print("    return %arg0: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value))
        print("    outputs:")
        print("    - scalar: {0}".format(max_value))
        print("---")
        # zero_tensor
        print("description: zero_tensor_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main() -> tensor<2x2x4x!FHE.eint<{0}>> {{".format(p))
        print(
            "    %0 = \"FHE.zero_tensor\"() : () -> tensor<2x2x4x!FHE.eint<{0}>>".format(p))
        print("    return %0: tensor<2x2x4x!FHE.eint<{0}>>".format(p))
        print("  }")
        print("tests:")
        print("  - outputs:")
        print("    - tensor: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]")
        print("      shape: [2,2,4]")
        print("---")
        # add_eint_int_cst
        print("description: add_eint_int_cst_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print("    %0 = arith.constant 1 : i{0}".format(integer_bitwidth))
        print(
            "    %1 = \"FHE.add_eint_int\"(%arg0, %0): (!FHE.eint<{0}>, i{1}) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
        print("    return %1: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value-1))
        print("    outputs:")
        print("    - scalar: {0}".format(max_value))
        print("---")
        # add_eint_int_arg
        if p <= 29:
            # above 29 bits the *arg test doesn't have solution
            # TODO: Make a test that test that
            print("description: add_eint_int_arg_{0}bits".format(p))
            print("program: |")
            print(
                "  func.func @main(%arg0: !FHE.eint<{0}>, %arg1: i{1}) -> !FHE.eint<{0}> {{".format(p, integer_bitwidth))
            print(
                "    %0 = \"FHE.add_eint_int\"(%arg0, %arg1): (!FHE.eint<{0}>, i{1}) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
            print("    return %0: !FHE.eint<{0}>".format(p))
            print("  }")
            print("tests:")
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value-1))
            print("    - scalar: {0}".format(1))
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: {0}".format(0))
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("  - inputs:")
            print("    - scalar: {0}".format((max_value-1) >> 1))
            print("    - scalar: {0}".format((max_value >> 1) + 1))
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("---")
        # add_eint
        print("description: add_eint_{0}_bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>, %arg1: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print(
            "    %res = \"FHE.add_eint\"(%arg0, %arg1): (!FHE.eint<{0}>, !FHE.eint<{0}>) -> !FHE.eint<{0}>".format(p))
        print("    return %res: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: {0}".format(((2 ** p) >> 1) - 1))
        print("    - scalar: {0}".format(((2 ** p) >> 1)))
        print("    outputs:")
        print("    - scalar: {0}".format((2 ** p) - 1))
        print("---")
        # sub_eint_int_cst
        print("description: sub_eint_int_cst_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print("    %0 = arith.constant {1} : i{0}".format(
            integer_bitwidth, max_constant))
        print(
            "    %1 = \"FHE.sub_eint_int\"(%arg0, %0): (!FHE.eint<{0}>, i{1}) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
        print("    return %1: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value))
        print("    outputs:")
        print("    - scalar: {0}".format(max_value-max_constant))
        print("---")
        # sub_eint_int_arg
        if p <= 29:
            # above 29 bits the *arg test doesn't have solution
            # TODO: Make a test that test that
            print("description: sub_eint_int_arg_{0}bits".format(p))
            print("program: |")
            print(
                "  func.func @main(%arg0: !FHE.eint<{0}>, %arg1: i{1}) -> !FHE.eint<{0}> {{".format(p, integer_bitwidth))
            print(
                "    %1 = \"FHE.sub_eint_int\"(%arg0, %arg1): (!FHE.eint<{0}>, i{1}) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
            print("    return %1: !FHE.eint<{0}>".format(p))
            print("  }")
            print("tests:")
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: {0}".format(max_value))
            print("    outputs:")
            print("    - scalar: 0")
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: 0")
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value - 1))
            print("    - scalar: {0}".format(max_value >> 1))
            print("    outputs:")
            print("    - scalar: {0}".format(max_value >> 1))
            print("---")
        # sub_int_eint_cst
        print("description: sub_int_eint_cst_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print("    %0 = arith.constant {1} : i{0}".format(
            integer_bitwidth, max_constant))
        print(
            "    %1 = \"FHE.sub_int_eint\"(%0, %arg0): (i{1}, !FHE.eint<{0}>) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
        print("    return %1: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_constant))
        print("    outputs:")
        print("    - scalar: 0")
        print("---")
        # sub_int_eint_arg
        if p <= 29:
            # above 29 bits the *arg test doesn't have solution
            # TODO: Make a test that test that
            print("description: sub_int_eint_arg_{0}bits".format(p))
            print("program: |")
            print(
                "  func.func @main(%arg0: i{1}, %arg1: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p, integer_bitwidth))
            print(
                "    %1 = \"FHE.sub_int_eint\"(%arg0, %arg1): (i{1}, !FHE.eint<{0}>) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
            print("    return %1: !FHE.eint<{0}>".format(p))
            print("  }")
            print("tests:")
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: {0}".format(max_value))
            print("    outputs:")
            print("    - scalar: 0")
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: 0")
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value - 1))
            print("    - scalar: {0}".format(max_value >> 1))
            print("    outputs:")
            print("    - scalar: {0}".format(max_value >> 1))
            print("---")
        # sub_eint
        print("description: sub_eint_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>, %arg1: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print(
            "    %1 = \"FHE.sub_eint\"(%arg0, %arg1): (!FHE.eint<{0}>, !FHE.eint<{0}>) -> (!FHE.eint<{0}>)".format(p))
        print("    return %1: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value))
        print("    - scalar: {0}".format(max_value))
        print("    outputs:")
        print("    - scalar: 0")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value))
        print("    - scalar: 0")
        print("    outputs:")
        print("    - scalar: {0}".format(max_value))
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value - 1))
        print("    - scalar: {0}".format(max_value >> 1))
        print("    outputs:")
        print("    - scalar: {0}".format(max_value >> 1))
        print("---")
        # mul_eint_int_cst
        print("description: mul_eint_int_cst_{0}bits".format(p))
        print("program: |")
        print(
            "  func.func @main(%arg0: !FHE.eint<{0}>) -> !FHE.eint<{0}> {{".format(p))
        print("    %0 = arith.constant 2 : i{0}".format(integer_bitwidth))
        print(
            "    %1 = \"FHE.mul_eint_int\"(%arg0, %0): (!FHE.eint<{0}>, i{1}) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
        print("    return %1: !FHE.eint<{0}>".format(p))
        print("  }")
        print("tests:")
        print("  - inputs:")
        print("    - scalar: 0")
        print("    outputs:")
        print("    - scalar: 0")
        print("  - inputs:")
        print("    - scalar: {0}".format(max_value >> 1))
        print("    outputs:")
        print("    - scalar: {0}".format(max_value - 1))
        print("---")
        # mul_eint_int_arg
        if p <= 29:
            # above 29 bits the *arg test doesn't have solution
            # TODO: Make a test that test that
            print("description: mul_eint_int_arg_{0}bits".format(p))
            print("program: |")
            print(
                "  func.func @main(%arg0: !FHE.eint<{0}>, %arg1: i{1}) -> !FHE.eint<{0}> {{".format(p, integer_bitwidth))
            print(
                "    %1 = \"FHE.mul_eint_int\"(%arg0, %arg1): (!FHE.eint<{0}>, i{1}) -> (!FHE.eint<{0}>)".format(p, integer_bitwidth))
            print("    return %1: !FHE.eint<{0}>".format(p))
            print("  }")
            print("tests:")
            print("  - inputs:")
            print("    - scalar: 0")
            print("    - scalar: {0}".format(max_value))
            print("    outputs:")
            print("    - scalar: 0")
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: 0")
            print("    outputs:")
            print("    - scalar: 0")
            print("  - inputs:")
            print("    - scalar: 1")
            print("    - scalar: {0}".format(max_value))
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("  - inputs:")
            print("    - scalar: {0}".format(max_value))
            print("    - scalar: 1")
            print("    outputs:")
            print("    - scalar: {0}".format(max_value))
            print("---")


main()
