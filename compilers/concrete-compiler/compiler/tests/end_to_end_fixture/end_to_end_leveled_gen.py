import argparse
import random

MIN_PRECISON = 1
from end_to_end_linalg_leveled_gen import P_ERROR

MAX_PRECISION = 57

TEST_ERROR_RATES = """\
test-error-rates:
  - global-p-error: 0.0001
    nb-repetition: 10000"""

PRECISIONS_WITH_ERROR_RATES = {
    1, 2, 3, 4, 9, 16, 24, 32, 57
}


def main(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED THANKS THE end_to_end_levelled_gen.py scripts")
    print("# This reference file aims to test all levelled ops with all bitwidth than we known that the compiler/optimizer support.\n\n")
    # unsigned
    for signed in [False, True]:
        for p in range(MIN_PRECISON, MAX_PRECISION+1):
            if p != 1:
                print("---")
            def may_check_error_rate():
                if p in PRECISIONS_WITH_ERROR_RATES:
                    print(TEST_ERROR_RATES)

            if signed:
                min_value = -(2 ** (p - 1))
                max_value = abs(min_value) - 1
                integer_bitwidth = p + 1
                max_constant = min((2 ** (57-p)) - 1, max_value)
            else:
                min_value = 0
                max_value = (2 ** p) - 1
                integer_bitwidth = p + 1
                max_constant = min((2 ** (57-p)) - 1, max_value)

            if signed:
                type_p = f"!FHE.esint<{p}>"
                type_p_plus_1 = f"!FHE.esint<{p+1}>"
                prefix = "signed_"
            else:
                type_p = f"!FHE.eint<{p}>"
                type_p_plus_1 = f"!FHE.eint<{p+1}>"
                prefix = ""

            def print_1in_1out(input1, output):
                print("  - inputs:")
                print(f"    - scalar: {input1}")
                if signed:
                    print("      signed: true")
                print("    outputs:")
                print(f"    - scalar: {output}")
                if signed:
                    print("      signed: true")

            def print_2ins_1out(input1, input2, output):
                print("  - inputs:")
                print(f"    - scalar: {input1}")
                if signed:
                    print("      signed: true")
                print(f"    - scalar: {input2}")
                if signed:
                    print("      signed: true")
                print("    outputs:")
                print(f"    - scalar: {output}")
                if signed:
                    print("      signed: true")

            # identity
            print(f"description: {prefix}identity_{p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}) -> {type_p} {{")
            print(f"    return %arg0: {type_p}")
            print("  }")
            print("tests:")
            print_1in_1out(min_value, min_value)
            print_1in_1out(max_value, max_value)
            may_check_error_rate()
            print("---")
            # zero_tensor
            print(f"description: {prefix}zero_tensor_{p}bits")
            print("program: |")
            print(f"  func.func @main() -> tensor<2x2x4x{type_p}> {{")
            print(f"    %0 = \"FHE.zero_tensor\"() : () -> tensor<2x2x4x{type_p}>")
            print(f"    return %0: tensor<2x2x4x{type_p}>")
            print("  }")
            print("tests:")
            print("  - outputs:")
            print("    - tensor: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]")
            print("      shape: [2,2,4]")
            if signed:
                print("      signed: true")
            may_check_error_rate()
            print("---")
            # add_eint_int_cst
            print(f"description: {prefix}add_eint_int_cst_{p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}) -> {type_p} {{")
            print(f"    %0 = arith.constant 1 : i{integer_bitwidth}")
            print(f"    %1 = \"FHE.add_eint_int\"(%arg0, %0): ({type_p}, i{integer_bitwidth}) -> ({type_p})")
            print(f"    return %1: {type_p}")
            print("  }")
            print("tests:")
            if signed:
                print_1in_1out(-1, 0)
                print_1in_1out(min_value, min_value + 1)
            print_1in_1out(max_value - 1, max_value)
            may_check_error_rate()
            print("---")

            # add_eint_int_arg
            if p <= 28:
                # above 28 bits the *arg test doesn't have solution
                # TODO: Make a test that test that
                print(f"description: {prefix}add_eint_int_arg_{p}bits")
                print("program: |")
                print(f"  func.func @main(%arg0: {type_p}, %arg1: i{integer_bitwidth}) -> {type_p} {{")
                print(f"    %0 = \"FHE.add_eint_int\"(%arg0, %arg1): ({type_p}, i{integer_bitwidth}) -> ({type_p})")
                print(f"    return %0: {type_p}")
                print("  }")
                print("tests:")
                print_2ins_1out(max_value - 1, 1, max_value)
                print_2ins_1out(max_value, 0, max_value)
                print_2ins_1out((max_value - 1) >> 1, (max_value >> 1) + 1, max_value )
                if signed:
                    print_2ins_1out(min_value, 0, min_value)
                    print_2ins_1out(min_value, 1, min_value + 1)
                    print_2ins_1out(-1, 0, -1)
                    print_2ins_1out(-1, 1, 0)
                    print_2ins_1out(0, 0, 0)
                print("---")
            # add_eint
            print(f"description: {prefix}add_eint_{p}_bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}, %arg1: {type_p}) -> {type_p} {{")
            print(f"    %res = \"FHE.add_eint\"(%arg0, %arg1): ({type_p}, {type_p}) -> {type_p}")
            print(f"    return %res: {type_p}")
            print("  }")
            print("tests:")
            if signed:
                print_2ins_1out(((2 ** p) >> 1) - 1, (2 ** p) >> 1, -1 if p == 1 else (2 ** (p - 1)) - 1)
                print_2ins_1out(-(2 ** (p - 1)), (2 ** (p - 1)) - 1, - 1)
            else:
                print_2ins_1out(((2 ** p) >> 1) - 1, (2 ** p) >> 1, (2 ** p) - 1)
            may_check_error_rate()
            print("---")
            # sub_eint_int_cst
            print(f"description: {prefix}sub_eint_int_cst_{p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}) -> {type_p} {{")
            print(f"    %0 = arith.constant {1 if signed else max_constant} : i{integer_bitwidth}")
            print(f"    %1 = \"FHE.sub_eint_int\"(%arg0, %0): ({type_p}, i{integer_bitwidth}) -> ({type_p})")
            print(f"    return %1: {type_p}")
            print("  }")
            print("tests:")

            if signed:
                print_1in_1out(max_value, max_value - 1)
                print_1in_1out(0, -1)
                print_1in_1out(min_value + 1, min_value)
            else:
                print_1in_1out(max_value, max_value - max_constant)
            may_check_error_rate()
            print("---")
            # sub_eint_int_arg
            if p <= 28:
                # above 28 bits the *arg test doesn't have solution
                # TODO: Make a test that test that
                print(f"description: {prefix}sub_eint_int_arg_{p}bits")
                print("program: |")
                print(f"  func.func @main(%arg0: {type_p}, %arg1: i{integer_bitwidth}) -> {type_p} {{")
                print(f"    %1 = \"FHE.sub_eint_int\"(%arg0, %arg1): ({type_p}, i{integer_bitwidth}) -> ({type_p})")
                print(f"    return %1: {type_p}")
                print("  }")
                print("tests:")
                print_2ins_1out(max_value, max_value, 0)
                print_2ins_1out(max_value, 0, max_value)
                if signed:
                    print_2ins_1out(max_value, 1, max_value - 1)
                    if p != 28:
                        print_2ins_1out(max_value, 2 * max_value, -max_value)
                else:
                    print_2ins_1out(max_value - 1, max_value >> 1, max_value >> 1)
                may_check_error_rate()
                print("---")
            # sub_int_eint_cst
            print(f"description: {prefix}sub_int_eint_cst_{p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}) -> {type_p} {{")
            print(f"    %0 = arith.constant {1 if signed else max_constant} : i{integer_bitwidth}")
            print(f"    %1 = \"FHE.sub_int_eint\"(%0, %arg0): (i{integer_bitwidth}, {type_p}) -> ({type_p})")
            print(f"    return %1: {type_p}")
            print("  }")
            print("tests:")
            if signed:
                if p != 1:
                    print_1in_1out(max_value, min_value + 2)
                    print_1in_1out(0, 1)
                    print_1in_1out(min_value + 2, max_value)
            else:
                print_1in_1out(max_constant, 0)
            may_check_error_rate()
            print("---")
            # sub_int_eint_arg
            if p <= 28:
                # above 28 bits the *arg test doesn't have solution
                # TODO: Make a test that test that
                print(f"description: {prefix}sub_int_eint_arg_{p}bits")
                print("program: |")
                print(f"  func.func @main(%arg0: i{integer_bitwidth}, %arg1: {type_p}) -> {type_p} {{")
                print(f"    %1 = \"FHE.sub_int_eint\"(%arg0, %arg1): (i{integer_bitwidth}, {type_p}) -> ({type_p})")
                print(f"    return %1: {type_p}")
                print("  }")
                print("tests:")
                print_2ins_1out(max_value, max_value, 0)
                print_2ins_1out(max_value, 0, max_value)
                if signed:
                    print_2ins_1out(max_value, 1, max_value - 1)
                    if p != 28:
                        print_2ins_1out(max_value, 2 * max_value, -max_value)
                else:
                    print_2ins_1out(max_value - 1, max_value >> 1,  max_value >> 1)
                may_check_error_rate()
                print("---")
            # sub_eint
            print(f"description: {prefix}sub_eint_{p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}, %arg1: {type_p}) -> {type_p} {{")
            print(f"    %1 = \"FHE.sub_eint\"(%arg0, %arg1): ({type_p}, {type_p}) -> ({type_p})")
            print(f"    return %1: {type_p}")
            print("  }")
            print("tests:")
            
            print_2ins_1out(max_value, 0, max_value)
            if signed:
                print_2ins_1out(0, 1, -1)
                print_2ins_1out(min_value, 0, min_value)
                print_2ins_1out(0, min_value + 1, max_constant)
                print_2ins_1out(0, max_value, min_value + 1)
            else:
                print_2ins_1out(max_value, max_value, 0)
                print_2ins_1out(max_value - 1, max_value >> 1,  max_value >> 1)
            may_check_error_rate()
            print("---")
            # mul_eint_int_cst
            print(f"description: {prefix}mul_eint_int_cst_{p}bits")
            print("program: |")
            print(f"  func.func @main(%arg0: {type_p}) -> {type_p} {{")
            print(f"    %0 = arith.constant 2 : i{integer_bitwidth}")
            print(f"    %1 = \"FHE.mul_eint_int\"(%arg0, %0): ({type_p}, i{integer_bitwidth}) -> ({type_p})")
            print(f"    return %1: {type_p}")
            print("  }")
            if p <= 57:
                print(f"p-error: {P_ERROR}")
            print("tests:")
            print_1in_1out(0, 0)
            if signed:
                if p!=1:
                    print_1in_1out(max_value // 2, max_value - 1)
                    print_1in_1out(min_value // 2, min_value)
            else:
                print_1in_1out(max_value >> 1, max_value - 1)
            may_check_error_rate()
            print("---")
            # mul_eint_int_arg
            if p <= 28:
                # above 28 bits the *arg test doesn't have solution
                # TODO: Make a test that test that
                print(f"description: {prefix}mul_eint_int_arg_{p}bits")
                print("program: |")
                print(f"  func.func @main(%arg0: {type_p}, %arg1: i{integer_bitwidth}) -> {type_p} {{")
                print(f"    %1 = \"FHE.mul_eint_int\"(%arg0, %arg1): ({type_p}, i{integer_bitwidth}) -> ({type_p})")
                print(f"    return %1: {type_p}")
                print("  }")
                print("tests:")
                print_2ins_1out(max_value, 1, max_value)
                if signed:
                    print_2ins_1out(1, min_value, min_value)
                if not args.minimal:
                    print_2ins_1out(0, max_value, 0)
                    print_2ins_1out(max_value, 0, 0)
                    print_2ins_1out(1, max_value,  max_value)
                    if signed:
                        print_2ins_1out(0, 0, 0)
                        print_2ins_1out(min_value, 0, 0)
                        print_2ins_1out(0, min_value, 0)
                        print_2ins_1out(min_value, 1, min_value)
                        print_2ins_1out(max_value, -1, min_value + 1)
                        print_2ins_1out(-1, max_value, min_value + 1)
                        print_2ins_1out(min_value + 1, -1, max_value)
                        print_2ins_1out(-1, min_value + 1, max_value)
                        if p>2:
                            print_2ins_1out(3, 1, 3)
                            print_2ins_1out(3, -1, -3)
                            print_2ins_1out(-3, 1, -3)
                            print_2ins_1out(-3, -1, 3)
                may_check_error_rate()
                print("---")
            # mul_eint
            if p <= 15:
                if signed:
                    if p==1:
                        continue
                    def gen_random_encodable(p):
                        while True:
                            a = random.randint(min_value, max_value)
                            b = random.randint(min_value, max_value)
                            if min_value <= a*b <= max_value:
                                if p == 3:
                                    return a, b
                                if not (a in [-1, 1, 0] or b in [-1, 1, 0]):
                                    return a, b
                else:
                    def gen_random_encodable(p):
                        while True:
                            a = random.randint(1, max_value)
                            b = random.randint(1, max_value)
                            if a*b <= max_value:
                                return a, b

                print(f"description: {prefix}mul_eint_{p+1}bits")
                print("program: |")
                print(f"  func.func @main(%arg0: {type_p_plus_1}, %arg1: {type_p_plus_1}) -> {type_p_plus_1} {{")
                print(f"    %1 = \"FHE.mul_eint\"(%arg0, %arg1): ({type_p_plus_1}, {type_p_plus_1}) -> ({type_p_plus_1})")
                print(f"    return %1: {type_p_plus_1}")
                print("  }")
                print("tests:")
                inp = gen_random_encodable(p+1)

                print_2ins_1out(inp[0], inp[1], inp[0] * inp[1])
                if not args.minimal:
                    print_2ins_1out(0, 0, 0)
                    print_2ins_1out(max_value, 0, 0)
                    print_2ins_1out(0, max_value, 0)
                    print_2ins_1out(1, max_value, max_value)
                    print_2ins_1out(max_value, 1, max_value)
                    if signed:
                        print_2ins_1out(min_value, 0, 0)
                        print_2ins_1out(0, min_value, 0)
                        print_2ins_1out(1, min_value, min_value)
                        print_2ins_1out(min_value, 1, min_value)
                        print_2ins_1out(-1, max_value, -max_value)
                        print_2ins_1out(max_value, -1, -max_value)
                        print_2ins_1out(-1, min_value + 1, -(min_value + 1))
                        print_2ins_1out(min_value + 1, -1, -(min_value + 1))
                        
                print("---")


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--minimal",
        help="Specify whether to generate minimal tests only",
        type=bool,
        default=False,
    )
    main(CLI.parse_args())
