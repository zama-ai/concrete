import argparse
from platform import mac_ver

import numpy as np

from end_to_end_linalg_leveled_gen import P_ERROR

PRECISION_FORCE_CRT = 9
P_ERROR_CRT = 1e-9

def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)

    for in_signed in [False, True]:
        for out_signed in [False, True]:
            for p in args.bitwidth:
                if in_signed:
                    in_lower_bound = -(2 ** (p-1))
                    in_upper_bound = (2 ** (p-1)) - 1
                else:
                    in_lower_bound = 0
                    in_upper_bound = (2 ** p) - 1

                if out_signed:
                    out_lower_bound = -(2 ** (p-1))
                    out_upper_bound = (2 ** (p-1)) - 1
                else:
                    out_lower_bound = 0
                    out_upper_bound = (2 ** p) - 1
                
                random_lut = np.random.randint(out_lower_bound, out_upper_bound, size=2**p)
                
                if in_signed:
                    in_type = f"!FHE.esint<{p}>"
                else:
                    in_type = f"!FHE.eint<{p}>"

                if out_signed:
                    out_type = f"!FHE.esint<{p}>"
                else:
                    out_type = f"!FHE.eint<{p}>"

                print(f"description: { 'signed' if in_signed else 'unsigned' }_{ 'signed' if out_signed else 'unsigned' }_apply_lookup_table_{p}bits")
                print("program: |")
                print(
                    f"  func.func @main(%arg0: {in_type}) -> {out_type} {{")
                print(f"    %tlu = arith.constant dense<[{','.join(map(str, random_lut))}]> : tensor<{2**p}xi64>")
                print(
                    f"    %1 = \"FHE.apply_lookup_table\"(%arg0, %tlu): ({in_type}, tensor<{2**p}xi64>) -> ({out_type})")
                print(f"    return %1: {out_type}")
                print("  }")
                if p >= PRECISION_FORCE_CRT:
                    print("encoding: crt")
                    print(f"p-error: {P_ERROR_CRT}")
                else:
                    print(f"p-error: {P_ERROR}")

                def print_in(value):
                    print("  - inputs:")
                    print(f"    - scalar: {value}")
                    if in_signed:
                        print(f"      signed: true")

                def print_out(value):
                    print("    outputs:")
                    print(f"    - scalar: {value}")
                    if out_signed:
                        print(f"      signed: true")
                
                print("tests:")
                random_i = np.random.randint(in_lower_bound, in_upper_bound)
                print_in(random_i)
                print_out(random_lut[random_i])
                if not args.minimal:
                    print_in(0)
                    print_out(random_lut[0])
                    
                    print_in(in_upper_bound)
                    print_out(random_lut[in_upper_bound])
                    
                    if in_signed:
                        #if in_signed, lower_bound==0, already exists
                        print_in(in_lower_bound)
                        print_out(random_lut[in_lower_bound])
                        
                        print_in(-1)
                        print_out(random_lut[-1])
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
        "--minimal",
        help="Specify whether to generate minimal tests only",
        type=bool,
        default=False,
    )
    generate(CLI.parse_args())
