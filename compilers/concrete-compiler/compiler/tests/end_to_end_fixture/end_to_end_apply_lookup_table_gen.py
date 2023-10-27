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
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    # unsigned_unsigned
    for p in args.bitwidth:
        max_value = (2 ** p) - 1
        random_lut = np.random.randint(max_value+1, size=2**p)
        itype = get_lut_integer_type(p)
        iprec = itype.replace("i", "")
        print(f"description: apply_lookup_table_{p}bits")
        print("program: |")
        print(
            f"  func.func @main(%arg0: !FHE.eint<{p}>, %arg1: tensor<{2**p}x{itype}>) -> !FHE.eint<{p}> {{")
        print(
            f"    %1 = \"FHE.apply_lookup_table\"(%arg0, %arg1): (!FHE.eint<{p}>, tensor<{2**p}x{itype}>) -> (!FHE.eint<{p}>)")
        print(f"    return %1: !FHE.eint<{p}>")
        print("  }")
        if p >= PRECISION_FORCE_CRT:
            print("encoding: crt")
        print(f"p-error: {P_ERROR}")
        print("tests:")
        print("  - inputs:")
        random_i = np.random.randint(max_value)
        print(f"    - scalar: {random_i}")
        print(f"    - tensor: [{','.join(map(str, random_lut))}]")
        print(f"      shape: [{2**p}]")
        print(f"      width: {iprec}")
        print("    outputs:")
        print(f"    - scalar: {random_lut[random_i]}")
        if not args.minimal:
            print("  - inputs:")
            print("    - scalar: 0")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print("    outputs:")
            print(f"    - scalar: {random_lut[0]}")
            print("  - inputs:")
            print(f"    - scalar: {max_value}")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print("    outputs:")
            print(f"    - scalar: {random_lut[max_value]}")
        print("---")
    # unsigned_signed
    for p in args.bitwidth:
        lower_bound = -(2 ** (p-1))
        upper_bound = (2 ** (p-1)) - 1
        max_value = (2 ** p) - 1
        random_lut = np.random.randint(lower_bound, upper_bound, size=2**p)
        itype = get_lut_integer_type(p)
        iprec = itype.replace("i", "")
        print(f"description: unsigned_signed_apply_lookup_table_{p}bits")
        print("program: |")
        print(
            f"  func.func @main(%arg0: !FHE.eint<{p}>, %arg1: tensor<{2**p}x{itype}>) -> !FHE.esint<{p}> {{")
        print(
            f"    %1 = \"FHE.apply_lookup_table\"(%arg0, %arg1): (!FHE.eint<{p}>, tensor<{2**p}x{itype}>) -> (!FHE.esint<{p}>)")
        print(f"    return %1: !FHE.esint<{p}>")
        print("  }")
        if p >= PRECISION_FORCE_CRT:
            print("encoding: crt")
        print(f"p-error: {P_ERROR}")
        print("tests:")
        print("  - inputs:")
        random_i = np.random.randint(max_value)
        print(f"    - scalar: {random_i}")
        print(f"    - tensor: [{','.join(map(str, random_lut))}]")
        print(f"      shape: [{2**p}]")
        print(f"      width: {iprec}")
        print(f"      signed: true")
        print("    outputs:")
        print(f"    - scalar: {random_lut[random_i]}")
        print(f"      signed: true")
        if not args.minimal:
            print("  - inputs:")
            print("    - scalar: 0")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print(f"      signed: true")
            print("    outputs:")
            print(f"    - scalar: {random_lut[0]}")
            print(f"      signed: true")
            print("  - inputs:")
            print(f"    - scalar: {max_value}")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print(f"      signed: true")
            print("    outputs:")
            print(f"    - scalar: {random_lut[max_value]}")
            print(f"      signed: true")
        print("---")
    # signed_signed
    for p in args.bitwidth:
        lower_bound = -(2 ** (p-1))
        upper_bound = (2 ** (p-1)) - 1
        random_lut = np.random.randint(lower_bound, upper_bound, size=2**p)
        itype = get_lut_integer_type(p)
        iprec = itype.replace("i", "")
        print(f"description: signed_apply_lookup_table_{p}bits")
        print("program: |")
        print(
            f"  func.func @main(%arg0: !FHE.esint<{p}>, %arg1: tensor<{2**p}x{itype}>) -> !FHE.esint<{p}> {{")
        
        print(
            f"    %1 = \"FHE.apply_lookup_table\"(%arg0, %arg1): (!FHE.esint<{p}>, tensor<{2**p}x{itype}>) -> (!FHE.esint<{p}>)")
        print(f"    return %1: !FHE.esint<{p}>")
        print("  }")
        if p >= PRECISION_FORCE_CRT:
            print("encoding: crt")
        print(f"p-error: {P_ERROR}")
        print("tests:")
        print("  - inputs:")
        random_i = np.random.randint(lower_bound, upper_bound)
        print(f"    - scalar: {random_i}")
        print(f"      signed: true")
        print(f"    - tensor: [{','.join(map(str, random_lut))}]")
        print(f"      shape: [{2**p}]")
        print(f"      width: {iprec}")
        print(f"      signed: true")
        print("    outputs:")
        print(f"    - scalar: {random_lut[random_i]}")
        print(f"      signed: true")
        if not args.minimal:
            print("  - inputs:")
            print(f"    - scalar: 0")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print(f"      signed: true")
            print("    outputs:")
            print(f"    - scalar: {random_lut[0]}")
            print(f"      signed: true")
            print("  - inputs:")
            print(f"    - scalar: {upper_bound}")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print(f"      signed: true")
            print("    outputs:")
            print(f"    - scalar: {random_lut[upper_bound]}")
            print(f"      signed: true")
            print("  - inputs:")
            print(f"    - scalar: {lower_bound}")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print(f"      signed: true")
            print("    outputs:")
            print(f"    - scalar: {random_lut[lower_bound]}")
            print(f"      signed: true")
            print("  - inputs:")
            print(f"    - scalar: -1")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print(f"      signed: true")
            print("    outputs:")
            print(f"    - scalar: {random_lut[-1]}")
            print(f"      signed: true")
        print("---")

    # signed_unsigned
    for p in args.bitwidth:
        lower_bound = -(2 ** (p-1))
        upper_bound = (2 ** (p-1)) - 1
        max_value = (2 ** p) - 1
        random_lut = np.random.randint(max_value+1, size=2**p)
        itype = get_lut_integer_type(p)
        iprec = itype.replace("i", "")
        print(f"description: signed_unsigned_apply_lookup_table_{p}bits")
        print("program: |")
        print(
            f"  func.func @main(%arg0: !FHE.esint<{p}>, %arg1: tensor<{2**p}x{itype}>) -> !FHE.eint<{p}> {{")
        print(
            f"    %1 = \"FHE.apply_lookup_table\"(%arg0, %arg1): (!FHE.esint<{p}>, tensor<{2**p}x{itype}>) -> (!FHE.eint<{p}>)")
        print(f"    return %1: !FHE.eint<{p}>")
        print("  }")
        if p >= PRECISION_FORCE_CRT:
            print("encoding: crt")
        print(f"p-error: {P_ERROR}")
        print("tests:")
        print("  - inputs:")
        random_i = np.random.randint(lower_bound, upper_bound)
        print(f"    - scalar: {random_i}")
        print(f"      signed: true")
        print(f"    - tensor: [{','.join(map(str, random_lut))}]")
        print(f"      shape: [{2**p}]")
        print(f"      width: {iprec}")
        print("    outputs:")
        print(f"    - scalar: {random_lut[random_i]}")
        if not args.minimal:
            print("  - inputs:")
            print(f"    - scalar: 0")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print("    outputs:")
            print(f"    - scalar: {random_lut[0]}")
            print("  - inputs:")
            print(f"    - scalar: {upper_bound}")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print("    outputs:")
            print(f"    - scalar: {random_lut[upper_bound]}")
            print("  - inputs:")
            print(f"    - scalar: {lower_bound}")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print("    outputs:")
            print(f"    - scalar: {random_lut[lower_bound]}")
            print("  - inputs:")
            print(f"    - scalar: -1")
            print(f"      signed: true")
            print(f"    - tensor: [{','.join(map(str, random_lut))}]")
            print(f"      shape: [{2**p}]")
            print(f"      width: {iprec}")
            print("    outputs:")
            print(f"    - scalar: {random_lut[-1]}")
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
