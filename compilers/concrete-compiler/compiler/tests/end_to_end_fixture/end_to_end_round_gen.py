import argparse
from functools import reduce
from platform import mac_ver

import numpy as np

from end_to_end_linalg_leveled_gen import P_ERROR

def min_max(from_p, signed):
    if signed:
        min_value = -(2 ** (from_p - 1))
        max_value = abs(min_value) - 1
    else:
        min_value = 0
        max_value = 2 ** from_p - 1
    return min_value, max_value

def round(inital_val, p_start, p_end, signed=False):
    assert p_start > p_end
    val = inital_val
    p_delta = p_start - p_end
    carry_mask = 1 << (p_delta - 1)
    new_val = val + 2 ** (p_delta - 1)
    if val & carry_mask != 0:
        val += carry_mask << 1
    assert val >> p_delta == new_val >> p_delta, f"{val} {new_val}"
    output = val >> p_delta
    if signed:
        if output >= (1 << (p_end - 1)):
            output = -output
    min_value, max_value = min_max(p_end, signed)
    if not(min_value <= output <= max_value):
        print(f"# warning: overflow on padding for input inital:{inital_val} -> rounded:{val} -> reduced:{output}")
    return output

def yaml_test(from_p, to_p, input_vals, output_vals, signed, with_tlu=False, with_shape=None):
    int_type = "esint" if signed else "eint"
    tlu_type = f"!FHE.{int_type}<{to_p}>"
    acc_type = f"!FHE.{int_type}<{from_p}>"
    if with_shape:
        shape_mlir = 'x'.join(map(str, with_shape))
        tlu_type = f"tensor<{shape_mlir} x {tlu_type}>"
        acc_type = f"tensor<{shape_mlir} x {acc_type}>"
    else:
        shape_mlir = "scalar"
    linalg = "Linalg" if with_shape else ""
    full_name = (
        f"{'signed' if signed else 'unsigned'}"
        f"_round_{'tlu' if with_tlu else 'alone'}"
        f"_{'tensorized_' if with_shape else ''}{shape_mlir}"
    )
    signed_yaml = "true" if signed else "false"
    print(f"""description: {full_name}_{from_p}to{to_p}bits""")
    if with_tlu:
        min_value, max_value = min_max(to_p, signed)
        tlu = list(range(max_value + 1)) + list(range(min_value, 0))
        tlu_const_type = f"tensor<{len(tlu)} x i64>"
        print(f"""\
program: |
    func.func @main(%arg0: {acc_type}) -> {acc_type} {{
      %1 = \"FHE{linalg}.round\"(%arg0) : ({acc_type}) -> {tlu_type}
      %tlu = arith.constant dense<{tlu}> : {tlu_const_type}
      %2 = \"FHE{linalg}.apply_lookup_table\"(%1, %tlu) : ({tlu_type}, {tlu_const_type}) -> {acc_type}
      return %2: {acc_type}
    }}
""")
    else:
        print(f"""\
program: |
    func.func @main(%arg0: {acc_type}) -> {tlu_type} {{
      %1 = \"FHE{linalg}.round\"(%arg0) : ({acc_type}) -> {tlu_type}
      return %1: {tlu_type}
    }}
""")
    print(f"""\
p-error: {P_ERROR}
tests:""")
    _, acc_max_value = min_max(from_p, signed)
    tlu_min_value, _ = min_max(to_p, signed)
    for input_val, output_val in zip(input_vals, output_vals):
        if input_val == acc_max_value and with_tlu:
            print("    # tlu with padding = 1")
            print("    # output is max_value + 1 <=> min_value + padding")
            if signed:
                # max_value + 1 <=> min_value + negacyclic
                output_val = -tlu_min_value
            else:
                output_val = 0
        if with_shape:
            from functools import reduce
            flat_size = reduce(lambda x, y: x * y, with_shape)
            input_vals = [input_val] * flat_size
            output_vals = [output_val] * flat_size
            print(f"""\
    - inputs:
        - tensor: {input_vals}
          shape: {with_shape}
          signed: {signed_yaml}
      outputs:
        - tensor: {output_vals}
          shape: {with_shape}
          signed: {signed_yaml}
""")
        else:
            print(f"""\
    - inputs:
        - scalar: {input_val}
          signed: {signed_yaml}
      outputs:
        - scalar: {output_val}
          signed: {signed_yaml}
""")
    print("---")

def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    # unsigned_unsigned
    shapes = tuple([shape] for shape in args.shapes) if args.shapes else (None, [3], [2, 3] , [1, 2, 3])
    domain = [
        (from_p, to_p, signed, with_tlu, with_shape)
        for from_p in args.acc_bitwidth
        for to_p in args.bitwidth
        for signed in (False, True)
        for with_tlu in (False, True)
        for with_shape in shapes
        if to_p < from_p
    ]
    for (from_p, to_p, signed, with_tlu, with_shape) in domain:
        min_value, max_value = min_max(from_p, signed)
        if with_shape:
            input_vals = list({min_value, 0, max_value})
        else:
            input_vals = list(range(min_value, max_value + 1))
        input_vals = [min_value, 0, max_value]
        output_vals = [
            round(val, from_p, to_p, signed)
            for val in input_vals
        ]
        yaml_test(from_p, to_p, input_vals, output_vals, signed, with_tlu=with_tlu, with_shape=with_shape)

if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--bitwidth",
        help="Specify the list of bitwidth to generate",
        nargs="+",
        type=int,
        default=[1, 4, 6, 8],
    )
    CLI.add_argument(
        "--acc-bitwidth",
        help="Specify the list of bitwidth to generate",
        nargs="+",
        type=int,
        default=[8, 13, 16],
    )
    CLI.add_argument(
         "--shapes",
         help="Specify the shapes to test",
         nargs='*',
         type=int,
         default=[],
    )
    CLI.add_argument(
         "--minimal",
         help="Specify whether to generate minimal tests only",
         type=bool,
         default=False,
    )
    generate(CLI.parse_args())
