import argparse
from platform import mac_ver

import numpy as np

from end_to_end_linalg_leveled_gen import P_ERROR

def get_identity_lut(precision):
    return list(range(2**precision))

def get_random_of_precision(precision):
    return np.random.randint(2**precision)

def get_simple_layered_inputs(precisions):
    max_result = 2**max(precisions)
    while True:
        tentative_inputs = list(map(get_random_of_precision, precisions))
        if (sum(tentative_inputs) < max_result):
            return tentative_inputs

def get_double_layered_inputs(small_precisions, big_precisions):
    small_max_result = 2**max(small_precisions)
    while True:
        tentative_small_inputs = list(map(get_random_of_precision, small_precisions))
        if (sum(tentative_small_inputs) < small_max_result):
            break
    big_max_result = 2**max(big_precisions)
    while True:
        tentative_big_inputs = list(map(get_random_of_precision, big_precisions))
        if (sum(tentative_small_inputs) + sum(tentative_big_inputs) < big_max_result):
            return tentative_small_inputs + tentative_big_inputs

def generate_simple_layered(precisions):
    max_prec = max(precisions)
    str_name = "_".join(map(str, precisions))
    arg_names = list(map(lambda x: f"%arg{x}" , range(len(precisions))))
    arg_types = list(map(lambda x: f"!FHE.eint<{x}>" , precisions))
    max_type = f"!FHE.eint<{max_prec}>"
    str_args = ", ".join(map(lambda x: f"{x[0]}: {x[1]}", zip(arg_names, arg_types)))
    print(f"description: multi_precision_simple_layered_{str_name}")
    print( "program: |")
    print(f"  func.func @main({str_args}) -> {max_type} {{")
    print(f"    %0 = \"FHE.zero\"() : () -> {max_type}")
    for i in range(len(precisions)): 
        current_precision = precisions[i]
        current_type = arg_types[i]
        current_arg = arg_names[i]
        identity_lut = get_identity_lut(current_precision)
        print(f"    %lut{i} = arith.constant dense<{identity_lut}> : tensor<{len(identity_lut)}xi64>")
        print(f"    %bs{i} = \"FHE.apply_lookup_table\"({current_arg}, %lut{i}): ({current_type}, tensor<{len(identity_lut)}xi64>) -> ({max_type})")
        print(f"    %{i+1} = \"FHE.add_eint\"(%{i}, %bs{i}) : ({max_type}, {max_type}) -> ({max_type})")
    print(f"    return %{i+1} : {max_type}")
    print("  }")
    print(f"p-error: {P_ERROR}")
    print("tests:")
    for i in range(8):
        print("  - inputs:")
        inputs = get_simple_layered_inputs(precisions)
        for input in inputs:
            print(f"    - scalar: {input}")
        print("    outputs:")
        print(f"    - scalar: {sum(inputs)}")
    print("---")

def generate_double_layered(small_precisions, big_precisions):
    assert(max(small_precisions) <= min(big_precisions))
    precisions = small_precisions + big_precisions
    str_name = "_".join(map(str, precisions))
    small_max_prec = max(small_precisions)
    max_prec = max(precisions)
    arg_names = list(map(lambda x: f"%arg{x}" , range(len(precisions))))
    arg_types = list(map(lambda x: f"!FHE.eint<{x}>" , precisions))
    max_type = f"!FHE.eint<{max_prec}>"
    small_max_type = f"!FHE.eint<{small_max_prec}>"
    str_args = ", ".join(map(lambda x: f"{x[0]}: {x[1]}", zip(arg_names, arg_types)))
    print(f"description: multi_precision_double_layered_{str_name}")
    print( "program: |")
    print(f"  func.func @main({str_args}) -> {max_type} {{")
    print(f"    %0 = \"FHE.zero\"() : () -> {small_max_type}")
    for i in range(len(small_precisions)): 
        current_precision = precisions[i]
        current_type = arg_types[i]
        current_arg = arg_names[i]
        identity_lut = get_identity_lut(current_precision)
        print(f"    %lut{i} = arith.constant dense<{identity_lut}> : tensor<{len(identity_lut)}xi64>")
        print(f"    %bs{i} = \"FHE.apply_lookup_table\"({current_arg}, %lut{i}): ({current_type}, tensor<{len(identity_lut)}xi64>) -> ({small_max_type})")
        print(f"    %{i+1} = \"FHE.add_eint\"(%{i}, %bs{i}) : ({small_max_type}, {small_max_type}) -> ({small_max_type})")

    identity_lut = get_identity_lut(current_precision)
    print(f"    %lut{i+1} = arith.constant dense<{identity_lut}> : tensor<{len(identity_lut)}xi64>")
    print(f"    %{i+2} = \"FHE.apply_lookup_table\"(%{i+1}, %lut{i+1}): ({small_max_type}, tensor<{len(identity_lut)}xi64>) -> ({max_type})")

    for j in range(len(big_precisions)): 
        current_precision = precisions[i+1+j]
        current_type = arg_types[i+1+j]
        current_arg = arg_names[i+1+j]
        identity_lut = get_identity_lut(current_precision)
        print(f"    %lut{i+2+j} = arith.constant dense<{identity_lut}> : tensor<{len(identity_lut)}xi64>")
        print(f"    %bs{i+2+j} = \"FHE.apply_lookup_table\"({current_arg}, %lut{i+2+j}): ({current_type}, tensor<{len(identity_lut)}xi64>) -> ({max_type})")
        print(f"    %{i+2+j+1} = \"FHE.add_eint\"(%{i+2+j}, %bs{i+2+j}) : ({max_type}, {max_type}) -> ({max_type})")

    print(f"    return %{i+2+j+1} : {max_type}")
    print("  }")
    print(f"p-error: {P_ERROR}")
    print("tests:")
    for i in range(8):
        print("  - inputs:")
        inputs = get_double_layered_inputs(small_precisions, big_precisions)
        for input in inputs:
            print(f"    - scalar: {input}")
        print("    outputs:")
        print(f"    - scalar: {sum(inputs)}")
    print("---")

def generate(args):
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    np.random.seed(0)
    generate_simple_layered([3,4])
    generate_simple_layered([3,4,5])
    generate_double_layered([3,4],[4,5])
    generate_double_layered([3,4],[5,6])


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--minimal",
        help="Specify whether to generate minimal tests only",
        type=bool,
        default=False,
    )
    generate(CLI.parse_args())
