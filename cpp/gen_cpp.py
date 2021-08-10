from V0Parameters.compile_test import check_codegen
from V0Parameters.tabulation import main_optimization_v0
from V0Parameters.compile_test import check_codegen
from V0Parameters.misc import NORM2_MAX, P_MAX, v0_parameters_path
import numpy as np

import_string = f"""

#include <iostream>
using namespace std;"""


constant_string = f"""
const int num_sec_levels = {num_sec_levels};"""

struct_string = """
typedef struct V0Parameter
{
    int k;
    int polynomialSize;
    int nSmall;
    int brLevel;
    int brLogBase;
    int ksLevel;
    int ksLogBase;

    V0Parameter(int k_,
                int polynomialSize_,
                int nSmall_,
                int brLevel_,
                int brLogBase_,
                int ksLevel_,
                int ksLogBase_)
    {
        k = k_;
        polynomialSize = polynomialSize_;
        nSmall = nSmall_;
        brLevel = brLevel_;
        brLogBase = brLogBase_;
        ksLevel = ksLevel_;
        ksLogBase = ksLogBase_;
    }

} V0Parameter;"""


table_string = """
V0Parameter parameters[NORM2_MAX][P_MAX] = """

get_string = """
extern "C" V0Parameter *get(int norm, int p)
{
    // - 1 is an offset as norm and p are in [1, ...] and not [0, ...]
    return &parameters[norm - 1][p - 1];
}"""


def constructor(k, polynomialSize, nSmall, brLevel, brLogBase, ksLevel, ksLogBase):
    return f"V0Parameter({k}, {polynomialSize}, {nSmall}, {brLevel}, {brLogBase}, {ksLevel}, {ksLogBase}),"


def fill_parameters(
    # Return a string with parameters for the c++ array initialization
    polynomial_size_results,
    rlwe_dimension_results,
):
    parameters = "{}{{".format(table_string)
    for norm in range(1, NORM2_MAX + 1):
        line = "{"
        for p in range(1, P_MAX + 1):
            try:
                line += constructor(
                    1,
                    int(polynomial_size_results[(norm, p)]),
                    int(lwe_dimension_results[(norm, p)]),
                    int(br_level_results[(norm, p)]),
                    int(br_logbase_results[(norm, p)]),
                    int(ks_level_results[(norm, p)]),
                    int(ks_logbase_results[(norm, p)]),
                )
            except ValueError:
                line += constructor(0, 0, 0, 0, 0, 0, 0)
        line = line[:-1]
        line += "},"
        parameters += line
    parameters = parameters[:-1]
    parameters += "} ;"
    return parameters


def codegen(
    polynomial_size_results,
    lwe_dimension_results,
    ks_base_results,
    ks_level_results,
    br_base_results,
    br_level_results,
):
    # Generate the C++ file as a string
    code = f"""
    {import_string}
    {constant_string}
    {struct_string} 
    {fill_parameters(
        polynomial_size_results,
        lwe_dimension_results,
        ks_base_results,
        ks_level_results,
        br_base_results,
        br_level_results,
    )}
    {get_string}
    """
    return code


def write_codegen(
    polynomial_size_results,
    lwe_dimension_results,
    ks_base_results,
    ks_level_results,
    br_base_results,
    br_level_results,
):
    # Create the c++ source
    code = codegen(
        polynomial_size_results,
        lwe_dimension_results,
        ks_base_results,
        ks_level_results,
        br_base_results,
        br_level_results,
    )

    with open(f"{v0_parameters_path}/parameters.cpp", "w") as f:
        f.write(code)
    print("> Successfully wrote C++ source to disk")


def main_codegen():
    # finding parameters for V0
    (
        polynomial_size_results,
        lwe_dimension_results,
        ks_base_results,
        ks_level_results,
        br_base_results,
        br_level_results,
    ) = main_optimization_v0()

    # code generation
    write_codegen(
        polynomial_size_results,
        lwe_dimension_results,
        ks_base_results,
        ks_level_results,
        br_base_results,
        br_level_results,
    )

    # check code gen worked
    check_codegen(
        polynomial_size_results,
        lwe_dimension_results,
        ks_base_results,
        ks_level_results,
        br_base_results,
        br_level_results,
    )
