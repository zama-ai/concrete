from v0curves import curves

# define the number of security levels in curves
num_sec_levels = len(curves)

import_string = f"""

#include <iostream>
using namespace std;"""


constant_string = f"""
const int num_sec_levels = {num_sec_levels};"""


struct_string = """
typedef struct v0curves
{
    int rlweDimension;
    int polynomialSize;
    int ciphertextModulus;
    int keyFormat;

    v0curves(int rlweDimension,
                int polynomialSize_,
                int ciphertextModulus,
                int keyFormat)
    {
        rlweDimension = rlweDimension_;
        polynomialSize = polynomialSize_;
        ciphertextModulus = ciphertextModulus_;
        keyFormat = keyFormat_;
    }

} v0curves;"""


table_string = """
v0curves parameters[num_sec_levels] = """


get_string = """
extern "C" int security_estimator(int securityLevel, int keyFormat)
{
    return &parameters[securityLevel][keyFormat];
}"""


def constructor(rlweDimension, polynomialSize, ciphertextModulus, keyFormat):
    return f"v0curves({rlweDimension}, {polynomialSize}, {ciphertextModulus}, {keyFormat}),"


def fill_parameters(
    # Return a string with parameters for the c++ array initialization
    polynomial_size_results,
    rlwe_dimension_results,
    ciphertext_modulus_results,
    key_format_results
):
    parameters = "{}{{".format(table_string)
    for security_level in range(num_sec_levels):
        print(security_level)
        line = "{"
        
        try:
            line += constructor(
                int(polynomial_size_results[security_level]),
                int(rlwe_dimension_results[security_level]),
                int(ciphertext_modulus_results[security_level]),
                int(key_format_results[security_level]),
            )
        except ValueError:
            line += constructor(0, 0, 0, 0)
        line = line[:-1]
        line += "},"
        parameters += line
    parameters = parameters[:-1]
    parameters += "} ;"
    return parameters


def codegen(
    polynomial_size_results,
    rlwe_dimension_results,
    ciphertext_modulus_results,
    key_format_results,
):
    # Generate the C++ file as a string
    code = f"""
    {import_string}
    {constant_string}
    {struct_string} 
    {fill_parameters(
        polynomial_size_results,
        rlwe_dimension_results,
        ciphertext_modulus_results,
        key_format_results
    )}
    {get_string}
    """
    return code


def write_codegen(
    polynomial_size_results,
    rlwe_dimension_results,
    ciphertext_modulus_results,
    key_format_results,
):
    # Create the c++ source
    code = codegen(
        polynomial_size_results,
        rlwe_dimension_results,
        ciphertext_modulus_results,
        key_format_results
    )
    # TODO: insert correct filename here with a path
    with open(f"test.cpp", "w") as f:
        f.write(code)
    print("> Successfully wrote C++ source to disk")


def main_codegen():
    # finding parameters for V0
    (
        polynomial_size_results,
        rlwe_dimension_results,
        ciphertext_modulus_results,
        key_format_results,
        
    ) = main_optimization_v0()

    # code generation
    write_codegen(
        polynomial_size_results,
        rlwe_dimension_results,
        ciphertext_modulus_results,
        key_format_results
    )
