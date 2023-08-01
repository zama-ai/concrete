import argparse
import numpy as np

PRECISIONS_TO_BENCH = [
    # output, input
    (6, 2),
    #, (16, 7)
]
SHAPES = [((2, 3, 4), (2, 4, 2)), ((3, 4), (4, 2)), ((3,), (3,)), ((3,), (3, 2)), ((3,), (4, 3, 2)), ((3,4), (4,)), ((2,3,4), (4,)), ((2, 1, 3, 4), (5, 4, 2))]
P_ERROR = 1.0 / 1e6

def format_shape(shape):    
    shape_str = "x".join(map(str, shape))
    if len(shape):
        shape_str += "x"
    else:
        shape_str = "1x"
    return shape_str

def flatten_and_to_str(data, is_tensor=True):
    payload = ", ".join(map(str, data.reshape((-1,))))
    if is_tensor:
        return "[" + payload + "]"
    return payload

def generate(op):
    assert(op in {"matmul_eint_eint", "dot_eint_eint"})
    print("# /!\ DO NOT EDIT MANUALLY THIS FILE MANUALLY")
    print("# /!\ THIS FILE HAS BEEN GENERATED")
    for p, p_inputs in PRECISIONS_TO_BENCH:
        for shapes in SHAPES:
            for signed in [False, True]:
                if signed:
                    min_value = - 2 ** (p_inputs - 1)
                    max_value = 2 ** (p_inputs - 1) - 1
                else:
                    min_value = 0
                    max_value = 2 ** p_inputs - 1

                inp_0 = np.random.randint(min_value, max_value/2, size=shapes[0])
                inp_1 = np.random.randint(min_value, max_value/2, size=shapes[1])

                expected_result = inp_0 @ inp_1

                assert(np.all(expected_result < 2**p))
                assert(np.all(expected_result >= 0))

                out_shape = expected_result.shape

                if len(out_shape) < 2 and op == "matmul_eint_eint":
                    # Matmul only works for matmuls on operands 
                    # that produce at least 2-dimensional outputs
                    continue
                elif len(out_shape) >= 1 and op == "dot_eint_eint":
                    # Dot will only be tested when the output is 
                    # a scalar
                    continue

                shape_0_str = format_shape(shapes[0])
                shape_1_str = format_shape(shapes[1])
                out_shape_str = format_shape(out_shape)

                dtype = "esint" if signed else "eint"

                op_outputs_scalar = op == "dot_eint_eint" and len(out_shape) == 0
                out_dtype_str = f"tensor<{out_shape_str}!FHE.{dtype}<{p}>>" if not op_outputs_scalar else f"!FHE.{dtype}<{p}>"

                program = (f"description: {op}_{p}bits_{'s' if signed else 'u'}_{shape_0_str}_{shape_1_str}\n"
                        f"program: |\n"
                        f"  func.func @main(%x: tensor<{shape_0_str}!FHE.{dtype}<{p}>>, "
                                 f"%y: tensor<{shape_1_str}!FHE.{dtype}<{p}>>) -> {out_dtype_str} {{\n"
                        f"       %0 = \"FHELinalg.{op}\"(%x, %y): (tensor<{shape_0_str}!FHE.{dtype}<{p}>>, "
                                 f"tensor<{shape_1_str}!FHE.{dtype}<{p}>>) -> {out_dtype_str}\n"
                        f"       return %0 : {out_dtype_str}\n"
                        f"  }}\n"
                )

                inp_0_str = flatten_and_to_str(inp_0)
                inp_1_str = flatten_and_to_str(inp_1)
                expected_str = flatten_and_to_str(expected_result, is_tensor=not op_outputs_scalar)

                shape_0_str_yaml = ",".join(map(str, shapes[0]))
                shape_1_str_yaml = ",".join(map(str, shapes[1]))
                expected_shape_yaml = ",".join(map(str, out_shape))

                if signed:
                    signed_line = "      signed: True\n"
                else:
                    signed_line = ""
                program += (
                    f"p-error: {P_ERROR}\n"
                    "tests:\n"
                    "  - inputs: \n"
                    f"    - tensor: {inp_0_str}\n"
                    f"      shape: [{shape_0_str_yaml}]\n"
                    f"{signed_line}"
                    f"    - tensor: {inp_1_str}\n"
                    f"      shape: [{shape_1_str_yaml}]\n"
                    f"{signed_line}"
                    f"    outputs:\n"
                )

                if op_outputs_scalar:
                    program += (
                        f"    - scalar: {expected_str}\n"
                        f"{signed_line}"
                    )
                else:
                    program += (
                        f"    - tensor: {expected_str}\n"
                        f"      shape: [{expected_shape_yaml}]\n"
                        f"{signed_line}"
                    )

                program += f"---"

                print(program)



if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--minimal",
        help="Specify whether to generate minimal tests only",
        type=bool,
        default=False,
    )
    args = CLI.parse_args()
    generate("matmul_eint_eint")
    generate("dot_eint_eint")
