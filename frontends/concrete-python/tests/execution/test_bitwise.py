"""
Tests of execution of bitwise operations.
"""

import pytest

from concrete import fhe


# @pytest.mark.parametrize(
#     "function",
#     [
#         pytest.param(
#             lambda x, y: x & y,
#             id="x & y",
#         ),
#         pytest.param(
#             lambda x, y: x | y,
#             id="x | y",
#         ),
#         pytest.param(
#             lambda x, y: x ^ y,
#             id="x ^ y",
#         ),
#     ],
# )
# @pytest.mark.parametrize(
#     "parameters",
#     [
#         {
#             "x": {"range": [0, 255], "status": "encrypted"},
#             "y": {"range": [0, 255], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted"},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "y": {"range": [0, 7], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#         },
#     ],
# )
# def test_bitwise(function, parameters, helpers):
#     """
#     Test bitwise operations between encrypted integers.
#     """

#     parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
#     configuration = helpers.configuration()

#     compiler = fhe.Compiler(function, parameter_encryption_statuses)

#     inputset = helpers.generate_inputset(parameters)
#     circuit = compiler.compile(inputset, configuration)

#     sample = helpers.generate_sample(parameters)
#     helpers.check_execution(circuit, function, sample)


# @pytest.mark.parametrize(
#     "function",
#     [
#         pytest.param(
#             lambda x, y: (x & y) + (2**6),
#             id="x & y",
#         ),
#         pytest.param(
#             lambda x, y: (x | y) + (2**6),
#             id="x | y",
#         ),
#         pytest.param(
#             lambda x, y: (x ^ y) + (2**6),
#             id="x ^ y",
#         ),
#     ],
# )
# @pytest.mark.parametrize(
#     "parameters",
#     [
#         {
#             "x": {"range": [0, 7], "status": "encrypted"},
#             "y": {"range": [0, 7], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted"},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "y": {"range": [0, 7], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#         },
#     ],
# )
# def test_bitwise_optimized(function, parameters, helpers):
#     """
#     Test optimized bitwise operations between encrypted integers.
#     """

#     parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
#     configuration = helpers.configuration()

#     compiler = fhe.Compiler(function, parameter_encryption_statuses)

#     inputset = helpers.generate_inputset(parameters)
#     circuit = compiler.compile(inputset, configuration)

#     sample = helpers.generate_sample(parameters)
#     helpers.check_execution(circuit, function, sample)


# @pytest.mark.parametrize(
#     "function",
#     [
#         pytest.param(
#             lambda x, y, z: (x & y) + z,
#             id="(x & y) + z",
#         ),
#         pytest.param(
#             lambda x, y, z: (x & y) + z,
#             id="(x | y) + z",
#         ),
#         pytest.param(
#             lambda x, y, z: (x & y) + z,
#             id="(x ^ y) + z",
#         ),
#     ],
# )
# @pytest.mark.parametrize(
#     "parameters",
#     [
#         {
#             "x": {"range": [0, 7], "status": "encrypted"},
#             "y": {"range": [0, 7], "status": "encrypted"},
#             "z": {"range": [0, 32], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted"},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "z": {"range": [0, 32], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "y": {"range": [0, 7], "status": "encrypted"},
#             "z": {"range": [0, 32], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
#             "z": {"range": [0, 32], "status": "encrypted"},
#         },
#         {
#             "x": {"range": [0, 7], "status": "encrypted", "shape": (3, 1, 2)},
#             "y": {"range": [0, 7], "status": "encrypted", "shape": (4, 2)},
#             "z": {"range": [0, 32], "status": "encrypted"}
#         },
#     ],
# )
# def test_bitwise_3bits_inputs_with_6bits_output(function, parameters, helpers):
#     """
#     Test bitwise operations between encrypted integers when the output need higher precision.
#     """

#     parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
#     configuration = helpers.configuration()

#     compiler = fhe.Compiler(function, parameter_encryption_statuses)

#     inputset = helpers.generate_inputset(parameters)
#     circuit = compiler.compile(inputset, configuration)
#     tlu_lines = [line for line in circuit.mlir.splitlines() if "apply_lookup_table" in line]
#     type_3bits = "eint<3>"
#     type_6bits = "eint<6>"
#     assert tlu_lines
#     for tlu in tlu_lines:
#         ftype = tlu.split(":")[1]  # instr : (input type, _) -> output type
#         params_type, result_type = ftype.split("->")
#         input_type = params_type.split(",")[0]
#         if configuration.single_precision:
#             # tlu : 6 bits input and output
#             assert type_6bits in input_type
#             assert type_6bits in result_type
#         else:
#             # tlu : 3 bits input and 3 or 6bits output
#             assert type_3bits in input_type
#             assert type_3bits in result_type or type_6bits in result_type

#     sample = helpers.generate_sample(parameters)
#     helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y, z: (x & y) | fhe.LookupTable(list(range(64)))[z],
            id="(x & y) | TLU[z]",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
            "z": {"range": [0, 63], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 7], "status": "encrypted", "shape": (3, 1, 2)},
            "y": {"range": [0, 7], "status": "encrypted", "shape": (4, 2)},
            "z": {"range": [0, 63], "status": "encrypted", "shape": (1,)}
        },
    ],
)
def test_bitwise_3bits_inputs_with_6bits_tlu_somewhere_optimized(function, parameters, helpers):
    """
    Test bitwise operations between encrypted integers when the output need higher precision.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration, show_mlir=True)
    tlu_lines = [line for line in circuit.mlir.splitlines() if "apply_lookup_table" in line]
    type_6bits = "eint<6>"
    assert tlu_lines
    for tlu in tlu_lines:
        ftype = tlu.split(":")[1]  # instr : (input type, _) -> output type
        params_type, result_type = ftype.split("->")
        input_type = params_type.split(",")[0]
        # due to packing everything is 6bits
        assert type_6bits in input_type
        assert type_6bits in result_type

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
