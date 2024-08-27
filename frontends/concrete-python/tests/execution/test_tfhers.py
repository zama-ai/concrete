"""
Tests execution of tfhers conversion operations.
"""

import json
from typing import List

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe import tfhers


def binary_tfhers(x, y, binary_op, tfhers_type):
    """wrap binary op in tfhers conversion (2 tfhers inputs)"""
    x = tfhers.to_native(x)
    y = tfhers.to_native(y)
    return tfhers.from_native(binary_op(x, y), tfhers_type)


def one_tfhers_one_native(x, y, binary_op, tfhers_type):
    """wrap binary op in tfhers conversion (1 tfhers, 1 native input)"""
    x = tfhers.to_native(x)
    return tfhers.from_native(binary_op(x, y), tfhers_type)


def parameterize_partial_dtype(partial_dtype) -> tfhers.TFHERSIntegerType:
    """Create a tfhers type from a partial func missing tfhers params.

    Args:
        partial_dtype (callable): partial function to create dtype (missing params)

    Returns:
        tfhers.TFHERSIntegerType: tfhers type
    """
    tfhers_params = tfhers.TFHERSParams(
        909,
        1,
        4096,
        15,
        2,
    )
    return partial_dtype(tfhers_params)


def is_input_and_output_tfhers(
    circuit: fhe.Circuit,
    lwe_dim: int,
    tfhers_ins: List[int],
    tfhers_outs: List[int],
) -> bool:
    """Check if inputs and outputs description match tfhers parameters"""
    params = json.loads(circuit.client.specs.client_parameters.serialize())
    main_circuit = params["circuits"][0]
    # check all encrypted input/output have the correct lwe_dim
    ins = main_circuit["inputs"]
    outs = main_circuit["outputs"]
    for indices, param in [(tfhers_ins, ins), (tfhers_outs, outs)]:
        for i in indices:
            if param[i]["rawInfo"]["shape"]["dimensions"][-1] != lwe_dim + 1:
                return False
    return True


@pytest.mark.parametrize(
    "function, parameters, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
                "y": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y big values",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**10, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**10], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            tfhers.uint8_2_2,
            id="x * y",
        ),
    ],
)
def test_tfhers_conversion_binary_encrypted(
    function, parameters, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (2 tfhers inputs).
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # Only valid when running in multi
    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    dtype = parameterize_partial_dtype(dtype)

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        [0, 1],
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)
    encoded_sample = (dtype.encode(v) for v in sample)
    encoded_result = circuit.encrypt_run_decrypt(*encoded_sample)

    assert (dtype.decode(encoded_result) == function(*sample)).all()


@pytest.mark.parametrize(
    "function, parameters, dtype",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "clear"},
            },
            tfhers.uint16_2_2,
            id="x + clear(y)",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
                "y": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x + y big values",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [2**14, 2**15 - 1], "status": "encrypted"},
                "y": {"range": [2**14, 2**15 - 1], "status": "clear"},
            },
            tfhers.uint16_2_2,
            id="x + clear(y) big values",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**10, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**10], "status": "encrypted"},
            },
            tfhers.uint16_2_2,
            id="x - y",
        ),
        pytest.param(
            lambda x, y: x - y,
            {
                "x": {"range": [2**10, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**10], "status": "clear"},
            },
            tfhers.uint16_2_2,
            id="x - clear(y)",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "encrypted"},
            },
            tfhers.uint8_2_2,
            id="x * y",
        ),
        pytest.param(
            lambda x, y: x * y,
            {
                "x": {"range": [0, 2**3], "status": "encrypted"},
                "y": {"range": [0, 2**3], "status": "clear"},
            },
            tfhers.uint8_2_2,
            id="x * clear(y)",
        ),
    ],
)
def test_tfhers_conversion_one_encrypted_one_native(
    function, parameters, dtype: tfhers.TFHERSIntegerType, helpers
):
    """
    Test different operations wrapped by tfhers conversion (1 tfhers, 1 native input).
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    # Only valid when running in multi
    if helpers.configuration().parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        return

    dtype = parameterize_partial_dtype(dtype)

    compiler = fhe.Compiler(
        lambda x, y: one_tfhers_one_native(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        (tfhers.TFHERSInteger(dtype, inpt[0]), inpt[1])
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, helpers.configuration())

    assert is_input_and_output_tfhers(
        circuit,
        dtype.params.polynomial_size,
        (
            [
                0,
            ]
        ),
        [
            0,
        ],
    )

    sample = helpers.generate_sample(parameters)
    encoded_sample = (dtype.encode(sample[0]), sample[1])
    encoded_result = circuit.encrypt_run_decrypt(*encoded_sample)

    assert (dtype.decode(encoded_result) == function(*sample)).all()


@pytest.mark.parametrize(
    "lhs,rhs,is_equal",
    [
        pytest.param(
            tfhers.uint16_2_2,
            tfhers.uint16_2_2,
            True,
        ),
        pytest.param(
            tfhers.uint16_2_2,
            tfhers.uint8_2_2,
            False,
        ),
    ],
)
def test_tfhers_integer_eq(lhs, rhs, is_equal):
    """
    Test TFHERSIntegerType equality.
    """
    assert is_equal == (parameterize_partial_dtype(lhs) == parameterize_partial_dtype(rhs))


@pytest.mark.parametrize(
    "dtype,value,encoded",
    [
        pytest.param(
            tfhers.uint16_2_2,
            [10, 20, 30],
            [
                [0, 0, 0, 0, 0, 0, 2, 2],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 3, 2],
            ],
        ),
    ],
)
def test_tfhers_integer_encode(dtype, value, encoded):
    """
    Test TFHERSIntegerType encode.
    """
    dtype = parameterize_partial_dtype(dtype)
    assert np.array_equal(dtype.encode(value), encoded)


@pytest.mark.parametrize(
    "dtype,value,expected_error,expected_message",
    [
        pytest.param(
            tfhers.uint16_2_2,
            "foo",
            TypeError,
            "can only encode int, np.integer, list or ndarray, but got <class 'str'>",
        ),
    ],
)
def test_tfhers_integer_bad_encode(dtype, value, expected_error, expected_message):
    """
    Test TFHERSIntegerType encode.
    """

    dtype = parameterize_partial_dtype(dtype)
    with pytest.raises(expected_error) as excinfo:
        dtype.encode(value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "dtype,encoded,decoded",
    [
        pytest.param(
            tfhers.uint16_2_2,
            [
                [0, 0, 0, 0, 0, 0, 2, 2],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 3, 2],
            ],
            [10, 20, 30],
        ),
    ],
)
def test_tfhers_integer_decode(dtype, encoded, decoded):
    """
    Test TFHERSIntegerType decode.
    """

    dtype = parameterize_partial_dtype(dtype)
    assert np.array_equal(dtype.decode(encoded), decoded)


@pytest.mark.parametrize(
    "dtype,value,expected_error,expected_message",
    [
        pytest.param(
            tfhers.uint16_2_2,
            "foo",
            TypeError,
            "can only decode list of integers or ndarray of integers, but got <class 'str'>",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            [1, 2, 3],
            ValueError,
            "expected the last dimension of encoded value to be 8 but it's 3",
        ),
    ],
)
def test_tfhers_integer_bad_decode(dtype, value, expected_error, expected_message):
    """
    Test TFHERSIntegerType decode.
    """

    dtype = parameterize_partial_dtype(dtype)
    with pytest.raises(expected_error) as excinfo:
        dtype.decode(value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "dtype,value,encoded",
    [
        pytest.param(
            tfhers.uint16_2_2,
            10,
            10,
        ),
        pytest.param(
            tfhers.uint16_2_2,
            20,
            20,
        ),
    ],
)
def test_tfhers_from_native_outside_tracing(dtype, value, encoded):
    """
    Test tfhers.from_native outside tracing.
    """

    assert np.array_equal(tfhers.from_native(value, dtype), encoded)


@pytest.mark.parametrize(
    "value,decoded",
    [
        pytest.param(
            tfhers.TFHERSInteger(parameterize_partial_dtype(tfhers.uint8_2_2), [0, 0, 2, 2]),
            [0, 0, 2, 2],
        ),
    ],
)
def test_tfhers_to_native_outside_tracing(value, decoded):
    """
    Test tfhers.to_native outside tracing.
    """

    print(tfhers.to_native(value))
    assert np.array_equal(tfhers.to_native(value), decoded)


@pytest.mark.parametrize(
    "value,expected_error,expected_message",
    [
        pytest.param(
            "foo",
            ValueError,
            "tfhers.to_native should be called with a TFHERSInteger",
        ),
    ],
)
def test_tfhers_bad_to_native(value, expected_error, expected_message):
    """
    Test tfhers.to_native with bad arguments.
    """

    with pytest.raises(expected_error) as excinfo:
        tfhers.to_native(value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "dtype,value,expected_error,expected_message",
    [
        pytest.param(
            tfhers.uint16_2_2,
            [[1], [2, 3, 4], [5, 6]],
            ValueError,
            "got error while trying to convert list value into a numpy array: "
            "setting an array element with a sequence. The requested array has "
            "an inhomogeneous shape after 1 dimensions. The detected shape was "
            "(3,) + inhomogeneous part.",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            [100_000],
            ValueError,
            "ndarray value has bigger elements than what the dtype can support",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            [-100_000],
            ValueError,
            "ndarray value has smaller elements than what the dtype can support",
        ),
        pytest.param(
            tfhers.uint16_2_2,
            "foo",
            TypeError,
            "value can either be an int or ndarray, not a <class 'str'>",
        ),
    ],
)
def test_tfhers_integer_bad_init(dtype, value, expected_error, expected_message):
    """
    Test __init__ of TFHERSInteger with bad arguments.
    """

    dtype = parameterize_partial_dtype(dtype)
    with pytest.raises(expected_error) as excinfo:
        tfhers.TFHERSInteger(dtype, value)

    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    "function, parameters, parameter_strategy",
    [
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            fhe.ParameterSelectionStrategy.MONO,
            id="mono",
        ),
        pytest.param(
            lambda x, y: x + y,
            {
                "x": {"range": [0, 2**14], "status": "encrypted"},
                "y": {"range": [0, 2**14], "status": "encrypted"},
            },
            fhe.ParameterSelectionStrategy.V0,
            id="v0",
        ),
    ],
)
def test_tfhers_conversion_without_multi(function, parameters, parameter_strategy, helpers):
    """
    Test that circuits using tfhers integers need to use multi parameters.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)

    conf = helpers.configuration()
    conf.parameter_selection_strategy = parameter_strategy

    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    with pytest.raises(RuntimeError, match=f"Can't use tfhers integers with {parameter_strategy}"):
        compiler.compile(inputset, conf)


def test_tfhers_circuit_eval():
    """Test evaluation of tfhers function."""
    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)
    x = tfhers.TFHERSInteger(dtype, 1)
    y = tfhers.TFHERSInteger(dtype, 2)
    result = binary_tfhers(x, y, lambda x, y: x + y, dtype)
    assert result == 3
