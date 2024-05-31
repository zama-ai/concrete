"""
Tests execution of tfhers conversion operations.
"""

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
        761,
        1,
        2048,
        6.36835566258815e-06,
        3.1529322391500584e-16,
        23,
        1,
        3,
        5,
        4,
        4,
        5,
        -40.05,
        None,
        True,
    )
    return partial_dtype(tfhers_params)


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
    # tfhers int works with multi-parameters only
    configuration = helpers.configuration().fork(
        parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI
    )

    dtype = parameterize_partial_dtype(dtype)

    compiler = fhe.Compiler(
        lambda x, y: binary_tfhers(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        tuple(tfhers.TFHERSInteger(dtype, arg) for arg in inpt)
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, configuration)

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
    # tfhers int works with multi-parameters only
    configuration = helpers.configuration().fork(
        parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI
    )

    dtype = parameterize_partial_dtype(dtype)

    compiler = fhe.Compiler(
        lambda x, y: one_tfhers_one_native(x, y, function, dtype),
        parameter_encryption_statuses,
    )

    inputset = [
        (tfhers.TFHERSInteger(dtype, inpt[0]), inpt[1])
        for inpt in helpers.generate_inputset(parameters)
    ]
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    encoded_sample = (dtype.encode(sample[0]), sample[1])
    encoded_result = circuit.encrypt_run_decrypt(*encoded_sample)

    assert (dtype.decode(encoded_result) == function(*sample)).all()


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
    # tfhers int works with multi-parameters only
    configuration = helpers.configuration().fork(parameter_selection_strategy=parameter_strategy)

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
        compiler.compile(inputset, configuration)
