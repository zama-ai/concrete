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
    configuration = helpers.configuration()

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
    configuration = helpers.configuration()

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
