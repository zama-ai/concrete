"""
Tests of execution of bitwise operations.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x & y,
            id="x & y",
        ),
        pytest.param(
            lambda x, y: x | y,
            id="x | y",
        ),
        pytest.param(
            lambda x, y: x ^ y,
            id="x ^ y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 7], "status": "encrypted", "shape": (3,)},
        },
    ],
)
def test_bitwise(function, parameters, helpers):
    """
    Test bitwise operations between encrypted integers.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
