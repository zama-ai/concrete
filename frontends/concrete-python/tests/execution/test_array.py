"""
Tests of execution of array operation.
"""

import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function,parameters",
    [
        pytest.param(
            lambda x: fhe.array([x, x + 1, 1]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
            },
            id="fhe.array([x, x + 1, 1])",
        ),
        pytest.param(
            lambda x, y: fhe.array([x, y]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
                "y": {"range": [0, 10], "status": "encrypted", "shape": ()},
            },
            id="fhe.array([x, y])",
        ),
        pytest.param(
            lambda x, y: fhe.array([x, y]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
                "y": {"range": [0, 10], "status": "clear", "shape": ()},
            },
            id="fhe.array([x, y])",
        ),
        pytest.param(
            lambda x, y: fhe.array([x, y]),
            {
                "x": {"range": [0, 10], "status": "clear", "shape": ()},
                "y": {"range": [0, 10], "status": "clear", "shape": ()},
            },
            id="fhe.array([x, y])",
        ),
        pytest.param(
            lambda x, y: fhe.array([[x, y], [y, x]]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
                "y": {"range": [0, 10], "status": "clear", "shape": ()},
            },
            id="fhe.array([[x, y], [y, x]])",
        ),
        pytest.param(
            lambda x, y, z: fhe.array([[x, 1], [y, 2], [z, 3]]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
                "y": {"range": [0, 10], "status": "clear", "shape": ()},
                "z": {"range": [0, 10], "status": "encrypted", "shape": ()},
            },
            id="fhe.array([[x, 1], [y, 2], [z, 3]])",
        ),
        pytest.param(
            lambda x, y: fhe.array([x, y]) + fhe.array([x, y]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
                "y": {"range": [0, 10], "status": "clear", "shape": ()},
            },
            id="fhe.array([x, y]) + fhe.array([x, y])",
        ),
    ],
)
def test_array(function, parameters, helpers):
    """
    Test array.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
