"""
Tests of execution of mul operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: x * 3,
            id="x * 3",
        ),
        pytest.param(
            lambda x: 3 * x,
            id="3 * x",
        ),
        pytest.param(
            lambda x: np.dot(x, 3),
            id="np.dot(x, 3)",
        ),
        pytest.param(
            lambda x: np.dot(3, x),
            id="np.dot(3, x)",
        ),
        pytest.param(
            lambda x: x * np.array([1, 2, 3]),
            id="x * [1, 2, 3]",
        ),
        pytest.param(
            lambda x: np.array([1, 2, 3]) * x,
            id="[1, 2, 3] * x",
        ),
        pytest.param(
            lambda x: x * np.array([[1, 2, 3], [3, 1, 2]]),
            id="x * [[1, 2, 3], [3, 1, 2]]",
        ),
        pytest.param(
            lambda x: np.array([[1, 2, 3], [3, 1, 2]]) * x,
            id="[[1, 2, 3], [3, 1, 2]] * x",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 40], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 40], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 40], "status": "encrypted", "shape": (2, 3)},
        },
    ],
)
def test_constant_mul(function, parameters, helpers):
    """
    Test mul where one of the operators is a constant.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
