"""
Tests of execution of sub operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: 42 - x,
            id="42 - x",
        ),
        pytest.param(
            lambda x: np.array([1, 2, 3]) - x,
            id="[1, 2, 3] - x",
        ),
        pytest.param(
            lambda x: np.array([[1, 2, 3], [4, 5, 6]]) - x,
            id="[[1, 2, 3], [4, 5, 6]] - x",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 60], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 60], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 60], "status": "encrypted", "shape": (2, 3)},
        },
    ],
)
def test_constant_sub(function, parameters, helpers):
    """
    Test sub where one of the operators is a constant.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses, configuration)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
