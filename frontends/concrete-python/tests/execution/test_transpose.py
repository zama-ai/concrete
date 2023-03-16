"""
Tests of execution of transpose operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function,parameters",
    [
        pytest.param(
            lambda x: np.transpose(x),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: x.transpose(),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: x.T,
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: x.transpose((1, 0, 2)),
            {
                "x": {"shape": (2, 3, 4), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: x.transpose((1, 2, 0)),
            {
                "x": {"shape": (2, 3, 4), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: x.transpose((0, 2, 1)),
            {
                "x": {"shape": (2, 3, 4), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: x.transpose((2, 0, 1)),
            {
                "x": {"shape": (2, 3, 4), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.transpose(x, (3, 0, 2, 1)),
            {
                "x": {"shape": (2, 3, 4, 5), "range": [0, 10], "status": "encrypted"},
            },
        ),
    ],
)
def test_transpose(function, parameters, helpers):
    """
    Test transpose.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
