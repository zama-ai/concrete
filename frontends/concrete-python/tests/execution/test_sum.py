"""
Tests of execution of sum operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function,parameters",
    [
        pytest.param(
            lambda x: np.sum(x),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, 0),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, 1),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=None),  # type: ignore
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=0),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=1),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=-1),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=-2),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=(0, 1)),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=(-2, -1)),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=0, keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=1, keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=-1, keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=-2, keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=(0, 1), keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x: np.sum(x, axis=(-2, -1), keepdims=True),
            {
                "x": {"shape": (3, 2), "range": [0, 10], "status": "encrypted"},
            },
        ),
    ],
)
def test_sum(function, parameters, helpers):
    """
    Test sum.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
