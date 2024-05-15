"""
Tests of execution of concatenate operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function,parameters",
    [
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {
                "x": {"shape": (4, 2), "status": "clear"},
                "y": {"shape": (3, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2), "status": "clear"},
            },
        ),
        pytest.param(
            lambda x, y: fhe.zero() + np.concatenate((x, y)),
            {
                "x": {"shape": (4, 2), "status": "clear"},
                "y": {"shape": (3, 2), "status": "clear"},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=0),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=1),
            {
                "x": {"shape": (2, 4)},
                "y": {"shape": (2, 3)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=-1),
            {
                "x": {"shape": (2, 4)},
                "y": {"shape": (2, 3)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=-2),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=None),
            {
                "x": {"shape": (3, 4)},
                "y": {"shape": (2, 3)},
            },
        ),
        pytest.param(
            lambda x, y, z: np.concatenate((x, y, z)),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2)},
                "z": {"shape": (5, 2)},
            },
        ),
        pytest.param(
            lambda x, y, z: np.concatenate((x, y, z), axis=0),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2)},
                "z": {"shape": (5, 2)},
            },
        ),
        pytest.param(
            lambda x, y, z: np.concatenate((x, y, z), axis=1),
            {
                "x": {"shape": (2, 4)},
                "y": {"shape": (2, 3)},
                "z": {"shape": (2, 5)},
            },
        ),
        pytest.param(
            lambda x, y, z: np.concatenate((x, y, z), axis=-1),
            {
                "x": {"shape": (2, 4)},
                "y": {"shape": (2, 3)},
                "z": {"shape": (2, 5)},
            },
        ),
        pytest.param(
            lambda x, y, z: np.concatenate((x, y, z), axis=-2),
            {
                "x": {"shape": (4, 2)},
                "y": {"shape": (3, 2)},
                "z": {"shape": (5, 2)},
            },
        ),
        pytest.param(
            lambda x, y, z: np.concatenate((x, y, z), axis=None),
            {
                "x": {"shape": (3, 4)},
                "y": {"shape": (2, 3)},
                "z": {"shape": (5, 1)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {
                "x": {"shape": (3, 4, 2)},
                "y": {"shape": (5, 4, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=0),
            {
                "x": {"shape": (3, 4, 2)},
                "y": {"shape": (5, 4, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=1),
            {
                "x": {"shape": (2, 4, 5)},
                "y": {"shape": (2, 3, 5)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=2),
            {
                "x": {"shape": (2, 3, 4)},
                "y": {"shape": (2, 3, 5)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=-1),
            {
                "x": {"shape": (2, 3, 4)},
                "y": {"shape": (2, 3, 5)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=-2),
            {
                "x": {"shape": (2, 4, 5)},
                "y": {"shape": (2, 3, 5)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=-3),
            {
                "x": {"shape": (3, 4, 2)},
                "y": {"shape": (5, 4, 2)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y), axis=None),
            {
                "x": {"shape": (3, 4, 5)},
                "y": {"shape": (5, 2, 3)},
            },
        ),
        pytest.param(
            lambda x, y: np.concatenate((x, y)),
            {
                "x": {"range": [0, 10], "shape": (4, 2)},
                "y": {"range": [-10, 10], "shape": (3, 2)},
            },
        ),
    ],
)
def test_concatenate(function, parameters, helpers):
    """
    Test concatenate.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
