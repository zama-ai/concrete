"""
Tests of execution of mul operation.
"""

import numpy as np
import pytest

from concrete import fhe


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

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x * y,
            id="x * y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 10], "status": "clear"},
            "y": {"range": [0, 10], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted"},
            "y": {"range": [0, 10], "status": "clear"},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted"},
            "y": {"range": [0, 10], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 10], "status": "clear", "shape": (3,)},
            "y": {"range": [0, 10], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 10], "status": "clear"},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 10], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 10], "status": "clear"},
            "y": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted"},
            "y": {"range": [0, 10], "status": "clear", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted"},
            "y": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "clear", "shape": (3,)},
            "y": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 10], "status": "clear", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "clear", "shape": (2, 1)},
            "y": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted", "shape": (2, 1)},
            "y": {"range": [0, 10], "status": "clear", "shape": (3,)},
        },
        {
            "x": {"range": [0, 10], "status": "encrypted", "shape": (2, 1)},
            "y": {"range": [0, 10], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [-10, 10], "status": "encrypted", "shape": (3, 2)},
            "y": {"range": [-10, 10], "status": "encrypted", "shape": (3, 2)},
        },
        {
            "x": {"range": [-10, 10], "status": "encrypted", "shape": (3, 1)},
            "y": {"range": [0, 0], "status": "encrypted", "shape": (3, 1)},
        },
        {
            "x": {"range": [10, 20], "status": "encrypted", "shape": (3, 1)},
            "y": {"range": [0, 0], "status": "encrypted", "shape": (1, 3)},
        },
        {
            "x": {"range": [2**12, 2**13 - 1], "status": "encrypted", "shape": (3, 1)},
            "y": {"range": [0, 0], "status": "encrypted", "shape": (1, 3)},
        },
        {
            "x": {"range": [2**12, 2**13 - 1], "status": "encrypted", "shape": (3, 1)},
            "y": {"range": [0, 2 * 3 - 1], "status": "encrypted", "shape": (1, 3)},
        },
        {
            "x": {"range": [-(2**7), 2**7 - 1], "status": "encrypted", "shape": (3, 1)},
            "y": {"range": [-(2**7), 2**7 - 1], "status": "encrypted", "shape": (1, 3)},
        },
    ],
)
@pytest.mark.minimal
def test_mul(function, parameters, helpers):
    """
    Test mul where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample, retries=3)


@pytest.mark.parametrize(
    "parameter_encryption_statuses,function,inputs",
    [
        pytest.param(
            {"x": "encrypted", "y": "encrypted"},
            lambda x, y: (x - y) * ((x - y) > 0),
            [
                np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
                np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
            ],
            id="(x - y) * ((x - y) > 0)",
        ),
    ],
)
def test_mul_specific(parameter_encryption_statuses, function, inputs, helpers):
    """
    Test mul with specific inputs.
    """

    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)
    circuit = compiler.compile([tuple(inputs)], configuration)

    helpers.check_execution(circuit, function, inputs)
