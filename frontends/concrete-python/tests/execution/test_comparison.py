"""
Tests of execution of comparison operations.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x == y,
            id="x == y",
        ),
        pytest.param(
            lambda x, y: x != y,
            id="x != y",
        ),
        pytest.param(
            lambda x, y: x < y,
            id="x < y",
        ),
        pytest.param(
            lambda x, y: x <= y,
            id="x <= y",
        ),
        pytest.param(
            lambda x, y: x > y,
            id="x > y",
        ),
        pytest.param(
            lambda x, y: x >= y,
            id="x >= y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 3], "status": "encrypted"},
            "y": {"range": [0, 3], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-8, 7], "status": "encrypted"},
            "y": {"range": [-8, 7], "status": "encrypted", "shape": (2,)},
        },
        {
            "x": {"range": [-8, 7], "status": "encrypted", "shape": (2,)},
            "y": {"range": [-8, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [-8, 7], "status": "encrypted", "shape": (2,)},
            "y": {"range": [-8, 7], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_comparison(function, parameters, helpers):
    """
    Test comparison operations between encrypted integers.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: (x == y) + 200,
            id="(x == y) + 200",
        ),
        pytest.param(
            lambda x, y: (x != y) + 200,
            id="(x != y) + 200",
        ),
        pytest.param(
            lambda x, y: (x < y) + 200,
            id="(x < y) + 200",
        ),
        pytest.param(
            lambda x, y: (x <= y) + 200,
            id="(x <= y) + 200",
        ),
        pytest.param(
            lambda x, y: (x > y) + 200,
            id="(x > y) + 200",
        ),
        pytest.param(
            lambda x, y: (x >= y) + 200,
            id="(x >= y) + 200",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 15], "status": "encrypted"},
            "y": {"range": [0, 15], "status": "encrypted"},
        },
        {
            "x": {"range": [-8, 7], "status": "encrypted"},
            "y": {"range": [-8, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted"},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
        {
            "x": {"range": [-8, 7], "status": "encrypted", "shape": (2,)},
            "y": {"range": [-8, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [-10, 10], "status": "encrypted", "shape": (2,)},
            "y": {"range": [-10, 10], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_optimized_comparison(function, parameters, helpers):
    """
    Test comparison operations between encrypted integers with a single TLU.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
