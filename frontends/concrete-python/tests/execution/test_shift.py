"""
Tests of execution of shift operations.
"""

import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x << y,
            id="x << y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 3], "status": "encrypted"},
            "y": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
        },
        {
            "x": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 3], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_left_shift(function, parameters, helpers):
    """
    Test left shift between encrypted integers.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample, retries=3)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x >> y,
            id="x >> y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1 << 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 1 << 4], "status": "encrypted"},
            "y": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
        },
        {
            "x": {"range": [0, 1 << 4], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 3], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 1 << 4], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_right_shift(function, parameters, helpers):
    """
    Test right shift between encrypted integers.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample, retries=3)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x << y,
            id="x << y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
    ],
)
def test_left_shift_coverage(function, parameters, helpers):
    """
    Test left shift between encrypted integers all cases.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    for i in range(2):
        for j in range(8):
            helpers.check_execution(circuit, function, [i, j], retries=3)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x >> y,
            id="x >> y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1 << 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
    ],
)
def test_right_shift_coverage(function, parameters, helpers):
    """
    Test right shift between encrypted integers all cases.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    helpers.check_execution(circuit, function, [0b11, 0], retries=3)
    helpers.check_execution(circuit, function, [0b11, 1], retries=3)
    helpers.check_execution(circuit, function, [0b110, 2], retries=3)
    helpers.check_execution(circuit, function, [0b1100, 3], retries=3)
    helpers.check_execution(circuit, function, [0b11000, 4], retries=3)
    helpers.check_execution(circuit, function, [0b110000, 5], retries=3)
    helpers.check_execution(circuit, function, [0b110000, 6], retries=3)
    helpers.check_execution(circuit, function, [0b1100000, 7], retries=3)
