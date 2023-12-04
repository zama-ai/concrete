"""
Tests of execution with multiple outputs.
"""

import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function,parameters",
    [
        pytest.param(
            lambda x: (x * 2, x - 2),
            {
                "x": {"range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: (x * 2, y - 2),
            {
                "x": {"range": [0, 2], "status": "encrypted"},
                "y": {"range": [0, 15], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: (x * 2, y - 2),
            {
                "x": {"range": [0, 2], "shape": (2, 3), "status": "encrypted"},
                "y": {"range": [0, 15], "status": "encrypted"},
            },
        ),
    ],
)
def test_multi_output(function, parameters, helpers):
    """
    Test functions with multiple outputs.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, parameter_encryption_statuses)
    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
