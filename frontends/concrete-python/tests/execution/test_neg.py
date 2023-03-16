"""
Tests of execution of neg operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 64], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 64], "status": "encrypted", "shape": (3, 2)},
        },
        {
            "x": {"range": [-63, 0], "status": "encrypted"},
        },
        {
            "x": {"range": [-63, 0], "status": "encrypted", "shape": (3, 2)},
        },
    ],
)
def test_neg(parameters, helpers):
    """
    Test neg.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    @fhe.compiler(parameter_encryption_statuses)
    def operator(x):
        return -x

    @fhe.compiler(parameter_encryption_statuses)
    def function(x):
        return np.negative(x)

    inputset = helpers.generate_inputset(parameters)

    operator_circuit = operator.compile(inputset, configuration)
    function_circuit = function.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)

    helpers.check_execution(operator_circuit, operator, sample)
    helpers.check_execution(function_circuit, function, sample)
