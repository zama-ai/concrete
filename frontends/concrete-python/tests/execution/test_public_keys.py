"""
Tests of execution with public keys
"""

import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x + y,
            id="x + y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 60], "status": "clear"},
            "y": {"range": [0, 60], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 60], "status": "encrypted"},
            "y": {"range": [0, 60], "status": "clear"},
        },
        {
            "x": {"range": [0, 60], "status": "encrypted"},
            "y": {"range": [0, 60], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 60], "status": "clear", "shape": (3,)},
            "y": {"range": [0, 60], "status": "encrypted", "shape": (3,)},
        },
        {
            "x": {"range": [0, 60], "status": "encrypted", "shape": (3,)},
            "y": {"range": [0, 60], "status": "clear", "shape": (3,)},
        },
    ],
)
def test_add(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()
    configuration.with_public_keys = fhe.PublicKeyKind.COMPACT
    compiler = fhe.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)
    sample = helpers.generate_sample(parameters)
    print(circuit.server.client_specs.client_parameters.serialize())
    helpers.check_execution(circuit, function, sample, 1, False, True)
