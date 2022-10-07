"""
Tests of 'array' extension.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function,parameters,expected_error",
    [
        pytest.param(
            lambda x, y: cnp.array([x, y]),
            {
                "x": {"range": [0, 10], "status": "encrypted", "shape": ()},
                "y": {"range": [0, 10], "status": "encrypted", "shape": (2, 3)},
            },
            "Encrypted arrays can only be created from scalars",
        ),
    ],
)
def test_bad_array(function, parameters, expected_error, helpers):
    """
    Test array with bad parameters.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    with pytest.raises(ValueError) as excinfo:
        inputset = helpers.generate_inputset(parameters)
        compiler.compile(inputset, configuration)

    assert str(excinfo.value) == expected_error
