"""
Tests of execution of zeros operation.
"""

import random

import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: fhe.zero() + x,
            id="fhe.zero() + x",
        ),
        pytest.param(
            lambda x: fhe.zeros(()) + x,
            id="fhe.zeros(()) + x",
        ),
        pytest.param(
            lambda x: fhe.zeros(10) + x,
            id="fhe.zeros(10) + x",
        ),
        pytest.param(
            lambda x: fhe.zeros((10,)) + x,
            id="fhe.zeros((10,)) + x",
        ),
        pytest.param(
            lambda x: fhe.zeros((3, 2)) + x,
            id="fhe.zeros((3, 2)) + x",
        ),
    ],
)
def test_zeros(function, helpers):
    """
    Test zeros.
    """

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = range(10)
    circuit = compiler.compile(inputset, configuration)

    sample = random.randint(0, 11)
    helpers.check_execution(circuit, function, sample)
