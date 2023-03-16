"""
Tests of execution of ones operation.
"""

import random

import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: fhe.one() + x,
            id="fhe.one() + x",
        ),
        pytest.param(
            lambda x: fhe.ones(()) + x,
            id="fhe.ones(()) + x",
        ),
        pytest.param(
            lambda x: fhe.ones(10) + x,
            id="fhe.ones(10) + x",
        ),
        pytest.param(
            lambda x: fhe.ones((10,)) + x,
            id="fhe.ones((10,)) + x",
        ),
        pytest.param(
            lambda x: fhe.ones((3, 2)) + x,
            id="fhe.ones((3, 2)) + x",
        ),
    ],
)
def test_ones(function, helpers):
    """
    Test ones.
    """

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = range(10)
    circuit = compiler.compile(inputset, configuration)

    sample = random.randint(0, 11)
    helpers.check_execution(circuit, function, sample)
