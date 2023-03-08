"""
Tests of execution of zeros operation.
"""

import random

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: cnp.zero() + x,
            id="cnp.zero() + x",
        ),
        pytest.param(
            lambda x: cnp.zeros(()) + x,
            id="cnp.zeros(()) + x",
        ),
        pytest.param(
            lambda x: cnp.zeros(10) + x,
            id="cnp.zeros(10) + x",
        ),
        pytest.param(
            lambda x: cnp.zeros((10,)) + x,
            id="cnp.zeros((10,)) + x",
        ),
        pytest.param(
            lambda x: cnp.zeros((3, 2)) + x,
            id="cnp.zeros((3, 2)) + x",
        ),
    ],
)
def test_zeros(function, helpers):
    """
    Test zeros.
    """

    configuration = helpers.configuration()
    compiler = cnp.Compiler(function, {"x": "encrypted"})

    inputset = range(10)
    circuit = compiler.compile(inputset, configuration)

    sample = random.randint(0, 11)
    helpers.check_execution(circuit, function, sample)
