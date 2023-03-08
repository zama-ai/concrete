"""
Tests of execution of ones operation.
"""

import random

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: cnp.one() + x,
            id="cnp.one() + x",
        ),
        pytest.param(
            lambda x: cnp.ones(()) + x,
            id="cnp.ones(()) + x",
        ),
        pytest.param(
            lambda x: cnp.ones(10) + x,
            id="cnp.ones(10) + x",
        ),
        pytest.param(
            lambda x: cnp.ones((10,)) + x,
            id="cnp.ones((10,)) + x",
        ),
        pytest.param(
            lambda x: cnp.ones((3, 2)) + x,
            id="cnp.ones((3, 2)) + x",
        ),
    ],
)
def test_ones(function, helpers):
    """
    Test ones.
    """

    configuration = helpers.configuration()
    compiler = cnp.Compiler(function, {"x": "encrypted"})

    inputset = range(10)
    circuit = compiler.compile(inputset, configuration)

    sample = random.randint(0, 11)
    helpers.check_execution(circuit, function, sample)
