"""
Tests of execution of broadcast to operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "from_shape,to_shape",
    [
        pytest.param((), (2,)),
        pytest.param((), (2, 3)),
        pytest.param((3,), (2, 3)),
        pytest.param((3,), (4, 2, 3)),
        pytest.param((1, 2), (4, 3, 2)),
        pytest.param((3, 2), (4, 3, 2)),
        pytest.param((3, 1), (4, 3, 5)),
        pytest.param((3, 1, 4), (3, 2, 4)),
        pytest.param((3, 1, 1), (5, 3, 1, 3)),
    ],
)
def test_broadcast_to(from_shape, to_shape, helpers):
    """
    Test broadcast to.
    """

    configuration = helpers.configuration()
    for status in ["clear", "encrypted"]:
        if status == "encrypted":

            def function(x):
                return np.broadcast_to(x, to_shape)

        else:

            def function(x):
                return fhe.zero() + np.broadcast_to(x, to_shape)

        compiler = fhe.Compiler(function, {"x": status})

        inputset = [np.random.randint(0, 2**2, size=from_shape) for _ in range(100)]
        circuit = compiler.compile(inputset, configuration)

        sample = np.random.randint(0, 2**2, size=from_shape)
        helpers.check_execution(circuit, function, sample)
