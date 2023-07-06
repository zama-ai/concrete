"""
Tests of execution of reshape operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "shape,newshape",
    [
        pytest.param(
            (12,),
            (12, 1),
        ),
        pytest.param(
            (12,),
            (1, 12),
        ),
        pytest.param(
            (12,),
            (3, 4),
        ),
        pytest.param(
            (12,),
            (3, 2, 2),
        ),
        pytest.param(
            (3, 4),
            12,
        ),
        pytest.param(
            (3, 4),
            (12,),
        ),
        pytest.param(
            (3, 4),
            (4, 3),
        ),
        pytest.param(
            (3, 4),
            (2, 2, 3),
        ),
        pytest.param(
            (3, 4),
            (2, 3, 2),
        ),
        pytest.param(
            (3, 4),
            (3, 2, 2),
        ),
        pytest.param(
            (3, 4),
            (3, 1, 4),
        ),
        pytest.param(
            (3, 4),
            (12, 1),
        ),
        pytest.param(
            (3, 4),
            (-1,),
        ),
        pytest.param(
            (3, 4),
            -1,
        ),
        pytest.param(
            (2, 2, 3),
            (3, 4),
        ),
        pytest.param(
            (2, 2, 3),
            (4, 3),
        ),
        pytest.param(
            (2, 2, 3),
            (3, 2, 2),
        ),
        pytest.param(
            (2, 3, 4, 5, 6),
            (6, 4, 30),
        ),
        pytest.param(
            (6, 4, 30),
            (2, 3, 4, 5, 6),
        ),
        pytest.param(
            (2, 3, 4, 5, 6),
            (2, 60, 6),
        ),
        pytest.param(
            (2, 60, 6),
            (2, 3, 4, 5, 6),
        ),
        pytest.param(
            (2, 3, 2, 3, 4),
            (6, 6, -1),
        ),
        pytest.param(
            (2, 3, 2, 3, 4),
            (6, -1, 12),
        ),
        pytest.param(
            (2, 3, 2, 3, 4),
            (-1, 18, 4),
        ),
    ],
)
def test_reshape(shape, newshape, helpers):
    """
    Test reshape.
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return np.reshape(x, newshape)

    @fhe.compiler({"x": "encrypted"})
    def method(x):
        return x.reshape(newshape)

    @fhe.compiler({"x": "encrypted"})
    def args(x):
        return x.reshape(*(newshape if isinstance(newshape, tuple) else (newshape,)))

    inputset = [np.random.randint(0, 2**5, size=shape) for i in range(100)]

    function_circuit = function.compile(inputset, configuration)
    method_circuit = method.compile(inputset, configuration)
    args_circuit = args.compile(inputset, configuration)

    sample = np.random.randint(0, 2**5, size=shape)

    helpers.check_execution(function_circuit, function, sample)
    helpers.check_execution(method_circuit, method, sample)
    helpers.check_execution(args_circuit, args, sample)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            (12,),
        ),
        pytest.param(
            (3, 4),
        ),
        pytest.param(
            (2, 2, 3),
        ),
        pytest.param(
            (2, 3, 4, 5, 6),
        ),
    ],
)
def test_flatten(shape, helpers):
    """
    Test flatten.
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return x.flatten()

    inputset = [np.random.randint(0, 2**5, size=shape) for i in range(100)]
    circuit = function.compile(inputset, configuration)

    sample = np.random.randint(0, 2**5, size=shape)
    helpers.check_execution(circuit, function, sample)
