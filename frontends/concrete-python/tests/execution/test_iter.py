"""
Tests of execution of iteration of tracer.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize("shape", [(3,), (3, 2), (3, 2, 4)])
def test_iter(shape, helpers):
    """
    Test iteration of tracers.
    """

    def function(x):
        result = fhe.zeros(x.shape[1:])
        for value in x:
            result += value
        return result

    configuration = helpers.configuration()
    compiler = fhe.Compiler(function, {"x": "encrypted"})

    inputset = [np.random.randint(0, 2**2, size=shape) for _ in range(100)]
    circuit = compiler.compile(inputset, configuration)

    sample = np.random.randint(0, 2**2, size=shape)
    helpers.check_execution(circuit, function, sample)
