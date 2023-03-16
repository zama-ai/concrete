"""
Tests of execution of dot operation.
"""

import numpy as np
import pytest

from concrete import fhe


@pytest.mark.parametrize(
    "size",
    [1, 4, 6, 10],
)
def test_dot(size, helpers):
    """
    Test dot.
    """

    configuration = helpers.configuration()

    bound = int(np.floor(np.sqrt(127 / size)))
    cst = np.random.randint(0, bound, size=(size,))

    @fhe.compiler({"x": "encrypted"})
    def left_function(x):
        return np.dot(x, cst)

    @fhe.compiler({"x": "encrypted"})
    def right_function(x):
        return np.dot(cst, x)

    @fhe.compiler({"x": "encrypted"})
    def method(x):
        return x.dot(cst)

    inputset = [np.random.randint(0, bound, size=(size,)) for i in range(100)]

    left_function_circuit = left_function.compile(inputset, configuration)
    right_function_circuit = right_function.compile(inputset, configuration)
    method_circuit = method.compile(inputset, configuration)

    sample = np.random.randint(0, bound, size=(size,))

    helpers.check_execution(left_function_circuit, left_function, sample)
    helpers.check_execution(right_function_circuit, right_function, sample)
    helpers.check_execution(method_circuit, method, sample)
