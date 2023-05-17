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
    def dot_enc_enc_function(x):
        return np.dot(x, cst)

    @fhe.compiler({"x": "encrypted"})
    def right_function(x):
        return np.dot(cst, x)

    @fhe.compiler({"x": "encrypted"})
    def method(x):
        return x.dot(cst)

    inputset = [np.random.randint(0, bound, size=(size,)) for i in range(100)]

    dot_enc_enc_function_circuit = dot_enc_enc_function.compile(inputset, configuration)
    right_function_circuit = right_function.compile(inputset, configuration)
    method_circuit = method.compile(inputset, configuration)

    sample = np.random.randint(0, bound, size=(size,))

    helpers.check_execution(dot_enc_enc_function_circuit, dot_enc_enc_function, sample)
    helpers.check_execution(right_function_circuit, right_function, sample)
    helpers.check_execution(method_circuit, method, sample)


@pytest.mark.parametrize(
    "size",
    [1, 10],
)
@pytest.mark.parametrize(
    "bitwidth",
    [2, 6],
)
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("negative_only", [True, False])
def test_dot_enc_enc(size, bitwidth, negative_only, signed, helpers):
    """
    Test dot.
    """

    configuration = helpers.configuration()

    minv = 0 if not signed else -(2 ** (bitwidth - 1))

    # +1 since randint max is not inclusive
    maxv = 2**bitwidth if not signed else 2 ** (bitwidth - 1)
    if negative_only:
        maxv = 1

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def dot_enc_enc_function(x, y):
        return np.dot(x, y)

    inputset = [
        (np.random.randint(minv, maxv, size=(size,)), np.random.randint(minv, maxv, size=(size,)))
        for i in range(100)
    ]

    dot_enc_enc_function_circuit = dot_enc_enc_function.compile(inputset, configuration)

    sample = [
        np.random.randint(minv, maxv, size=(size,)),
        np.random.randint(minv, maxv, size=(size,)),
    ]

    helpers.check_execution(dot_enc_enc_function_circuit, dot_enc_enc_function, sample, retries=3)
