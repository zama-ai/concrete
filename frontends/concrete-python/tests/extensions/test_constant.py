"""
Tests of 'constant' extension.
"""

import numpy as np

from concrete import fhe


def test_constant_scalar(helpers):
    """
    Test that fhe.constant works with scalars.
    """
    configuration = helpers.configuration()

    @fhe.compiler({"x": "clear", "y": "encrypted"})
    def f(x, y):
        return fhe.constant(x) + y

    inputset = [(np.random.randint(0, 2**5), np.random.randint(0, 2**5)) for _ in range(100)]
    circuit = f.compile(inputset, configuration.fork())
    x = np.random.randint(0, 2**5)
    y = np.random.randint(0, 2**5)
    assert circuit.encrypt_run_decrypt(x, y) == x + y


def test_constant_tensor(helpers):
    """
    Test that fhe.constant works with arrays.
    """
    configuration = helpers.configuration()

    @fhe.compiler({"x": "clear", "y": "encrypted"})
    def f(x, y):
        return fhe.constant(x) + y

    inputset = [
        (np.random.randint(0, 2**5, size=10), np.random.randint(0, 2**5, size=10))
        for _ in range(100)
    ]
    circuit = f.compile(inputset, configuration.fork())
    x = np.random.randint(0, 2**5, size=10)
    y = np.random.randint(0, 2**5, size=10)
    assert np.all(circuit.encrypt_run_decrypt(x, y) == x + y)
