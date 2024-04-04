"""
Tests of everything related to multi-circuit.
"""
import numpy as np
import pytest

from concrete import fhe

# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-argument
# pylint: disable=unused-variable, no-member
# ruff: noqa: N805


def test_empty_module():
    """
    Test that defining a module without functions is an error.
    """
    with pytest.raises(
        RuntimeError, match="Tried to define an @fhe.module without any @fhe.function"
    ):

        @fhe.module()
        class Module:
            def square(x):
                return x**2


def test_call_clear_circuits():
    """
    Test that calling clear functions works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def square(x):
            return x**2

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def add_sub(x, y):
            return (x + y), (x - y)

        @fhe.function({"x": "encrypted", "y": "encrypted"})
        def mul(x, y):
            return x * y

    assert Module.square(2) == 4
    assert Module.add_sub(2, 3) == (5, -1)
    assert Module.mul(3, 4) == 12


def test_compile():
    """
    Test that compiling a module works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    Module.compile({"inc": inputset, "dec": inputset}, verbose=True)


def test_compiled_clear_call():
    """
    Test that cleartext execution works on compiled objects.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile(
        {"inc": inputset, "dec": inputset},
    )

    assert module.inc(5) == 6
    assert module.dec(5) == 4


def test_encrypted_execution():
    """
    Test that encrypted execution works.
    """

    @fhe.module()
    class Module:
        @fhe.function({"x": "encrypted"})
        def inc(x):
            return x + 1 % 20

        @fhe.function({"x": "encrypted"})
        def dec(x):
            return x - 1 % 20

    inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
    module = Module.compile(
        {"inc": inputset, "dec": inputset},
    )

    x = 5
    x_enc = module.inc.encrypt(x)
    x_inc_enc = module.inc.run(x_enc)
    x_inc = module.inc.decrypt(x_inc_enc)
    assert x_inc == 6

    x_inc_dec_enc = module.dec.run(x_inc_enc)
    x_inc_dec = module.dec.decrypt(x_inc_dec_enc)
    assert x_inc_dec == 5

    for _ in range(10):
        x_enc = module.inc.run(x_enc)
    x_dec = module.inc.decrypt(x_enc)
    assert x_dec == 15
