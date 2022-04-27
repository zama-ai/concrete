"""
Tests of `Compiler` class.
"""

import pytest

from concrete.numpy.compilation import Compiler


def test_compiler_bad_init(helpers):
    """
    Test `__init__` method of `Compiler` class with bad parameters.
    """

    configuration = helpers.configuration()

    def f(x, y, z):
        return x + y + z

    # missing all
    # -----------

    with pytest.raises(ValueError) as excinfo:
        Compiler(f, {}, configuration=configuration)

    assert str(excinfo.value) == (
        "Encryption statuses of parameters 'x', 'y' and 'z' of function 'f' are not provided"
    )

    # missing x and y
    # ---------------

    with pytest.raises(ValueError) as excinfo:
        Compiler(f, {"z": "clear"}, configuration=configuration)

    assert str(excinfo.value) == (
        "Encryption statuses of parameters 'x' and 'y' of function 'f' are not provided"
    )

    # missing x
    # ---------

    with pytest.raises(ValueError) as excinfo:
        Compiler(f, {"y": "encrypted", "z": "clear"}, configuration=configuration)

    assert str(excinfo.value) == (
        "Encryption status of parameter 'x' of function 'f' is not provided"
    )

    # additional p
    # ------------

    # this is fine and `p` is just ignored

    Compiler(
        f,
        {"x": "encrypted", "y": "encrypted", "z": "clear", "p": "clear"},
        configuration=configuration,
    )


def test_compiler_bad_call(helpers):
    """
    Test `__call__` method of `Compiler` class with bad parameters.
    """

    configuration = helpers.configuration()

    def f(x, y, z):
        return x + y + z

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
            configuration=configuration,
        )
        compiler(1, 2, 3, invalid=4)

    assert str(excinfo.value) == "Calling function 'f' with kwargs is not supported"


def test_compiler_bad_trace(helpers):
    """
    Test `trace` method of `Compiler` class with bad parameters.
    """

    configuration = helpers.configuration()

    def f(x, y, z):
        return x + y + z

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
            configuration=configuration,
        )
        compiler.trace()

    assert str(excinfo.value) == "Tracing function 'f' without an inputset is not supported"


def test_compiler_bad_compile(helpers):
    """
    Test `compile` method of `Compiler` class with bad parameters.
    """

    configuration = helpers.configuration()

    def f(x, y, z):
        return x + y + z

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(
            f,
            {"x": "encrypted", "y": "encrypted", "z": "clear"},
            configuration=configuration,
        )
        compiler.compile()

    assert str(excinfo.value) == "Compiling function 'f' without an inputset is not supported"

    configuration.enable_unsafe_features = False

    with pytest.raises(RuntimeError) as excinfo:
        compiler = Compiler(lambda x: x, {"x": "encrypted"}, configuration=configuration)
        compiler.compile(virtual=True)

    assert str(excinfo.value) == (
        "Virtual compilation is not allowed without enabling unsafe features"
    )


def test_compiler_virtual_compile(helpers):
    """
    Test `compile` method of `Compiler` class with virtual=True.
    """

    configuration = helpers.configuration()

    def f(x):
        return x + 400

    compiler = Compiler(f, {"x": "encrypted"}, configuration=configuration)
    circuit = compiler.compile(inputset=range(400), virtual=True)

    assert circuit.encrypt_run_decrypt(200) == 600
