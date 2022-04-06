"""
Tests of `Circuit` class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from concrete.numpy.compilation import compiler


def test_circuit_str(helpers):
    """
    Test `__str__` method of `Circuit` class.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"}, configuration=configuration)
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2 ** 4), np.random.randint(0, 2 ** 5)) for _ in range(100)]
    circuit = f.compile(inputset)

    assert str(circuit) == (
        """

%0 = x                  # EncryptedScalar<uint4>
%1 = y                  # EncryptedScalar<uint5>
%2 = add(%0, %1)        # EncryptedScalar<uint6>
return %2

        """.strip()
    )


def test_circuit_draw(helpers):
    """
    Test `draw` method of `Circuit` class.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"}, configuration=configuration)
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2 ** 4), np.random.randint(0, 2 ** 5)) for _ in range(100)]
    circuit = f.compile(inputset)

    with tempfile.TemporaryDirectory() as path:
        tmpdir = Path(path)

        png = tmpdir / "drawing.png"
        circuit.draw(save_to=png)

        assert png.exists()


def test_circuit_bad_run(helpers):
    """
    Test `run` method of `Circuit` class with bad parameters.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"}, configuration=configuration)
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2 ** 4), np.random.randint(0, 2 ** 5)) for _ in range(100)]
    circuit = f.compile(inputset)

    # with 1 argument
    # ---------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(1)

    assert str(excinfo.value) == "Expected 2 inputs but got 1"

    # with 3 arguments
    # ----------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(1, 2, 3)

    assert str(excinfo.value) == "Expected 2 inputs but got 3"

    # with negative argument 0
    # ------------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(-1, 11)

    assert str(excinfo.value) == (
        "Expected argument 0 to be EncryptedScalar<uint4> but it's EncryptedScalar<int1>"
    )

    # with negative argument 1
    # ------------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(1, -11)

    assert str(excinfo.value) == (
        "Expected argument 1 to be EncryptedScalar<uint5> but it's EncryptedScalar<int5>"
    )

    # with large argument 0
    # ---------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(100, 10)

    assert str(excinfo.value) == (
        "Expected argument 0 to be EncryptedScalar<uint4> but it's EncryptedScalar<uint7>"
    )

    # with large argument 1
    # ---------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(1, 100)

    assert str(excinfo.value) == (
        "Expected argument 1 to be EncryptedScalar<uint5> but it's EncryptedScalar<uint7>"
    )
