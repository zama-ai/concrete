"""
Tests of `Circuit` class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from concrete.numpy import Circuit
from concrete.numpy.compilation import compiler


def test_circuit_str(helpers):
    """
    Test `__str__` method of `Circuit` class.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(100)]
    circuit = f.compile(inputset, configuration)

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

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(100)]
    circuit = f.compile(inputset, configuration)

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

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(100)]
    circuit = f.compile(inputset, configuration)

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


def test_circuit_virtual_explicit_api(helpers):
    """
    Test `keygen`, `encrypt`, `run`, and `decrypt` methods of `Circuit` class with virtual circuit.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(100)]
    circuit = f.compile(inputset, configuration, virtual=True)

    with pytest.raises(RuntimeError) as excinfo:
        circuit.keygen()

    assert str(excinfo.value) == "Virtual circuits cannot use `keygen` method"

    with pytest.raises(RuntimeError) as excinfo:
        circuit.encrypt(1, 2)

    assert str(excinfo.value) == "Virtual circuits cannot use `encrypt` method"

    with pytest.raises(RuntimeError) as excinfo:
        circuit.run(None)

    assert str(excinfo.value) == "Virtual circuits cannot use `run` method"

    with pytest.raises(RuntimeError) as excinfo:
        circuit.decrypt(None)

    assert str(excinfo.value) == "Virtual circuits cannot use `decrypt` method"


def test_circuit_bad_save(helpers):
    """
    Test `save` method of `Circuit` class with bad parameters.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    circuit = function.compile(inputset, configuration)

    with pytest.raises(RuntimeError) as excinfo:
        circuit.save("circuit.zip")

    assert str(excinfo.value) == "JIT Circuits cannot be saved"


@pytest.mark.parametrize(
    "virtual",
    [False, True],
)
def test_circuit_save_load(virtual, helpers):
    """
    Test `save`, `load`, and `cleanup` methods of `Circuit` class.
    """

    configuration = helpers.configuration().fork(jit=False, virtual=virtual)

    def save(base):
        @compiler({"x": "encrypted"})
        def function(x):
            return x + 42

        inputset = range(10)
        circuit = function.compile(inputset, configuration)

        circuit.save(base / "circuit.zip")
        circuit.cleanup()

    def load(base):
        circuit = Circuit.load(base / "circuit.zip")

        helpers.check_str(
            """

%0 = x                  # EncryptedScalar<uint4>
%1 = 42                 # ClearScalar<uint6>
%2 = add(%0, %1)        # EncryptedScalar<uint6>
return %2

            """,
            str(circuit),
        )
        if virtual:
            helpers.check_str("Virtual circuits doesn't have MLIR.", circuit.mlir)
        else:
            helpers.check_str(
                """

module  {
  func @main(%arg0: !FHE.eint<6>) -> !FHE.eint<6> {
    %c42_i7 = arith.constant 42 : i7
    %0 = "FHE.add_eint_int"(%arg0, %c42_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
    return %0 : !FHE.eint<6>
  }
}

                """,
                circuit.mlir,
            )
        helpers.check_execution(circuit, lambda x: x + 42, 4)

        circuit.cleanup()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        save(path)
        load(path)
