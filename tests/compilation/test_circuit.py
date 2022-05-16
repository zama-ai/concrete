"""
Tests of `Circuit` class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from concrete.numpy import Client, ClientSpecs, Server
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
        "Expected argument 0 to be EncryptedScalar<uint6> but it's EncryptedScalar<int1>"
    )

    # with negative argument 1
    # ------------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(1, -11)

    assert str(excinfo.value) == (
        "Expected argument 1 to be EncryptedScalar<uint6> but it's EncryptedScalar<int5>"
    )

    # with large argument 0
    # ---------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(100, 10)

    assert str(excinfo.value) == (
        "Expected argument 0 to be EncryptedScalar<uint6> but it's EncryptedScalar<uint7>"
    )

    # with large argument 1
    # ---------------------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(1, 100)

    assert str(excinfo.value) == (
        "Expected argument 1 to be EncryptedScalar<uint6> but it's EncryptedScalar<uint7>"
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


def test_client_server_api(helpers):
    """
    Test client/server API.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    circuit = function.compile(inputset, configuration.fork(jit=False))

    # for coverage
    circuit.keygen()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        server_path = tmp_dir_path / "server.zip"
        circuit.server.save(server_path)

        client_path = tmp_dir_path / "client.zip"
        circuit.client.save(client_path)

        circuit.cleanup()

        server = Server.load(server_path)

        serialized_client_specs = server.client_specs.serialize()
        client_specs = ClientSpecs.unserialize(serialized_client_specs)

        clients = [
            Client(client_specs, configuration.insecure_keycache_location),
            Client.load(client_path, configuration.insecure_keycache_location),
        ]

        for client in clients:
            args = client.encrypt(4)
            serialized_args = client.specs.serialize_public_args(args)

            result = server.unserialize_and_run(serialized_args)
            serialized_result = server.client_specs.serialize_public_result(result)

            output = client.unserialize_and_decrypt(serialized_result)
            assert output == 46

        server.cleanup()


def test_bad_server_save(helpers):
    """
    Test `save` method of `Server` class with bad parameters.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    circuit = function.compile(inputset, configuration)

    with pytest.raises(RuntimeError) as excinfo:
        circuit.server.save("test.zip")

    assert str(excinfo.value) == "Just-in-Time compilation cannot be saved"
