"""
Tests of `Circuit` class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from concrete.numpy import Client, ClientSpecs, EvaluationKeys, LookupTable, Server, compiler


def test_circuit_str(helpers):
    """
    Test `__str__` method of `Circuit` class.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(100)]
    circuit = f.compile(inputset, configuration.fork(p_error=6e-5))

    assert str(circuit) == circuit.graph.format()


def test_circuit_feedback(helpers):
    """
    Test feedback properties of `Circuit` class.
    """

    configuration = helpers.configuration()

    p_error = 0.1
    global_p_error = 0.05

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return np.sqrt(((x + y) ** 2) + 10).astype(np.int64)

    inputset = [(np.random.randint(0, 2**2), np.random.randint(0, 2**2)) for _ in range(100)]
    circuit = f.compile(inputset, configuration, p_error=p_error, global_p_error=global_p_error)

    assert isinstance(circuit.complexity, float)
    assert isinstance(circuit.size_of_secret_keys, int)
    assert isinstance(circuit.size_of_bootstrap_keys, int)
    assert isinstance(circuit.size_of_keyswitch_keys, int)
    assert isinstance(circuit.size_of_inputs, int)
    assert isinstance(circuit.size_of_outputs, int)
    assert isinstance(circuit.p_error, float)
    assert isinstance(circuit.global_p_error, float)

    assert circuit.p_error <= p_error
    assert circuit.global_p_error <= global_p_error


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


def test_client_server_api(helpers):
    """
    Test client/server API.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = [np.random.randint(0, 10, size=(3,)) for _ in range(10)]
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
            Client(client_specs, configuration.insecure_key_cache_location),
            Client.load(client_path, configuration.insecure_key_cache_location),
        ]

        for client in clients:
            args = client.encrypt([3, 8, 1])

            serialized_args = client.specs.serialize_public_args(args)
            serialized_evaluation_keys = client.evaluation_keys.serialize()

            unserialized_args = server.client_specs.unserialize_public_args(serialized_args)
            unserialized_evaluation_keys = EvaluationKeys.unserialize(serialized_evaluation_keys)

            result = server.run(unserialized_args, unserialized_evaluation_keys)
            serialized_result = server.client_specs.serialize_public_result(result)

            unserialized_result = client.specs.unserialize_public_result(serialized_result)
            output = client.decrypt(unserialized_result)

            assert np.array_equal(output, [45, 50, 43])

        with pytest.raises(RuntimeError) as excinfo:
            server.save("UNUSED", via_mlir=True)

        assert str(excinfo.value) == "Loaded server objects cannot be saved again via MLIR"

        server.cleanup()


def test_client_server_api_via_mlir(helpers):
    """
    Test client/server API.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = [np.random.randint(0, 10, size=(3,)) for _ in range(10)]
    circuit = function.compile(inputset, configuration.fork(jit=False))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        server_path = tmp_dir_path / "server.zip"
        circuit.server.save(server_path, via_mlir=True)

        client_path = tmp_dir_path / "client.zip"
        circuit.client.save(client_path)

        circuit.cleanup()

        server = Server.load(server_path)

        serialized_client_specs = server.client_specs.serialize()
        client_specs = ClientSpecs.unserialize(serialized_client_specs)

        clients = [
            Client(client_specs, configuration.insecure_key_cache_location),
            Client.load(client_path, configuration.insecure_key_cache_location),
        ]

        for client in clients:
            args = client.encrypt([3, 8, 1])

            serialized_args = client.specs.serialize_public_args(args)
            serialized_evaluation_keys = client.evaluation_keys.serialize()

            unserialized_args = server.client_specs.unserialize_public_args(serialized_args)
            unserialized_evaluation_keys = EvaluationKeys.unserialize(serialized_evaluation_keys)

            result = server.run(unserialized_args, unserialized_evaluation_keys)
            serialized_result = server.client_specs.serialize_public_result(result)

            unserialized_result = client.specs.unserialize_public_result(serialized_result)
            output = client.decrypt(unserialized_result)

            assert np.array_equal(output, [45, 50, 43])

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


@pytest.mark.parametrize("p_error", [0.75, 0.5, 0.4, 0.25, 0.2, 0.1, 0.01, 0.001])
@pytest.mark.parametrize("bit_width", [5])
@pytest.mark.parametrize("sample_size", [1_000_000])
@pytest.mark.parametrize("tolerance", [0.075])
def test_p_error_simulation(p_error, bit_width, sample_size, tolerance, helpers):
    """
    Test p_error simulation.
    """

    configuration = helpers.configuration().fork(global_p_error=None)

    table = LookupTable([0] + [x - 1 for x in range(1, 2**bit_width)])

    @compiler({"x": "encrypted"})
    def function(x):
        return table[x + 1]

    inputset = [np.random.randint(0, (2**bit_width) - 1, size=(sample_size,)) for _ in range(100)]
    circuit = function.compile(inputset, configuration=configuration, p_error=p_error)

    assert circuit.p_error < p_error

    sample = np.random.randint(0, (2**bit_width) - 1, size=(sample_size,))
    output = circuit.simulate(sample)

    errors = np.sum(output != sample)

    expected_number_of_errors_on_average = sample_size * circuit.p_error
    tolerated_difference = expected_number_of_errors_on_average * tolerance

    acceptable_number_of_errors = [
        round(expected_number_of_errors_on_average - tolerated_difference),
        round(expected_number_of_errors_on_average + tolerated_difference),
    ]
    assert acceptable_number_of_errors[0] <= errors <= acceptable_number_of_errors[1]


def test_circuit_run_with_unused_arg(helpers):
    """
    Test `encrypt_run_decrypt` method of `Circuit` class with unused arguments.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):  # pylint: disable=unused-argument
        return x + 10

    inputset = [
        (np.random.randint(2**3, 2**4), np.random.randint(2**4, 2**5)) for _ in range(100)
    ]
    circuit = f.compile(inputset, configuration)

    with pytest.raises(ValueError, match="Expected 2 inputs but got 1"):
        circuit.encrypt_run_decrypt(10)

    assert circuit.encrypt_run_decrypt(10, 0) == 20
    assert circuit.encrypt_run_decrypt(10, 10) == 20
    assert circuit.encrypt_run_decrypt(10, 20) == 20
