"""
Tests of `Circuit` class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from concrete.fhe import Client, ClientSpecs, EvaluationKeys, LookupTable, Server, Value, compiler


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

    # with None
    # ---------

    with pytest.raises(ValueError) as excinfo:
        circuit.encrypt_run_decrypt(None, 10)

    assert str(excinfo.value) == "Expected argument 0 to be an fhe.Value but it's None"

    # with non Value
    # --------------

    with pytest.raises(ValueError) as excinfo:
        _, b = circuit.encrypt(None, 10)
        circuit.run({"yes": "no"}, b)

    assert str(excinfo.value) == "Expected argument 0 to be an fhe.Value but it's dict"


def test_circuit_separate_args(helpers):
    """
    Test running circuit with separately encrypted args.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def function(x, y):
        return x + y

    inputset = [
        (
            np.random.randint(0, 10, size=()),
            np.random.randint(0, 10, size=(3,)),
        )
        for _ in range(10)
    ]
    circuit = function.compile(inputset, configuration)

    x = 4
    y = [1, 2, 3]

    x_encrypted, _ = circuit.encrypt(x, None)
    _, y_encrypted = circuit.encrypt(None, y)

    x_plus_y_encrypted = circuit.run(x_encrypted, y_encrypted)
    x_plus_y = circuit.decrypt(x_plus_y_encrypted)

    assert np.array_equal(x_plus_y, x + np.array(y))


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
        client_specs = ClientSpecs.deserialize(serialized_client_specs)

        clients = [
            Client(client_specs, configuration.insecure_key_cache_location),
            Client.load(client_path, configuration.insecure_key_cache_location),
        ]

        for client in clients:
            arg = client.encrypt([3, 8, 1])

            serialized_arg = arg.serialize()
            serialized_evaluation_keys = client.evaluation_keys.serialize()

            deserialized_arg = Value.deserialize(serialized_arg)
            deserialized_evaluation_keys = EvaluationKeys.deserialize(serialized_evaluation_keys)

            result = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys)
            serialized_result = result.serialize()

            deserialized_result = Value.deserialize(serialized_result)
            output = client.decrypt(deserialized_result)

            assert np.array_equal(output, [45, 50, 43])

        with pytest.raises(RuntimeError) as excinfo:
            server.save("UNUSED", via_mlir=True)

        assert str(excinfo.value) == "Loaded server objects cannot be saved again via MLIR"

        server.cleanup()


def test_client_server_api_crt(helpers):
    """
    Test client/server API on a CRT circuit.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted"})
    def function(x):
        return x**2

    inputset = [np.random.randint(0, 200, size=(3,)) for _ in range(10)]
    circuit = function.compile(inputset, configuration.fork(jit=False))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        server_path = tmp_dir_path / "server.zip"
        circuit.server.save(server_path)

        client_path = tmp_dir_path / "client.zip"
        circuit.client.save(client_path)

        server = Server.load(server_path)

        serialized_client_specs = server.client_specs.serialize()
        client_specs = ClientSpecs.deserialize(serialized_client_specs)

        clients = [
            Client(client_specs, configuration.insecure_key_cache_location),
            Client.load(client_path, configuration.insecure_key_cache_location),
        ]

        for client in clients:
            arg = client.encrypt([100, 150, 10])

            serialized_arg = arg.serialize()
            serialized_evaluation_keys = client.evaluation_keys.serialize()

            deserialized_arg = Value.deserialize(serialized_arg)
            deserialized_evaluation_keys = EvaluationKeys.deserialize(serialized_evaluation_keys)

            result = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys)
            serialized_result = result.serialize()

            deserialized_result = Value.deserialize(serialized_result)
            output = client.decrypt(deserialized_result)

            assert np.array_equal(output, [100**2, 150**2, 10**2])


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
        client_specs = ClientSpecs.deserialize(serialized_client_specs)

        clients = [
            Client(client_specs, configuration.insecure_key_cache_location),
            Client.load(client_path, configuration.insecure_key_cache_location),
        ]

        for client in clients:
            arg = client.encrypt([3, 8, 1])

            serialized_arg = arg.serialize()
            serialized_evaluation_keys = client.evaluation_keys.serialize()

            deserialized_arg = Value.deserialize(serialized_arg)
            deserialized_evaluation_keys = EvaluationKeys.deserialize(serialized_evaluation_keys)

            result = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys)
            serialized_result = result.serialize()

            deserialized_result = Value.deserialize(serialized_result)
            output = client.decrypt(deserialized_result)

            assert np.array_equal(output, [45, 50, 43])

        server.cleanup()


def test_bad_server_save(helpers):
    """
    Test `save` method of `Server` class with bad parameters.
    """

    configuration = helpers.configuration().fork(jit=True)

    @compiler({"x": "encrypted"})
    def function(x):
        return x + 42

    inputset = range(10)
    circuit = function.compile(inputset, configuration)

    with pytest.raises(RuntimeError) as excinfo:
        circuit.server.save("test.zip")

    assert str(excinfo.value) == "Just-in-Time compilation cannot be saved"


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


@pytest.mark.dataflow
def test_dataflow_circuit(helpers):
    """
    Test execution with dataflow_parallelize=True.
    """

    configuration = helpers.configuration().fork(dataflow_parallelize=True)

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return (x**2) + (y // 2)

    inputset = [(np.random.randint(0, 2**3), np.random.randint(0, 2**3)) for _ in range(100)]
    circuit = f.compile(inputset, configuration)

    assert circuit.encrypt_run_decrypt(5, 6) == 28


def test_circuit_sim_disabled(helpers):
    """
    Test attempt to simulate without enabling fhe simulation.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(2)]
    circuit = f.compile(inputset, configuration)

    assert circuit.simulate(*inputset[0]) == f(*inputset[0])


def test_circuit_fhe_exec_disabled(helpers):
    """
    Test attempt to run fhe execution without it being enabled.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(2)]
    circuit = f.compile(inputset, configuration.fork(fhe_execution=False))

    assert circuit.encrypt_run_decrypt(*inputset[0]) == f(*inputset[0])


def test_circuit_fhe_exec_no_eval_keys(helpers):
    """
    Test attempt to run fhe execution without eval keys.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        return x + y

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(2)]
    circuit = f.compile(inputset, configuration)
    with pytest.raises(RuntimeError) as excinfo:
        # as we can't encrypt, we just pass plain inputs, and it should lead to the expected error
        encrypted_args = inputset[0]
        circuit.server.run(*encrypted_args)
    assert (
        str(excinfo.value) == "Expected evaluation keys to be provided when not in simulation mode"
    )


def test_circuit_eval_graph_scalar(helpers):
    """
    Test evaluation of the graph.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        lut = LookupTable(list(range(128)))
        return lut[x + y]

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(2)]
    circuit = f.compile(inputset, configuration.fork(fhe_simulation=False, fhe_execution=False))
    assert f(*inputset[0]) == circuit.graph(*inputset[0], p_error=0.01)


def test_circuit_eval_graph_tensor(helpers):
    """
    Test evaluation of the graph.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        lut = LookupTable(list(range(128)))
        return lut[x + y]

    inputset = [
        (
            np.random.randint(0, 2**4, size=[2, 2]),
            np.random.randint(0, 2**5, size=[2, 2]),
        )
        for _ in range(2)
    ]
    circuit = f.compile(inputset, configuration.fork(fhe_simulation=False, fhe_execution=False))
    assert np.all(f(*inputset[0]) == circuit.graph(*inputset[0], p_error=0.01))


def test_circuit_compile_sim_only(helpers):
    """
    Test compiling with simulation only.
    """

    configuration = helpers.configuration()

    @compiler({"x": "encrypted", "y": "encrypted"})
    def f(x, y):
        lut = LookupTable(list(range(128)))
        return lut[x + y]

    inputset = [(np.random.randint(0, 2**4), np.random.randint(0, 2**5)) for _ in range(2)]
    circuit = f.compile(inputset, configuration.fork(fhe_simulation=True, fhe_execution=False))
    assert f(*inputset[0]) == circuit.simulate(*inputset[0])
