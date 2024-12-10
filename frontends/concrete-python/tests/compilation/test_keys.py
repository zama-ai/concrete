"""
Tests of `Keys` class.
"""

import tempfile
from pathlib import Path

import pytest

from concrete import fhe


def test_keys_save_load(helpers):
    """
    Test saving and loading keys.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    inputset = range(10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        keys_path = tmp_dir_path / "keys"

        circuit1 = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))
        circuit1.keygen()

        sample = circuit1.encrypt(5)
        evaluation = circuit1.run(sample)

        circuit1.keys.save(str(keys_path))
        circuit2 = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))
        circuit2.keys.load(str(keys_path))

        assert circuit2.decrypt(evaluation) == 25


def test_keys_bad_save_load(helpers):
    """
    Test saving/loading keys where location is (not) empty.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        keys_path = tmp_dir_path / "keys"

        inputset = range(10)
        circuit = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))

        with pytest.raises(ValueError) as excinfo:
            circuit.keys.load(keys_path)

        expected_message = f"Unable to load keys from {keys_path} because it doesn't exist"
        helpers.check_str(expected_message, str(excinfo.value))

        with open(keys_path, "w", encoding="utf-8") as f:
            f.write("foo")

        circuit.keys.generate()
        with pytest.raises(ValueError) as excinfo:
            circuit.keys.save(keys_path)

        expected_message = f"Unable to save keys to {keys_path} because it already exists"
        helpers.check_str(expected_message, str(excinfo.value))


def test_keys_load_if_exists_generate_and_save_otherwise(helpers):
    """
    Test saving and loading keys using `load_if_exists_generate_and_save_otherwise`.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    inputset = range(10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        keys_path = tmp_dir_path / "keys"

        circuit1 = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))
        circuit1.keys.load_if_exists_generate_and_save_otherwise(str(keys_path))

        sample = circuit1.encrypt(5)
        evaluation = circuit1.run(sample)

        circuit2 = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))
        circuit2.keys.load_if_exists_generate_and_save_otherwise(str(keys_path))

        assert circuit2.decrypt(evaluation) == 25


def test_keys_bad_seed(helpers):
    """
    Test serializing and deserializing keys.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    inputset = range(10)

    circuit = f.compile(inputset, helpers.configuration())
    server = circuit.server

    client1 = fhe.Client(server.client_specs)

    with pytest.raises(ValueError) as excinfo:
        client1.keys.generate(secret_seed=-1)
    assert str(excinfo.value) == "secret_seed must be a positive 128 bits integer"

    with pytest.raises(ValueError) as excinfo:
        client1.keys.generate(secret_seed=2**128)
    assert str(excinfo.value) == "secret_seed must be a positive 128 bits integer"

    with pytest.raises(ValueError) as excinfo:
        client1.keys.generate(encryption_seed=2**128)
    assert str(excinfo.value) == "encryption_seed must be a positive 128 bits integer"

    with pytest.raises(ValueError) as excinfo:
        client1.keys.generate(encryption_seed=-1)
    assert str(excinfo.value) == "encryption_seed must be a positive 128 bits integer"


def test_keys_serialize_deserialize(helpers):
    """
    Test serializing and deserializing keys.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    inputset = range(10)

    circuit = f.compile(inputset, helpers.configuration())
    server = circuit.server

    client1 = fhe.Client(server.client_specs)
    client1.keys.generate()

    sample = client1.encrypt(5)
    evaluation = server.run(sample, evaluation_keys=client1.evaluation_keys)

    client2 = fhe.Client(server.client_specs)
    client2.keys = fhe.Keys.deserialize(client1.keys.serialize())

    assert client2.decrypt(evaluation) == 25


def test_keys_serialize_before_generation(helpers):
    """
    Test serialization of keys before their generation.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x + 42

    inputset = range(10)
    circuit = f.compile(inputset, configuration=helpers.configuration())

    with pytest.raises(RuntimeError) as excinfo:
        circuit.keys.serialize()

    expected_message = "Keys cannot be serialized before they are generated"
    helpers.check_str(expected_message, str(excinfo.value))

    with pytest.raises(RuntimeError) as excinfo:
        # path doesn't matter as it will fail
        circuit.keys.serialize_to_file(Path("_keys_file"))

    expected_message = "Keys cannot be serialized before they are generated"
    helpers.check_str(expected_message, str(excinfo.value))


def test_keys_generate_manual_seed(helpers):
    """
    Test key generation with custom seed.
    """

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    inputset = range(10)

    circuit = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))
    assert not circuit.keys.are_generated
    circuit.keygen(seed=42, encryption_seed=24)
    assert circuit.keys.are_generated

    sample = circuit.encrypt(5)
    evaluation = circuit.run(sample)

    same_circuit = f.compile(inputset, helpers.configuration().fork(use_insecure_key_cache=False))
    assert not same_circuit.keys.are_generated
    same_circuit.keygen(seed=42)
    assert same_circuit.keys.are_generated

    assert same_circuit.decrypt(evaluation) == 25
