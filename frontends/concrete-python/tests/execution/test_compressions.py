"""
Basic tests of compressions feature.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from concrete import fhe
from concrete.fhe import Client, ClientSpecs, EvaluationKeys, LookupTable, Server, Value


def test_circuit_compress_input_ciphertexts(helpers):
    """
    Test running circuit with compressed inpu ciphertexts
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def function(x, y):
        return (x + y) ** 2

    inputset = fhe.inputset(fhe.uint4, fhe.uint4, size=10)
    print(f"inputset  {inputset}")
    circuit_without_compression = function.compile(inputset, compress_input_ciphertexts=False)
    circuit_with_compression = function.compile(
        inputset, compress_input_ciphertexts=True
    )

    x, y = circuit_without_compression.encrypt(1, 2)
    x_compressed, y_compressed = circuit_with_compression.encrypt(1, 2)

    assert len(x_compressed.serialize()) < len(x.serialize())
    assert len(y_compressed.serialize()) < len(y.serialize())

    assert circuit_with_compression.decrypt(circuit_with_compression.run(x_compressed, y_compressed)) == circuit_without_compression.decrypt(circuit_without_compression.run(x, y))

def test_circuit_compress_evaluation_keys(helpers):
    """
    Test running circuit with compressed evaluation keys
    """

    configuration = helpers.configuration()

    @fhe.compiler({"x": "encrypted", "y": "encrypted"})
    def function(x, y):
        return (x + y) ** 2

    inputset = fhe.inputset(fhe.uint4, fhe.uint4, size=10)
    print(f"inputset  {inputset}")
    circuit_without_compression = function.compile(inputset, compress_evaluation_keys=False)
    circuit_with_compression = function.compile(
        inputset, compress_evaluation_keys=True
    )

    evaluation_keys = circuit_without_compression.client.evaluation_keys
    evaluation_keys_compressed = circuit_with_compression.client.evaluation_keys


    assert len(evaluation_keys_compressed.serialize()) < len(evaluation_keys.serialize())
