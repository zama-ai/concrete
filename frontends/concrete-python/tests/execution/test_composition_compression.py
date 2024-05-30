"""
Tests composition + compression
"""

import numpy as np

from concrete import fhe


def test_composable_with_input_compression(helpers):
    """
    Test that the composable circuit and compression works together
    """

    conf = helpers.configuration()
    if conf.parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        # Composability is for now only valid with multi
        return

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return (x**2) % 2**3

    conf.composable = True
    conf.compress_input_ciphertexts = True
    circuit = f.compile(fhe.inputset(fhe.uint3), conf)
    result = circuit.run(circuit.run(circuit.encrypt(2)))
    assert circuit.decrypt(result) == f(f(2))
