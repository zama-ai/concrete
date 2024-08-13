"""
Tests composition + compression
"""

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


def test_composable_with_input_compression_decrypt_fresh_encrypted(helpers):
    """
    Test that we can decrypt a value which has just been encrypted with compression,
    it requires composability as we need to ensure input and output partitions are the same.

    https://github.com/zama-ai/concrete-internal/issues/758
    """
    conf = helpers.configuration()
    if conf.parameter_selection_strategy != fhe.ParameterSelectionStrategy.MULTI:
        # Composability is for now only valid with multi
        return

    @fhe.compiler({"x": "encrypted"})
    def f(x):
        return x**2

    inputset = [2, 3, 4, 5]
    conf.composable = True
    conf.compress_input_ciphertexts = True
    circuit = f.compile(inputset, conf)
    assert circuit.decrypt(circuit.encrypt(2)) == 2
