"""Tests for the quantized activation functions."""
import numpy
import pytest

from concrete.quantization import QuantizedArray, QuantizedSigmoid

N_BITS_ATOL_TUPLE_LIST = [
    (32, 10 ** -2),
    (28, 10 ** -2),
    (20, 10 ** -2),
    (16, 10 ** -1),
    (8, 10 ** -0),
    (4, 10 ** -0),
]


@pytest.mark.parametrize(
    "n_bits, atol",
    [pytest.param(n_bits, atol) for n_bits, atol in N_BITS_ATOL_TUPLE_LIST],
)
@pytest.mark.parametrize(
    "quant_activation, values",
    [pytest.param(QuantizedSigmoid, numpy.random.uniform(size=(10, 40, 20)))],
)
def test_activations(quant_activation, values, n_bits, atol):
    """Test activation functions."""
    q_inputs = QuantizedArray(n_bits, values)
    quant_sigmoid = quant_activation(n_bits)
    quant_sigmoid.calibrate(values)
    expected_output = quant_sigmoid.q_out.values
    q_output = quant_sigmoid(q_inputs)
    qvalues = q_output.qvalues

    # Quantized values must be contained between 0 and 2**n_bits - 1.
    assert numpy.max(qvalues) <= 2 ** n_bits - 1
    assert numpy.min(qvalues) >= 0

    # Dequantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    assert numpy.isclose(dequant_values, expected_output, atol=atol).all()
