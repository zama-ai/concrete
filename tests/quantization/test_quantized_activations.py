"""Tests for the quantized activation functions."""
import numpy
import pytest

from concrete.quantization import QuantizedArray, QuantizedReLU6, QuantizedSigmoid

N_BITS_ATOL_TUPLE_LIST = [
    (32, 10 ** -2),
    (28, 10 ** -2),
    (20, 10 ** -2),
    (16, 10 ** -1),
    (8, 10 ** -0),
    (5, 10 ** -0),
]


@pytest.mark.parametrize(
    "n_bits, atol",
    [pytest.param(n_bits, atol) for n_bits, atol in N_BITS_ATOL_TUPLE_LIST],
)
@pytest.mark.parametrize(
    "input_range",
    [pytest.param((-1, 1)), pytest.param((-2, 2)), pytest.param((-10, 10)), pytest.param((0, 20))],
)
@pytest.mark.parametrize(
    "input_shape",
    [pytest.param((10, 40, 20)), pytest.param((100, 400))],
)
@pytest.mark.parametrize(
    "quant_activation",
    [
        pytest.param(QuantizedSigmoid),
        pytest.param(QuantizedReLU6),
    ],
)
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
def test_activations(quant_activation, input_shape, input_range, n_bits, atol, is_signed):
    """Test activation functions."""
    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed)
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
    assert numpy.isclose(dequant_values.ravel(), expected_output.ravel(), atol=atol).all()
