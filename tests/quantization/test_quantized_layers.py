"""Tests for the quantized layers."""
import numpy
import pytest

from concrete.quantization import QuantizedArray, QuantizedLinear

# QuantizedLinear unstable with n_bits>23
# and hard to test with numpy.isclose with n_bits < 8
N_BITS_LIST = [20, 16, 8]


@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in N_BITS_LIST],
)
@pytest.mark.parametrize(
    "n_examples, n_features, n_neurons",
    [
        pytest.param(50, 3, 4),
        pytest.param(20, 500, 30),
        pytest.param(200, 300, 50),
        pytest.param(10000, 100, 1),
        pytest.param(10, 20, 1),
    ],
)
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
def test_quantized_linear(n_examples, n_features, n_neurons, n_bits, is_signed):
    """Test the quantization linear layer of numpy.array.

    With n_bits>>0 we expect the results of the quantized linear
    to be the same as the standard linear layer.
    """
    inputs = numpy.random.uniform(size=(n_examples, n_features))
    q_inputs = QuantizedArray(n_bits, inputs)

    # shape of weights: (n_features, n_neurons)
    weights = numpy.random.uniform(size=(n_features, n_neurons))
    q_weights = QuantizedArray(n_bits, weights, is_signed)

    bias = numpy.random.uniform(size=(1, n_neurons))
    q_bias = QuantizedArray(n_bits, bias, is_signed)

    # Define our QuantizedLinear layer
    q_linear = QuantizedLinear(n_bits, q_weights, q_bias)

    # Calibrate the Quantized layer
    q_linear.calibrate(inputs)

    expected_outputs = q_linear.q_out.values
    actual_output = q_linear(q_inputs).dequant()

    assert numpy.isclose(expected_outputs, actual_output, atol=10 ** -0).all()

    # Same test without bias
    q_linear = QuantizedLinear(n_bits, q_weights)

    # Calibrate the Quantized layer
    q_linear.calibrate(inputs)
    expected_outputs = q_linear.q_out.values
    actual_output = q_linear(q_inputs).dequant()

    assert numpy.isclose(expected_outputs, actual_output, atol=10 ** -0).all()
