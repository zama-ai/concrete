"""Tests for the quantized layers."""
import numpy
import pytest

from concrete.quantization import QuantizedArray, QuantizedLinear

# QuantizedLinear unstable with n_bits>23.
N_BITS_LIST = [20, 16, 8, 4]


@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in N_BITS_LIST],
)
@pytest.mark.parametrize(
    "n_examples, n_features, n_neurons",
    [
        pytest.param(20, 500, 30),
        pytest.param(200, 300, 50),
        pytest.param(10000, 100, 1),
        pytest.param(10, 20, 1),
    ],
)
def test_quantized_linear(n_examples, n_features, n_neurons, n_bits):
    """Test the quantization linear layer of numpy.array.

    With n_bits>>0 we expect the results of the quantized linear
    to be the same as the standard linear layer.
    """
    inputs = numpy.random.uniform(size=(n_examples, n_features))
    q_inputs = QuantizedArray(n_bits, inputs)

    # shape of weights: (n_examples, n_features, n_neurons)
    weights = numpy.random.uniform(size=(n_neurons, n_features))
    q_weights = QuantizedArray(n_bits, weights)

    bias = numpy.random.uniform(size=(n_neurons))
    q_bias = QuantizedArray(n_bits, bias)

    # Define our QuantizedLinear layer
    q_linear = QuantizedLinear(n_bits, q_weights, q_bias)

    # Calibrate the Quantized layer
    q_linear.calibrate(inputs)
    expected_outputs = q_linear.q_out.values
    actual_output = q_linear(q_inputs).dequant()

    assert numpy.isclose(expected_outputs, actual_output, rtol=10 ** -1).all()

    # Same test without bias
    q_linear = QuantizedLinear(n_bits, q_weights)

    # Calibrate the Quantized layer
    q_linear.calibrate(inputs)
    expected_outputs = q_linear.q_out.values
    actual_output = q_linear(q_inputs).dequant()

    assert numpy.isclose(expected_outputs, actual_output, rtol=10 ** -1).all()
