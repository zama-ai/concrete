"""Test Neural Networks compilations"""
import numpy
import pytest
from torch import nn

from concrete.quantization import PostTrainingAffineQuantization, QuantizedArray
from concrete.torch import NumpyModule

# INPUT_OUTPUT_FEATURE is the number of input and output of each of the network layers.
# (as well as the input of the network itself)
INPUT_OUTPUT_FEATURE = [1, 2, 3]


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self, input_output):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=input_output)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=input_output, out_features=input_output)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)

        return out


@pytest.mark.parametrize(
    "model",
    [pytest.param(FC, marks=pytest.mark.xfail())],  # [TEMPORARY] xfail #1042
)
@pytest.mark.parametrize(
    "input_output_feature",
    [
        pytest.param(input_output_feature, marks=pytest.mark.xfail())  # [TEMPORARY] xfail #1042
        for input_output_feature in INPUT_OUTPUT_FEATURE
    ],
)
def test_quantized_module_compilation(
    input_output_feature, model, seed_torch, default_compilation_configuration
):
    """Test a neural network compilation for FHE inference."""
    # Seed torch
    seed_torch()

    n_bits = 2

    # Define an input shape (n_examples, n_features)
    input_shape = (10, input_output_feature)

    # Build a random Quantized Fully Connected Neural Network

    # Define the torch model
    torch_fc_model = model(input_output_feature)
    # Create random input
    numpy_input = numpy.random.uniform(-1, 1, size=input_shape)

    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model)
    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)
    # Quantize input
    q_input = QuantizedArray(n_bits, numpy_input)
    quantized_model(q_input)

    # Compile
    quantized_model.compile(q_input, default_compilation_configuration)
    dequant_predictions = quantized_model.forward_and_dequant(q_input)

    # Compare predictions between FHE and QuantizedModule
    homomorphic_predictions = []
    for x_q in q_input.qvalues:
        homomorphic_predictions.append(
            quantized_model.forward_fhe.run(numpy.array([x_q]).astype(numpy.uint8))
        )
    homomorphic_predictions = quantized_model.dequantize_output(
        numpy.array(homomorphic_predictions, dtype=numpy.float32)
    )

    homomorphic_predictions.reshape(dequant_predictions.shape)

    # Make sure homomorphic_predictions are the same as dequant_predictions
    assert numpy.isclose(homomorphic_predictions.ravel(), dequant_predictions.ravel()).all()
