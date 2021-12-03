"""Tests for the torch to numpy module."""
import numpy
import pytest
from torch import nn

from concrete.quantization import QuantizedArray
from concrete.torch.compile import compile_torch_model

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
    [pytest.param(FC, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
def test_compile_torch(input_output_feature, model, seed_torch, default_compilation_configuration):
    """Test the different model architecture from torch numpy."""

    # Seed torch
    seed_torch()

    n_bits = 2

    # Define an input shape (n_examples, n_features)
    n_examples = 10

    # Define the torch model
    torch_fc_model = model(input_output_feature)
    # Create random input
    inputset = [numpy.random.uniform(-1, 1, size=input_output_feature) for _ in range(n_examples)]

    # Compile
    quantized_numpy_module = compile_torch_model(
        torch_fc_model,
        inputset,
        default_compilation_configuration,
        n_bits=n_bits,
    )

    # Compare predictions between FHE and QuantizedModule
    clear_predictions = []
    homomorphic_predictions = []
    for numpy_input in inputset:
        q_input = QuantizedArray(n_bits, numpy_input)
        x_q = q_input.qvalues
        clear_predictions.append(quantized_numpy_module.forward(x_q))
        homomorphic_predictions.append(
            quantized_numpy_module.forward_fhe.run(numpy.array([x_q]).astype(numpy.uint8))
        )

    clear_predictions = numpy.array(clear_predictions)
    homomorphic_predictions = numpy.array(homomorphic_predictions)

    # Make sure homomorphic_predictions are the same as dequant_predictions
    assert numpy.array_equal(homomorphic_predictions, clear_predictions)
