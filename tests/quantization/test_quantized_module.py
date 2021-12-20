"""Tests for the quantized module."""
import numpy
import pytest
import torch
from torch import nn

from concrete.quantization import PostTrainingAffineQuantization, QuantizedArray
from concrete.torch import NumpyModule


class CNN(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=32 * 32 * 3, out_features=128)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.sigmoid3 = nn.Sigmoid()
        self.fc4 = nn.Linear(in_features=64, out_features=64)
        self.sigmoid4 = nn.Sigmoid()
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        out = self.fc3(out)
        out = self.sigmoid3(out)
        out = self.fc4(out)
        out = self.sigmoid4(out)
        out = self.fc5(out)

        return out


N_BITS_ATOL_TUPLE_LIST = [
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
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
    ],
)
@pytest.mark.parametrize(
    "is_signed",
    [pytest.param([False, True])],
)
def test_quantized_linear(model, input_shape, n_bits, atol, is_signed, seed_torch):
    """Test the quantized module with a post-training static quantization.

    With n_bits>>0 we expect the results of the quantized module
    to be the same as the standard module.
    """
    # Seed torch
    seed_torch()
    # Define the torch model
    torch_fc_model = model()
    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model)
    # Predict with real model
    numpy_prediction = numpy_fc_model(numpy_input)
    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(
        n_bits, numpy_fc_model, is_signed=is_signed
    )
    quantized_model = post_training_quant.quantize_module(numpy_input)
    # Quantize input
    q_input = QuantizedArray(n_bits, numpy_input)
    # Forward and Dequantize to get back to real values
    dequant_prediction = quantized_model.forward_and_dequant(q_input)

    assert numpy.isclose(numpy_prediction, dequant_prediction, atol=atol).all()
