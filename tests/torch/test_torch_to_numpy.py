"""Tests for the torch to numpy module."""
import numpy
import pytest
import torch
from torch import nn

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


@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
        pytest.param(CNN, (100, 3, 32, 32), marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_torch_to_numpy(model, input_shape):
    """Test the different model architecture from torch numpy."""

    torch_fc_model = model()
    torch_input = torch.randn(input_shape)
    torch_predictions = torch_fc_model(torch_input).detach().numpy()
    numpy_fc_model = NumpyModule(torch_fc_model)
    # torch_input to numpy.
    numpy_input = torch_input.detach().numpy()
    numpy_predictions = numpy_fc_model(numpy_input)

    assert numpy_predictions.shape == torch_predictions.shape
    assert numpy.isclose(torch_predictions, numpy_predictions, rtol=10 - 3).all()
