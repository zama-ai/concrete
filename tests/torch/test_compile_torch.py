"""Tests for the torch to numpy module."""
import pytest
import torch
from torch import nn

from concrete.torch.compile import compile_torch_model


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
    ],
)
def test_compile_torch(model, input_shape, default_compilation_configuration, seed_torch):
    """Test the different model architecture from torch numpy."""

    # Seed torch
    seed_torch()

    # Define the torch model
    torch_fc_model = model()

    # Create random input
    torch_inputset = torch.randn(input_shape)

    # Compile
    compile_torch_model(
        torch_fc_model,
        torch_inputset,
        default_compilation_configuration,
    )
