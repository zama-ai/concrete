"""A torch to numpy module."""
import numpy
from torch import nn


class NumpyModule:
    """General interface to transform a torch.nn.Module to numpy module."""

    IMPLEMENTED_MODULES = {nn.Linear, nn.Sigmoid, nn.ReLU6}

    def __init__(self, torch_model: nn.Module):
        """Initialize our numpy module.

        Current constraint:    All objects used in the forward have to be defined in the
                               __init__() of torch.nn.Module and follow the exact same order.
                               (i.e. each linear layer must have one variable defined in the
                               right order). This constraint will disappear when
                               TorchScript is in place. (issue #818)

        Args:
            torch_model (nn.Module): A fully trained, torch model alond with its parameters.
        """
        self.torch_model = torch_model
        self.check_compatibility()
        self.convert_to_numpy()

    def check_compatibility(self):
        """Check the compatibility of all layers in the torch model."""

        for _, layer in self.torch_model.named_children():
            if (layer_type := type(layer)) not in self.IMPLEMENTED_MODULES:
                raise ValueError(
                    f"The following module is currently not implemented: {layer_type.__name__}. "
                    f"Please stick to the available torch modules: "
                    f"{', '.join(sorted(module.__name__ for module in self.IMPLEMENTED_MODULES))}."
                )
        return True

    def convert_to_numpy(self):
        """Transform all parameters from torch tensor to numpy arrays."""
        self.numpy_module_dict = {}

        for name, weights in self.torch_model.state_dict().items():
            params = weights.detach().numpy()
            self.numpy_module_dict[name] = params.T if "weight" in name else params

    def __call__(self, x: numpy.ndarray):
        """Return the function to be compiled."""
        return self.forward(x)

    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        """Apply a forward pass with numpy function only.

        Args:
            x (numpy.array): Input to be processed in the forward pass.

        Returns:
            x (numpy.array): Processed input.
        """

        for name, layer in self.torch_model.named_children():

            if isinstance(layer, nn.Linear):
                # Apply a matmul product and add the bias.
                x = (
                    x @ self.numpy_module_dict[f"{name}.weight"]
                    + self.numpy_module_dict[f"{name}.bias"]
                )
            elif isinstance(layer, nn.Sigmoid):
                x = 1 / (1 + numpy.exp(-x))
            elif isinstance(layer, nn.ReLU6):
                x = numpy.minimum(numpy.maximum(0, x), 6)
        return x
