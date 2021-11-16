"""QuantizedModule API."""
import copy

from .quantized_array import QuantizedArray


class QuantizedModule:
    """Inference for a quantized model."""

    def __init__(self, quant_layers_dict: dict):
        self.quant_layers_dict = copy.deepcopy(quant_layers_dict)

    def __call__(self, x: QuantizedArray) -> QuantizedArray:
        return self.forward(x)

    def forward(self, q_x: QuantizedArray) -> QuantizedArray:
        """Forward pass with numpy function only.

        Args:
            q_x (QuantizedArray): QuantizedArray containing the inputs.

        Returns:
            (QuantizedArray): Prediction of the quantized model
        """
        for _, layer in self.quant_layers_dict.items():
            q_x = layer(q_x)

        return q_x
