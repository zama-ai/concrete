"""Post Training Quantization methods."""

import numpy
from torch import nn

from ..torch import NumpyModule
from .quantized_activations import QuantizedReLU6, QuantizedSigmoid
from .quantized_array import QuantizedArray
from .quantized_layers import QuantizedLinear
from .quantized_module import QuantizedModule


class PostTrainingAffineQuantization:
    """Post-training Affine Quantization."""

    IMPLEMENTED_MODULES = {nn.Linear, nn.Sigmoid, nn.ReLU6}

    quant_layers_dict: dict
    n_bits: int
    quant_params: dict
    numpy_model: NumpyModule
    is_signed: bool

    def __init__(self, n_bits: int, numpy_model: NumpyModule, is_signed: bool = False):
        """Create the quantized version of numpy module.

        Args:
            n_bits (int):                   Number of bits to quantize the model. Currently this
                                            n_bits will be used for all activation/inputs/weights
            numpy_model (NumpyModule):      Model in numpy.
            is_signed:                      Whether the weights of the layers can be signed.
                                            Currently, only the weights can be signed.

        Returns:
            QuantizedModule: A quantized version of the numpy model.
        """
        self.quant_layers_dict = {}
        self.n_bits = n_bits
        self.quant_params = {}
        self.numpy_model = numpy_model
        self.is_signed = is_signed

    def quantize_module(self, calibration_data: numpy.ndarray) -> QuantizedModule:
        """Quantize numpy module.

        Following https://arxiv.org/abs/1712.05877 guidelines.

        Args:
            calibration_data (numpy.ndarray):   Data that will be used to compute the bounds,
                                                scales and zero point values for every quantized
                                                object.

        Returns:
            QuantizedModule: Quantized numpy module
        """
        # First transform all parameters to their quantized version
        self._quantize_params()
        # Quantize and calibrate each output layer/activation
        self._quantize_layers(calibration_data=calibration_data)
        # Create quantized module from self.quant_layers_dict
        return QuantizedModule(self.quant_layers_dict)

    def _quantize_params(self):
        """Transform all floating points parameters to integers."""

        for name, params in self.numpy_model.numpy_module_dict.items():
            self.quant_params[name] = QuantizedArray(self.n_bits, params, self.is_signed)

    def _calibrate_layers_activation(self, name, q_function, calibration_data):
        # Calibrate the output of the layer
        q_function.calibrate(calibration_data)
        # Store the learned quantized layer
        self.quant_layers_dict[name] = q_function
        # Create new calibration data (output of the previous layer)
        q_calibration_data = QuantizedArray(self.n_bits, calibration_data)
        # Dequantize to have the value in clear and ready for next calibration
        return q_function(q_calibration_data).dequant()

    def _quantize_layers(self, calibration_data: numpy.ndarray):
        """Compute all parameters for the static post-training quantization.

        Does a forward pass over a batch of data and compute all
        quantization parameters for activations and layers.
        """
        for name, layer in self.numpy_model.torch_model.named_children():

            if isinstance(layer, nn.Linear):
                # Create a QuantizedLinear layer
                q_weights = self.quant_params[f"{name}.weight"]
                q_bias = self.quant_params[f"{name}.bias"]
                # Check if layer is last layer from the model
                if name == list(self.numpy_model.torch_model.named_children())[-1][0]:
                    # If last layer, we can use 7 bits (maximum allowed) of precision.
                    # However, 6 bits is currently used to allow 100% FHE precision
                    # compared to its quantized counterpart.
                    # Since this is the last layer and mostly used for classification,
                    # this does not have much impact.
                    # TODO: Put back 7 bits when 100% at 7b is achieved (see issue #1332).
                    q_layer = QuantizedLinear(numpy.maximum(6, self.n_bits), q_weights, q_bias)
                else:
                    q_layer = QuantizedLinear(self.n_bits, q_weights, q_bias)
                # Calibrate and get new calibration_data for next layer/activation
                calibration_data = self._calibrate_layers_activation(
                    name, q_layer, calibration_data
                )
            elif isinstance(layer, nn.Sigmoid):
                # Create a new quantized layer (based on type(layer))
                q_sigmoid = QuantizedSigmoid(n_bits=self.n_bits)
                calibration_data = self._calibrate_layers_activation(
                    name, q_sigmoid, calibration_data
                )
            elif isinstance(layer, nn.ReLU6):
                # Create a new quantized layer (based on type(layer))
                q_relu = QuantizedReLU6(n_bits=self.n_bits)
                calibration_data = self._calibrate_layers_activation(name, q_relu, calibration_data)
            else:  # pragma: no cover
                # If we find a layer that has not been implemented we throw an error
                hf_m_names = sorted(module.__name__ for module in self.IMPLEMENTED_MODULES)
                raise ValueError(
                    f"The following module is currently not implemented: {type(layer).__name__}"
                    f"Please stick to the available quantized modules:"
                    f"{', '.join(hf_m_names)}."
                )
