"""Modules for quantization."""
from .post_training import PostTrainingAffineQuantization
from .quantized_activations import QuantizedReLU6, QuantizedSigmoid
from .quantized_array import QuantizedArray
from .quantized_layers import QuantizedLinear
from .quantized_module import QuantizedModule
