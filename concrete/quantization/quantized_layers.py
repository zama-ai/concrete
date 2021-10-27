"""Quantized layers."""
import copy
from typing import Optional

import numpy

from .quantized_array import QuantizedArray


class QuantizedLinear:
    """Fully connected quantized layer."""

    q_out: Optional[QuantizedArray]

    def __init__(
        self, n_bits: int, q_weights: QuantizedArray, q_bias: Optional[QuantizedArray] = None
    ):
        """Implement the forward pass of a quantized linear layer.

        Note: QuantizedLinear seems to become unstable when n_bits > 23.

        Args:
            n_bits (int): Maximum number of bits for the ouput.
            q_weights (QuantizedArray): Quantized weights (n_examples, n_neurons, n_features).
            q_bias (QuantizedArray, optional): Quantized bias (n_neurons). Defaults to None.
        """
        self.q_weights = q_weights
        self.q_bias = q_bias
        self.n_bits = n_bits

        if self.q_bias is None:
            self.q_bias = QuantizedArray(n_bits, numpy.zeros(self.q_weights.values.shape[:-1]))
        self.q_out = None

    def calibrate(self, x: numpy.ndarray):
        """Create corresponding QuantizedArray for the output of QuantizedLinear.

        Args:
            x (numpy.ndarray): Inputs.
        """
        assert self.q_bias is not None
        self.q_out = QuantizedArray(self.n_bits, x @ self.q_weights.values.T + self.q_bias.values)

    def __call__(self, q_input: QuantizedArray) -> QuantizedArray:
        """Process the forward pass of the quantized linear layer.

        Note: in standard quantization, floats are problematics as quantization
        targets a specific integer only hardware. However in FHE, we can create a table lookup
        to bypass this problem. Thus we leave the floats as is.
        Args:
            q_input (QuantizedArray): Quantized input.

        Returns:
            q_out_ (QuantizedArray): Quantized output.
        """
        # Satisfy mypy.
        assert self.q_out is not None
        assert self.q_bias is not None
        # We need to develop the following equation to have the main computation
        # (self.q_weights.q_values @ self.q_inputs.q_values) without zero_point values.
        # See https://github.com/google/gemmlowp/blob/master/doc/quantization.md #852

        m_product = (q_input.scale * self.q_weights.scale) / (self.q_out.scale)
        dot_product = (q_input.qvalues - q_input.zero_point) @ (
            self.q_weights.qvalues - self.q_weights.zero_point
        ).T

        m_bias = self.q_bias.scale / (q_input.scale * self.q_weights.scale)
        bias_part = m_bias * (self.q_bias.qvalues - self.q_bias.zero_point)
        numpy_q_out = m_product * (dot_product + bias_part) + self.q_out.zero_point

        numpy_q_out = numpy_q_out.round().clip(0, 2 ** self.q_out.n_bits - 1).astype(int)

        # TODO find a more intuitive way to do the following (see issue #832)
        # We should be able to reuse q_out quantization parameters
        # easily to get a new QuantizedArray
        q_out_ = copy.copy(self.q_out)
        q_out_.update_qvalues(numpy_q_out)

        return q_out_
