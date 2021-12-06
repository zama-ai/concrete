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
            q_weights (QuantizedArray): Quantized weights (n_features, n_neurons).
            q_bias (QuantizedArray, optional): Quantized bias (1, n_neurons). Defaults to None.
        """
        self.q_weights = q_weights
        self.q_bias = q_bias
        self.n_bits = n_bits

        if self.q_bias is None:
            self.q_bias = QuantizedArray(n_bits, numpy.zeros(self.q_weights.values.shape[-1]))
        self.q_out = None

    def calibrate(self, x: numpy.ndarray):
        """Create corresponding QuantizedArray for the output of QuantizedLinear.

        Args:
            x (numpy.ndarray): Inputs.
        """
        assert self.q_bias is not None
        self.q_out = QuantizedArray(self.n_bits, (x @ self.q_weights.values) + self.q_bias.values)

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

        # The following MatMul is done with integers, and thus, does not use of any PBS.
        # Only the final conversion to float is done with a PBS, which can actually
        # be merged with the PBS of following activation.
        # State of the art quantization method assumes the following results in a int32 accumulator.

        # Here we follow Eq.7 in https://arxiv.org/abs/1712.05877 to split the core computation
        # from the zero points and scales.

        p = self.q_weights.qvalues.shape[0]

        # Core matmul operation in full intergers with a shape change (INTEGERS)
        matmul = q_input.qvalues @ self.q_weights.qvalues

        # Sum operation in full integers resulting in large integers (INTEGERS)
        # [WORKAROUND #995] numpy.sum can't be currently done in our framework
        # sum_input = self.q_weights.zero_point * numpy.sum(q_input.qvalues, axis=1, keepdims=True)
        # Hack because we can't do numpy.sum(axis...,keepdims...)
        const_ones = numpy.ones(shape=(q_input.n_features, 1), dtype=int)
        sum_input = self.q_weights.zero_point * (q_input.qvalues @ const_ones)

        # Last part that has to be done in FHE the rest must go in a PBS.
        # Forced fusing using .astype(numpy.float32)
        numpy_q_out = (matmul + (numpy.negative(sum_input))).astype(numpy.float32)

        # sum_weights is a constant
        sum_weights = q_input.zero_point * numpy.sum(self.q_weights.qvalues, axis=0, keepdims=True)

        # Quantization scales and zero points (FLOATS involved)
        # This is going to be compiled with a PBS (along with the following activation function)
        m_matmul = (q_input.scale * self.q_weights.scale) / (self.q_out.scale)
        bias_part = (
            self.q_bias.scale / self.q_out.scale * (self.q_bias.qvalues - self.q_bias.zero_point)
        )
        final_term = p * q_input.zero_point * self.q_weights.zero_point

        numpy_q_out = numpy_q_out + final_term + (numpy.negative(sum_weights))
        numpy_q_out = m_matmul * numpy_q_out
        numpy_q_out = self.q_out.zero_point + bias_part + numpy_q_out

        numpy_q_out = numpy.rint(numpy_q_out).clip(0, 2 ** self.q_out.n_bits - 1).astype(int)

        # TODO find a more intuitive way to do the following (see issue #832)
        # We should be able to reuse q_out quantization parameters
        # easily to get a new QuantizedArray
        q_out_ = copy.copy(self.q_out)
        q_out_.update_qvalues(numpy_q_out)

        return q_out_
