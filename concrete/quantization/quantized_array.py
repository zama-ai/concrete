"""Quantization utilities for a numpy array/tensor."""
from copy import deepcopy
from typing import Optional

import numpy

STABILITY_CONST = 10 ** -12


class QuantizedArray:
    """Abstraction of quantized array."""

    def __init__(self, n_bits: int, values: numpy.ndarray, is_signed=False):
        """Quantize an array.

        See https://arxiv.org/abs/1712.05877.

        Args:
            values (numpy.ndarray): Values to be quantized.
            n_bits (int): The number of bits to use for quantization.
            is_signed (bool): Whether the quantization can be on signed integers.
        """

        self.offset = 0
        if is_signed:
            self.offset = 2 ** (n_bits - 1)
        self.values = values
        self.n_bits = n_bits
        self.is_signed = is_signed
        self.scale, self.zero_point, self.qvalues = self.compute_quantization_parameters()

    def __call__(self) -> Optional[numpy.ndarray]:
        return self.qvalues

    def compute_quantization_parameters(self):
        """Compute the quantization parameters."""
        # Small constant needed for stability
        rmax = numpy.max(self.values) + STABILITY_CONST
        rmin = numpy.min(self.values)
        scale = (
            (rmax - rmin) / ((2 ** self.n_bits - 1 - self.offset) - (-self.offset))
            if rmax != rmin
            else 1.0
        )

        zero_point = numpy.round(
            (rmax * (-self.offset) - (rmin * (2 ** self.n_bits - 1 - self.offset))) / (rmax - rmin)
        )

        # Compute quantized values and store
        qvalues = self.values / scale + zero_point

        qvalues = (
            qvalues.round()
            .clip(-self.offset, 2 ** (self.n_bits) - 1 - self.offset)
            .astype(int)  # Careful this can be very large with high number of bits
        )

        return scale, zero_point, qvalues

    def update_values(self, values: numpy.ndarray) -> Optional[numpy.ndarray]:
        """Update values to get their corresponding qvalues using the related quantized parameters.

        Args:
            values (numpy.ndarray): Values to replace self.values

        Returns:
            qvalues (numpy.ndarray): Corresponding qvalues
        """
        self.values = deepcopy(values)
        self.quant()
        return self.qvalues

    def update_qvalues(self, qvalues: numpy.ndarray) -> Optional[numpy.ndarray]:
        """Update qvalues to get their corresponding values using the related quantized parameters.

        Args:
            qvalues (numpy.ndarray): Values to replace self.qvalues

        Returns:
            values (numpy.ndarray): Corresponding values
        """
        self.qvalues = deepcopy(qvalues)
        self.dequant()
        return self.values

    def quant(self) -> Optional[numpy.ndarray]:
        """Quantize self.values.

        Returns:
            numpy.ndarray: Quantized values.
        """

        self.qvalues = (
            (self.values / self.scale + self.zero_point)
            .round()
            .clip(-self.offset, 2 ** (self.n_bits) - 1 - self.offset)
            .astype(int)
        )
        return self.qvalues

    def dequant(self) -> numpy.ndarray:
        """Dequantize self.qvalues.

        Returns:
            numpy.ndarray: Dequantized values.
        """
        self.values = self.scale * (self.qvalues - self.zero_point)
        return self.values
