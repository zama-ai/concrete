"""
Declaration of `TFHERSIntegerType` class.
"""

from functools import partial
from typing import Any, Union

import numpy as np

from ..dtypes import Integer


class TFHERSParams:
    """Crypto parameters used for a tfhers integer."""

    pass


class TFHERSIntegerType(Integer):
    """
    TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types.
    """

    carry_width: int
    msg_width: int
    params: TFHERSParams

    def __init__(
        self,
        is_signed: bool,
        bit_width: int,
        carry_width: int,
        msg_width: int,
        params: TFHERSParams,
    ):
        super().__init__(is_signed, bit_width)
        self.carry_width = carry_width
        self.msg_width = msg_width
        self.params = params

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and super().__eq__(other)
            and self.carry_width == other.carry_width
            and self.msg_width == other.msg_width
            and self.params == other.params
        )

    def __str__(self) -> str:
        return (
            f"tfhers<{('int' if self.is_signed else 'uint')}"
            f"{self.bit_width}, {self.carry_width}, {self.msg_width}, {self.params}>"
        )

    def encode(self, value: Union[int, np.integer, np.ndarray]) -> np.ndarray:
        """Encode a scalar or tensor to tfhers integers.

        Args:
            value (Union[int, np.ndarray]): scalar or tensor of integer to encode

        Raises:
            TypeError: wrong value type

        Returns:
            np.ndarray: encoded scalar or tensor
        """
        bit_width = self.bit_width
        msg_width = self.msg_width
        if isinstance(value, (int, np.integer)):
            value_bin = bin(value)[2:].zfill(bit_width)
            # msb first
            return np.array(
                [int(value_bin[i : i + msg_width], 2) for i in range(0, bit_width, msg_width)]
            )
        if isinstance(value, np.ndarray):
            return np.array([self.encode(int(v)) for v in value.flatten()]).reshape(
                value.shape + (bit_width // msg_width,)
            )
        msg = f"can only encode int or ndarray, but got {type(value)}"
        raise TypeError(msg)

    def decode(self, value: np.ndarray) -> Union[int, np.ndarray]:
        """Decode a tfhers-encoded integer (scalar or tensor).

        Args:
            value (np.ndarray): encoded value

        Raises:
            ValueError: bad encoding

        Returns:
            Union[int, np.ndarray]: decoded value
        """
        bit_width = self.bit_width
        msg_width = self.msg_width
        expected_ct_shape = bit_width // msg_width
        if value.shape[-1] != expected_ct_shape:
            msg = (
                f"bad encoding: expected value with last shape being {expected_ct_shape} "
                f"but got {value.shape[-1]}"
            )
            raise ValueError(msg)
        if len(value.shape) == 1:
            # reversed because it's msb first and we are computing powers lsb first
            return sum(v << i * msg_width for i, v in enumerate(reversed(value)))
        cts = value.reshape((-1, expected_ct_shape))
        return np.array([self.decode(ct) for ct in cts]).reshape(value.shape[:-1])


int8 = partial(TFHERSIntegerType, True, 8)
uint8 = partial(TFHERSIntegerType, False, 8)
int16 = partial(TFHERSIntegerType, True, 16)
uint16 = partial(TFHERSIntegerType, False, 16)

# TODO: make these partials as well, so that params have to be specified
int8_2_2 = int8(2, 2, TFHERSParams())
uint8_2_2 = uint8(2, 2, TFHERSParams())
int16_2_2 = int16(2, 2, TFHERSParams())
uint16_2_2 = uint16(2, 2, TFHERSParams())
