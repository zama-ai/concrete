"""
Declaration of `Integer` class.
"""

import math
from functools import partial
from typing import Any

import numpy as np

from .base import BaseDataType


class Integer(BaseDataType):
    """
    Integer class, to represent integers.
    """

    is_signed: bool
    bit_width: int

    @staticmethod
    def that_can_represent(value: Any, force_signed: bool = False) -> "Integer":
        """
        Get the minimal `Integer` that can represent `value`.

        Args:
            value (Any):
                value that needs to be represented

            force_signed (bool, default = False):
                whether to force signed integers or not

        Returns:
            Integer:
                minimal `Integer` that can represent `value`

        Raises:
            ValueError:
                if `value` cannot be represented by `Integer`
        """

        lower_bound: int
        upper_bound: int

        if isinstance(value, list):
            try:
                value = np.array(value)
            except Exception:  # pylint: disable=broad-except
                # here we try our best to convert the list to np.ndarray
                # if it fails we raise the exception at the else branch below
                pass

        if isinstance(value, (int, np.integer)):
            lower_bound = int(value)
            upper_bound = int(value)
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
            lower_bound = int(value.min())
            upper_bound = int(value.max())
        else:
            message = f"Integer cannot represent {repr(value)}"
            raise ValueError(message)

        def bits_to_represent_int(value: int, force_signed: bool) -> int:
            bits: int

            if value == 0:
                return 1

            if value < 0:
                bits = int(math.ceil(math.log2(abs(value)))) + 1
            else:
                bits = int(math.ceil(math.log2(value + 1)))
                if force_signed:
                    bits += 1

            return bits

        is_signed = force_signed or lower_bound < 0
        bit_width = (
            bits_to_represent_int(lower_bound, is_signed)
            if lower_bound == upper_bound
            else max(
                bits_to_represent_int(lower_bound, is_signed),
                bits_to_represent_int(upper_bound, is_signed),
            )
        )

        return Integer(is_signed, bit_width)

    def __init__(self, is_signed: bool, bit_width: int):
        super().__init__()

        if not isinstance(bit_width, int) or bit_width <= 0:
            integer_str = "SignedInteger" if is_signed else "UnsignedInteger"
            message = (
                f"{integer_str}({repr(bit_width)}) is not supported "
                f"(bit width must be a positive integer)"
            )
            raise ValueError(message)

        self.is_signed = is_signed
        self.bit_width = bit_width

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.is_signed == other.is_signed
            and self.bit_width == other.bit_width
        )

    def __str__(self) -> str:
        return f"{('int' if self.is_signed else 'uint')}{self.bit_width}"

    def min(self) -> int:
        """
        Get the minumum value that can be represented by the `Integer`.

        Returns:
            int:
                minumum value that can be represented by the `Integer`
        """

        return 0 if not self.is_signed else -(2 ** (self.bit_width - 1))

    def max(self) -> int:
        """
        Get the maximum value that can be represented by the `Integer`.

        Returns:
            int:
                maximum value that can be represented by the `Integer`
        """

        return (2**self.bit_width) - 1 if not self.is_signed else (2 ** (self.bit_width - 1)) - 1

    def can_represent(self, value: int) -> bool:
        """
        Get whether `value` can be represented by the `Integer` or not.

        Args:
            value (int):
                value to check representability

        Returns:
            bool:
                True if `value` is representable by the `integer`, False otherwise
        """

        return self.min() <= value <= self.max()


SignedInteger = partial(Integer, True)

UnsignedInteger = partial(Integer, False)
