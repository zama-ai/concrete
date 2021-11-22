"""This file holds the definitions for floating point types."""

from functools import partial

from ..debugging.custom_assert import assert_true
from . import base


class Float(base.BaseDataType):
    """Class representing a float."""

    # bit_width is the total number of bits used to represent a floating point number, including
    # sign bit, exponent and mantissa
    bit_width: int

    def __init__(self, bit_width: int) -> None:
        super().__init__()
        assert_true(bit_width in (16, 32, 64), "Only 16, 32 and 64 bits floats are supported")
        self.bit_width = bit_width

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.bit_width} bits>"

    def __str__(self) -> str:
        return f"float{self.bit_width}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.bit_width == other.bit_width


Float16 = partial(Float, 16)
Float32 = partial(Float, 32)
Float64 = partial(Float, 64)
