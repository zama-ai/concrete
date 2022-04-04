"""
Declaration of various functions and constants related to data types.
"""

from typing import List

from ..internal.utils import assert_that
from .base import BaseDataType
from .float import Float
from .integer import Integer, SignedInteger, UnsignedInteger


def combine_dtypes(dtypes: List[BaseDataType]) -> BaseDataType:
    """
    Get the 'BaseDataType' that can represent a set of 'BaseDataType's.

    Args:
        dtypes (List[BaseDataType]):
            dtypes to combine

    Returns:
        BaseDataType:
            dtype that can hold all the given dtypes (potentially lossy)
    """

    assert_that(len(dtypes) != 0)
    assert_that(all(isinstance(dtype, (Integer, Float)) for dtype in dtypes))

    def combine_2_dtypes(dtype1: BaseDataType, dtype2: BaseDataType) -> BaseDataType:
        result: BaseDataType = dtype1

        if isinstance(dtype1, Integer) and isinstance(dtype2, Integer):
            max_bits = max(dtype1.bit_width, dtype2.bit_width)

            if dtype1.is_signed and dtype2.is_signed:
                result = SignedInteger(max_bits)

            elif not dtype1.is_signed and not dtype2.is_signed:
                result = UnsignedInteger(max_bits)

            elif dtype1.is_signed and not dtype2.is_signed:
                # if dtype2 has the bigger bit_width,
                # we need a signed integer that can hold
                # it, so add 1 bit of sign to its bit_width
                if dtype2.bit_width >= dtype1.bit_width:
                    new_bit_width = dtype2.bit_width + 1
                    result = SignedInteger(new_bit_width)
                else:
                    result = SignedInteger(dtype1.bit_width)

            elif not dtype1.is_signed and dtype2.is_signed:
                # Same as above, with dtype1 and dtype2 switched around
                if dtype1.bit_width >= dtype2.bit_width:
                    new_bit_width = dtype1.bit_width + 1
                    result = SignedInteger(new_bit_width)
                else:
                    result = SignedInteger(dtype2.bit_width)

        elif isinstance(dtype1, Float) and isinstance(dtype2, Float):
            max_bits = max(dtype1.bit_width, dtype2.bit_width)
            result = Float(max_bits)

        elif isinstance(dtype1, Float):
            result = dtype1

        elif isinstance(dtype2, Float):
            result = dtype2

        return result

    result = dtypes[0]
    for other in dtypes[1:]:
        result = combine_2_dtypes(result, other)
    return result
