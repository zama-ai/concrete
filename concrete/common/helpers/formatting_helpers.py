"""Helpers for formatting functionality."""

from typing import Any, Dict, Hashable

import numpy

from ..debugging.custom_assert import assert_true

SPECIAL_OBJECT_MAPPING: Dict[Any, str] = {
    numpy.float32: "float32",
    numpy.float64: "float64",
    numpy.int8: "int8",
    numpy.int16: "int16",
    numpy.int32: "int32",
    numpy.int64: "int64",
    numpy.uint8: "uint8",
    numpy.uint16: "uint16",
    numpy.uint32: "uint32",
    numpy.uint64: "uint64",
}


def format_constant(constant: Any, maximum_length: int = 45) -> str:
    """Format a constant.

    Args:
        constant (Any): the constant to format
        maximum_length (int): maximum length of the resulting string

    Returns:
        str: the formatted constant
    """

    if isinstance(constant, Hashable) and constant in SPECIAL_OBJECT_MAPPING:
        return SPECIAL_OBJECT_MAPPING[constant]

    # maximum_length should not be smaller than 7 characters because
    # the constant will be formatted to `x ... y`
    # where x and y are part of the constant and they are at least 1 character
    assert_true(maximum_length >= 7)

    content = str(constant).replace("\n", "")
    if len(content) > maximum_length:
        from_start = (maximum_length - 5) // 2
        from_end = (maximum_length - 5) - from_start
        content = f"{content[:from_start]} ... {content[-from_end:]}"
    return content
