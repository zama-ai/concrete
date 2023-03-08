"""
Declaration of various functions and constants related to representation of computation.
"""

from typing import Any, Dict, Hashable, Set, Union

import numpy as np

from ..internal.utils import assert_that

KWARGS_IGNORED_IN_FORMATTING: Set[str] = {
    "subgraph",
    "terminal_node",
}

SPECIAL_OBJECT_MAPPING: Dict[Any, str] = {
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.byte: "byte",
    np.short: "short",
    np.intc: "intc",
    np.int_: "int_",
    np.longlong: "longlong",
    np.ubyte: "ubyte",
    np.ushort: "ushort",
    np.uintc: "uintc",
    np.uint: "uint",
    np.ulonglong: "ulonglong",
}


def format_constant(constant: Any, maximum_length: int = 45, keep_newlines: bool = False) -> str:
    """
    Get the textual representation of a constant.

    Args:
        constant (Any):
            constant to format

        maximum_length (int, default = 45):
            maximum length of the resulting string

        keep_newlines (bool, default = False):
            whether to keep newlines or not

    Returns:
        str:
            textual representation of `constant`
    """

    if isinstance(constant, Hashable) and constant in SPECIAL_OBJECT_MAPPING:
        return SPECIAL_OBJECT_MAPPING[constant]

    # maximum_length should not be smaller than 7 characters because
    # the constant will be formatted to `x ... y`
    # where x and y are part of the constant, and they are at least 1 character
    assert_that(maximum_length >= 7)

    result = str(constant)
    if not keep_newlines:
        result = result.replace("\n", "")

    if len(result) > maximum_length:
        from_start = (maximum_length - 5) // 2
        from_end = (maximum_length - 5) - from_start

        if keep_newlines and "\n" in result:
            result = f"{result[:from_start]}\n...\n{result[-from_end:]}"
        else:
            result = f"{result[:from_start]} ... {result[-from_end:]}"

    return result


def format_indexing_element(indexing_element: Union[int, np.integer, slice]):
    """
    Format an indexing element.

    This is required mainly for slices. The reason is that string representation of slices
    are very long and verbose. To give an example, `x[:, 2:]` will have the following index
    `[slice(None, None, None), slice(2, None, None)]` if printed naively. With this helper,
    it will be formatted as `[:, 2:]`.

    Args:
        indexing_element (Union[int, np.integer, slice]):
            indexing element to format

    Returns:
        str:
            textual representation of `indexing_element`
    """

    result = ""
    if isinstance(indexing_element, slice):
        if indexing_element.start is not None:
            result += str(indexing_element.start)
        result += ":"
        if indexing_element.stop is not None:
            result += str(indexing_element.stop)
        if indexing_element.step is not None:
            result += ":"
            result += str(indexing_element.step)
    else:
        result += str(indexing_element)
    return result.replace("\n", " ")
