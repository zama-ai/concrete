"""
Declaration of `ones` and `one` functions, to simplify creation of encrypted ones.
"""

from typing import Union

import numpy as np

from ..representation import Node
from ..tracing import Tracer
from ..values import ValueDescription


def ones(shape: Union[int, tuple[int, ...]]) -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted array of ones.

    Args:
         shape (Tuple[int, ...]):
            shape of the array

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that represents the operation during tracing
            ndarray filled with ones otherwise
    """

    # pylint: disable=protected-access
    is_tracing = Tracer._is_tracing
    # pylint: enable=protected-access

    numpy_ones = np.ones(shape, dtype=np.int64)

    if is_tracing:
        computation = Node.generic(
            "ones",
            [],
            ValueDescription.of(numpy_ones, is_encrypted=True),
            lambda: np.ones(shape, dtype=np.int64),
        )
        return Tracer(computation, [])

    return numpy_ones


def one() -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted scalar with the value of one.

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that represents the operation during tracing
            ndarray with one otherwise
    """

    return ones(())


def ones_like(array: Union[np.ndarray, Tracer]) -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted array of ones with the same shape as another array.

    Args:
         array (Union[np.ndarray, Tracer]):
            original array

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that represent the operation during tracing
            ndarray filled with ones otherwise
    """

    return ones(array.shape)
