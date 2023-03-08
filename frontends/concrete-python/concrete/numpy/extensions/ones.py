"""
Declaration of `ones` and `one` functions, to simplify creation of encrypted ones.
"""

from typing import Tuple, Union

import numpy as np

from ..representation import Node
from ..tracing import Tracer
from ..values import Value


def ones(shape: Union[int, Tuple[int, ...]]) -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted array of ones.

    Args:
         shape (Tuple[int, ...]):
            shape of the array

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that respresents the operation during tracing
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
            Value.of(numpy_ones, is_encrypted=True),
            lambda: np.ones(shape, dtype=np.int64),
        )
        return Tracer(computation, [])

    return numpy_ones


def one() -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted scalar with the value of one.

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that respresents the operation during tracing
            ndarray with one otherwise
    """

    return ones(())
