"""
Declaration of `zeros` and `zero` functions, to simplify creation of encrypted zeros.
"""

from typing import Union

import numpy as np

from ..representation import Node
from ..tracing import Tracer
from ..values import ValueDescription


def zeros(shape: Union[int, tuple[int, ...]]) -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted array of zeros.

    Args:
         shape (Tuple[int, ...]):
            shape of the array

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that represents the operation during tracing
            ndarray filled with zeros otherwise
    """

    # pylint: disable=protected-access
    is_tracing = Tracer._is_tracing
    # pylint: enable=protected-access

    numpy_zeros = np.zeros(shape, dtype=np.int64)

    if is_tracing:
        computation = Node.generic(
            "zeros",
            [],
            ValueDescription.of(numpy_zeros, is_encrypted=True),
            lambda: np.zeros(shape, dtype=np.int64),
        )
        return Tracer(computation, [])

    return numpy_zeros


def zero() -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted scalar with the value of zero.

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that respresents the operation during tracing
            ndarray with zero otherwise
    """

    return zeros(())


def zeros_like(array: Union[np.ndarray, Tracer]) -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted array of zeros with the same shape as another array.

    Args:
         array (Union[np.ndarray, Tracer]):
            original array

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that represent the operation during tracing
            ndarray filled with zeros otherwise
    """

    return zeros(array.shape)
