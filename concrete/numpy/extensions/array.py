"""
Declaration of `array` function, to simplify creation of encrypted arrays.
"""

from typing import Any, Union

import numpy as np

from ..dtypes.utils import combine_dtypes
from ..representation import Node
from ..tracing import Tracer
from ..values import Value


def array(values: Any) -> Union[np.ndarray, Tracer]:
    """
    Create an encrypted array from either encrypted or clear values.

    Args:
        values (Any):
            array like object compatible with numpy to construct the resulting encrypted array

    Returns:
        Union[np.ndarray, Tracer]:
            Tracer that respresents the operation during tracing
            ndarray with values otherwise
    """

    # pylint: disable=protected-access
    is_tracing = Tracer._is_tracing
    # pylint: enable=protected-access

    if not isinstance(values, np.ndarray):
        values = np.array(values)

    if not is_tracing:
        return values

    shape = values.shape
    values = values.flatten()

    for i, value in enumerate(values):
        if not isinstance(value, Tracer):
            values[i] = Tracer.sanitize(value)

        if not values[i].output.is_scalar:
            message = "Encrypted arrays can only be created from scalars"
            raise ValueError(message)

    dtype = combine_dtypes([value.output.dtype for value in values])
    is_encrypted = True

    computation = Node.generic(
        "array",
        [value.output for value in values],
        Value(dtype, shape, is_encrypted),
        lambda *args: np.array(args).reshape(shape),
    )
    return Tracer(computation, values)
