"""
Tracing of tfhers operations.
"""

from copy import deepcopy
from typing import Union

import numpy as np

from ..dtypes import Integer
from ..extensions.hint import hint
from ..representation import Node
from ..tracing import Tracer
from ..values import EncryptedTensor
from .dtypes import TFHERSIntegerType
from .values import TFHERSInteger


def to_native(value: Union[Tracer, TFHERSInteger]) -> Union[Tracer, int, np.ndarray]:
    """Convert a tfhers integer to the Concrete representation.

    Args:
        value (Union[Tracer, TFHERSInteger]): tfhers integer

    Raises:
        TypeError: wrong input type

    Returns:
        Union[Tracer, int, ndarray]: Tracer if the input is a tracer. int or ndarray otherwise.
    """

    if isinstance(value, Tracer) and isinstance(value.output.dtype, TFHERSIntegerType):
        dtype = value.output.dtype
        if not isinstance(dtype, TFHERSIntegerType):  # pragma: no cover
            msg = f"tracer didn't contain an output of TFHEInteger type. Type is: {dtype}"
            raise TypeError(msg)
        return _trace_to_native(value, dtype)

    if isinstance(value, TFHERSInteger):
        return _eval_to_native(value)

    msg = "tfhers.to_native should be called with a TFHERSInteger"
    raise ValueError(msg)


def from_native(
    value: Union[Tracer, int, np.ndarray], dtype_to: TFHERSIntegerType
) -> Union[Tracer, int, np.ndarray]:
    """Convert a Concrete integer to the tfhers representation.

    The returned value isn't wrapped in a TFHERSInteger, but should have its representation.

    Args:
        value (Union[Tracer, int, ndarray]): Concrete value to convert to tfhers
        dtype_to (TFHERSIntegerType): tfhers integer type to convert to

    Returns:
        Union[Tracer, int, ndarray]
    """
    if isinstance(value, Tracer):
        return _trace_from_native(value, dtype_to)
    return _eval_from_native(value)


def _trace_to_native(tfhers_int: Tracer, dtype: TFHERSIntegerType):
    output = EncryptedTensor(
        Integer(dtype.is_signed, dtype.bit_width),
        tfhers_int.output.shape,
    )

    computation = Node.generic(
        "tfhers_to_native",
        deepcopy(
            [
                tfhers_int.output,
            ]
        ),
        output,
        _eval_to_native,
        args=(),
        attributes={"type": dtype},
    )
    tracer = Tracer(
        computation,
        input_tracers=[
            tfhers_int,
        ],
    )
    hint(tracer, bit_width=dtype.bit_width)
    return tracer


def _trace_from_native(native_int: Tracer, dtype_to: TFHERSIntegerType):
    output = EncryptedTensor(dtype_to, native_int.output.shape)

    computation = Node.generic(
        "tfhers_from_native",
        deepcopy(
            [
                native_int.output,
            ]
        ),
        output,
        _eval_from_native,
        args=(),
        attributes={"type": dtype_to},
    )
    tracer = Tracer(
        computation,
        input_tracers=[
            native_int,
        ],
    )
    hint(tracer, bit_width=dtype_to.bit_width)
    return tracer


def _eval_to_native(tfhers_int: TFHERSInteger):
    return tfhers_int.value


def _eval_from_native(native_value):
    return native_value
