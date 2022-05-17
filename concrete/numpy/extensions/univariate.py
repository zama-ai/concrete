"""
Declaration of `univariate` function.
"""

from typing import Any, Callable, Union

import numpy as np

from ..dtypes import Float
from ..representation import Node
from ..tracing import Tracer
from ..values import Value


def univariate(
    function: Callable[[Any], Any],
) -> Callable[[Union[Tracer, Any]], Union[Tracer, Any]]:
    """
    Wrap a univariate function so that it is traced into a single generic node.

    Args:
        function (Callable[[Any], Any]):
            univariate function to wrap

    Returns:
        Callable[[Union[Tracer, Any]], Union[Tracer, Any]]:
            another univariate function that can be called with a Tracer as well
    """

    def wrapper(x: Union[Tracer, Any]) -> Union[Tracer, Any]:
        """
        Evaluate or trace wrapped univariate function.

        Args:
            x (Union[Tracer, Any]):
                input of the function

        Returns:
            Union[Tracer, Any]:
                result of tracing or evaluation
        """

        if isinstance(x, Tracer):
            dtype = (
                {64: np.float64, 32: np.float32, 16: np.float16}[x.output.dtype.bit_width]
                if isinstance(x.output.dtype, Float)
                else np.int64
            )

            sample = dtype(1) if x.output.shape == () else np.ones(x.output.shape, dtype=dtype)
            evaluation = function(sample)

            output_value = Value.of(evaluation, is_encrypted=x.output.is_encrypted)
            if output_value.shape != x.output.shape:
                raise ValueError(f"Function {function.__name__} cannot be used with cnp.univariate")

            computation = Node.generic(
                function.__name__,
                [x.output],
                output_value,
                lambda x: function(x),  # pylint: disable=unnecessary-lambda
            )
            return Tracer(computation, [x])

        return function(x)

    return wrapper
