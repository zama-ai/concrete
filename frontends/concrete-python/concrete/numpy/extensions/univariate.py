"""
Declaration of `univariate` function.
"""

from typing import Any, Callable, Optional, Type, Union

import numpy as np

from ..dtypes import BaseDataType, Float
from ..representation import Node
from ..tracing import ScalarAnnotation, Tracer
from ..values import Value


def univariate(
    function: Callable[[Any], Any],
    outputs: Optional[Union[BaseDataType, Type[ScalarAnnotation]]] = None,
) -> Callable[[Union[Tracer, Any]], Union[Tracer, Any]]:
    """
    Wrap a univariate function so that it is traced into a single generic node.

    Args:
        function (Callable[[Any], Any]):
            univariate function to wrap

        outputs (Optional[Union[BaseDataType, Type[ScalarAnnotation]]], default = None):
            data type of the result, unused during compilation, required for direct definition

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

            if x.output.shape == ():
                sample = dtype(1)  # type: ignore
            else:
                sample = np.ones(x.output.shape, dtype=dtype)
            evaluation = function(sample)

            output_value = Value.of(evaluation, is_encrypted=x.output.is_encrypted)
            if output_value.shape != x.output.shape:
                message = f"Function {function.__name__} cannot be used with cnp.univariate"
                raise ValueError(message)

            # pylint: disable=protected-access
            is_direct = Tracer._is_direct
            # pylint: enable=protected-access

            if is_direct:
                if outputs is None:
                    message = (
                        "Univariate extension requires "
                        "`outputs` argument for direct circuit definition "
                        "(e.g., cnp.univariate(function, outputs=cnp.uint4)(x))"
                    )
                    raise ValueError(message)
                output_value.dtype = outputs if isinstance(outputs, BaseDataType) else outputs.dtype

            computation = Node.generic(
                function.__name__,
                [x.output],
                output_value,
                lambda x: function(x),  # pylint: disable=unnecessary-lambda
            )
            return Tracer(computation, [x])

        return function(x)

    return wrapper
