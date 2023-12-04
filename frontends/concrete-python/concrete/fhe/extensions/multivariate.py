"""
Declaration of `multivariate` extension.
"""

from copy import deepcopy
from typing import Any, Callable, Optional, Type, Union

import numpy as np

from ..dtypes import BaseDataType, Float
from ..representation import Node
from ..tracing import ScalarAnnotation, Tracer
from ..values import ValueDescription


def multivariate(
    function: Callable,
    outputs: Optional[Union[BaseDataType, Type[ScalarAnnotation]]] = None,
) -> Callable:
    """
    Wrap a multivariate function so that it is traced into a single generic node.

    Args:
        function (Callable[[Any, ...], Any]):
            multivariate function to wrap

        outputs (Optional[Union[BaseDataType, Type[ScalarAnnotation]]], default = None):
            data type of the result
            only required for direct circuits
            ignored when compiling with inputsets

    Returns:
        Callable[[Union[Tracer, Any], ...], Union[Tracer, Any]]:
            another multivariate function that can be called with Tracers as well
    """

    def wrapper(*args: Union[Tracer, Any]) -> Union[Tracer, Any]:
        """
        Evaluate or trace wrapped multivariate function.

        Args:
            args (Tuple[Union[Tracer, Any]]):
                inputs of the function

        Returns:
            Union[Tracer, Any]:
                result of tracing or evaluation
        """

        if not any(isinstance(arg, Tracer) for arg in args):
            return function(*args)

        # pylint: disable=protected-access
        is_direct = Tracer._is_direct
        # pylint: enable=protected-access

        if is_direct:
            if outputs is None:
                message = (
                    "Multivariate extension requires "
                    "`outputs` argument for direct circuit definition "
                    "(e.g., fhe.multivariate(function, outputs=fhe.uint4)(x, y, z))"
                )
                raise ValueError(message)

        if any(not isinstance(arg, Tracer) or not arg.output.is_encrypted for arg in args):
            message = "Multivariate extension requires all of its inputs to be encrypted"
            raise ValueError(message)

        if any(arg.computation.properties.get("name") == "round_bit_pattern" for arg in args):
            message = "Multivariate extension cannot be used with rounded inputs"
            raise ValueError(message)

        dtypes = tuple(
            (
                {64: np.float64, 32: np.float32, 16: np.float16}[arg.output.dtype.bit_width]
                if isinstance(arg.output.dtype, Float)
                else np.int64
            )
            for arg in args
        )
        sample = tuple(
            dtype(1) if arg.output.is_scalar else np.ones(arg.output.shape, dtype=dtype)
            for arg, dtype in zip(args, dtypes)
        )
        evaluation = function(*sample)

        output_value = ValueDescription.of(evaluation, is_encrypted=True)
        if output_value.shape != sum(sample).shape:  # type: ignore
            message = f"Function {function.__name__} is not compatible with fhe.multivariate"
            raise ValueError(message)

        if is_direct:
            assert outputs is not None
            output_value.dtype = outputs if isinstance(outputs, BaseDataType) else outputs.dtype

        computation = Node.generic(
            function.__name__,
            [deepcopy(arg.output) for arg in args],
            output_value,
            lambda *args: function(*args),  # pylint: disable=unnecessary-lambda
            attributes={"is_multivariate": True},
        )
        return Tracer(computation, list(args))

    return wrapper
