"""
Declaration of `relu` extension.
"""

from copy import deepcopy
from typing import Any, Union

import numpy as np

from ..dtypes import Integer
from ..representation import Node
from ..tracing import Tracer


def relu(x: Union[Tracer, Any]) -> Union[Tracer, Any]:
    """
    Rectified linear unit extension.

    Computes:
        x if x >= 0 else 0

    Args:
        x (Union[Tracer, Any]):
            input to apply ReLU

    Returns:
        Union[Tracer, Any]:
            Tracer that represent the operation during tracing
            result of ReLU on `x` otherwise
    """

    def evaluator(x):
        return np.where(x >= 0, x, 0)

    if not isinstance(x, Tracer):
        return evaluator(x)

    resulting_value = deepcopy(x.output)
    if isinstance(resulting_value.dtype, Integer) and resulting_value.dtype.is_signed:
        resulting_value.dtype.is_signed = False

    computation = Node.generic(
        "relu",
        [deepcopy(x.output)],
        resulting_value,
        evaluator,
    )
    return Tracer(computation, [x])
