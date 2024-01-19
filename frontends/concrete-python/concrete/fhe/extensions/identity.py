"""
Declaration of `identity` extension.
"""

from copy import deepcopy
from typing import Any, Union

from ..representation import Node
from ..tracing import Tracer


def identity(x: Union[Tracer, Any]) -> Union[Tracer, Any]:
    """
    Apply identity function to x.

    Bit-width of the input and the output can be different.

    Args:
        x (Union[Tracer, Any]):
            input to identity

    Returns:
        Union[Tracer, Any]:
            identity tracer if called with a tracer
            deepcopy of the input otherwise
    """

    if not isinstance(x, Tracer):
        return deepcopy(x)

    computation = Node.generic(
        "identity",
        [deepcopy(x.output)],
        x.output,
        lambda x: deepcopy(x),  # pylint: disable=unnecessary-lambda
    )
    return Tracer(computation, [x])
