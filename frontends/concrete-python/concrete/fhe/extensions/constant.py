"""
Declaration of `constant` functions, to allow server side trivial encryption.
"""

from typing import Any, Union

from ..tracing import Tracer
from .zeros import zeros


def constant(x: Union[Tracer, Any]) -> Union[Tracer, Any]:
    """
    Trivial encryption of a cleartext value.
    """

    return zeros(() if isinstance(x, int) else x.shape) + x
