"""
Declaration of hinting extensions, to provide more information to Concrete.
"""

from typing import Any, Optional, Union

from ..tracing import Tracer


def hint(x: Union[Tracer, Any], *, bit_width: Optional[int] = None) -> Union[Tracer, Any]:
    """
    Hint the compilation process about properties of a value.

    Hints are useful if you know something about a value, but it's hard to cover in the inputset.
    An example of this can be a complex circuit doing a lot of bitwise operations on 8-bits.
    It's very hard to make sure every intermediate has 8-bits, but you can use hints to solve this.
    If you mark your intermediates using this function to be 8-bits, they'll be assigned
    at least 8-bits during the bit-width assignment step.

    Args:
        x (Union[Tracer, Any]):
            value to hint

        bit_width (Optional[int], default = None):
            hint about bit width

    Returns:
        Union[Tracer, Any]:
            hinted value
    """

    if not isinstance(x, Tracer):  # pragma: no cover
        return x

    if bit_width is not None:
        x.computation.properties["bit_width_hint"] = bit_width

    return x
