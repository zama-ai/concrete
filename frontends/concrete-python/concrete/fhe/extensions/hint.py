"""
Declaration of hinting extensions, to provide more information to Concrete.
"""

from typing import Any, Optional, Union

from ..dtypes import Integer
from ..tracing import Tracer


def hint(
    x: Union[Tracer, Any],
    *,
    bit_width: Optional[int] = None,
    can_store: Optional[Any] = None,
) -> Union[Tracer, Any]:
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

        can_store (Optional[Any], default = None):
            hint that the value needs to be able to store the given value

    Returns:
        Union[Tracer, Any]:
            hinted value
    """

    if not isinstance(x, Tracer):  # pragma: no cover
        return x

    bit_width_hint = 0

    if bit_width is not None:
        bit_width_hint = max(bit_width_hint, bit_width)

    if can_store is not None:
        bit_width_hint = max(bit_width_hint, Integer.that_can_represent(can_store).bit_width)

    if bit_width_hint > 0:
        node_to_hint = x if x.last_version is None else x.last_version
        node_to_hint.computation.properties["bit_width_hint"] = bit_width_hint

    return x
