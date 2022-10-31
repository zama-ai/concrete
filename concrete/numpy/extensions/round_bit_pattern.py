"""
Declaration of `round_bit_pattern` function, to provide an interface for rounded table lookups.
"""

from copy import deepcopy
from typing import List, Union

import numpy as np

from ..representation import Node
from ..tracing import Tracer


def round_bit_pattern(
    x: Union[int, List, np.ndarray, Tracer],
    lsbs_to_remove: int,
) -> Union[int, List, np.ndarray, Tracer]:
    """
    Round the bit pattern of an integer.

    x = 0b_0000_0000 , lsbs_to_remove = 3 => 0b_0000_0000
    x = 0b_0000_0001 , lsbs_to_remove = 3 => 0b_0000_0000
    x = 0b_0000_0010 , lsbs_to_remove = 3 => 0b_0000_0000
    x = 0b_0000_0011 , lsbs_to_remove = 3 => 0b_0000_0000
    x = 0b_0000_0100 , lsbs_to_remove = 3 => 0b_0000_1000
    x = 0b_0000_0101 , lsbs_to_remove = 3 => 0b_0000_1000
    x = 0b_0000_0110 , lsbs_to_remove = 3 => 0b_0000_1000
    x = 0b_0000_0111 , lsbs_to_remove = 3 => 0b_0000_1000

    x = 0b_1010_0000 , lsbs_to_remove = 3 => 0b_1010_0000
    x = 0b_1010_0001 , lsbs_to_remove = 3 => 0b_1010_0000
    x = 0b_1010_0010 , lsbs_to_remove = 3 => 0b_1010_0000
    x = 0b_1010_0011 , lsbs_to_remove = 3 => 0b_1010_0000
    x = 0b_1010_0100 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_0101 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_0110 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_0111 , lsbs_to_remove = 3 => 0b_1010_1000

    x = 0b_1010_1000 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_1001 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_1010 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_1011 , lsbs_to_remove = 3 => 0b_1010_1000
    x = 0b_1010_1100 , lsbs_to_remove = 3 => 0b_1011_0000
    x = 0b_1010_1101 , lsbs_to_remove = 3 => 0b_1011_0000
    x = 0b_1010_1110 , lsbs_to_remove = 3 => 0b_1011_0000
    x = 0b_1010_1111 , lsbs_to_remove = 3 => 0b_1011_0000

    x = 0b_1011_1000 , lsbs_to_remove = 3 => 0b_1011_1000
    x = 0b_1011_1001 , lsbs_to_remove = 3 => 0b_1011_1000
    x = 0b_1011_1010 , lsbs_to_remove = 3 => 0b_1011_1000
    x = 0b_1011_1011 , lsbs_to_remove = 3 => 0b_1011_1000
    x = 0b_1011_1100 , lsbs_to_remove = 3 => 0b_1100_0000
    x = 0b_1011_1101 , lsbs_to_remove = 3 => 0b_1100_0000
    x = 0b_1011_1110 , lsbs_to_remove = 3 => 0b_1100_0000
    x = 0b_1011_1111 , lsbs_to_remove = 3 => 0b_1100_0000

    Args:
         x (Union[int, np.ndarray, Tracer]):
            input to round

         lsbs_to_remove (int):
            number of the least significant numbers to remove

    Returns:
        Union[int, np.ndarray, Tracer]:
            Tracer that respresents the operation during tracing
            rounded value(s) otherwise
    """

    def evaluator(x: Union[int, np.ndarray], lsbs_to_remove: int) -> Union[int, np.ndarray]:
        unit = 1 << lsbs_to_remove
        half = 1 << lsbs_to_remove - 1
        rounded = (x + half) // unit
        return rounded * unit

    if isinstance(x, Tracer):
        computation = Node.generic(
            "round_bit_pattern",
            [x.output],
            deepcopy(x.output),
            evaluator,
            kwargs={"lsbs_to_remove": lsbs_to_remove},
        )
        return Tracer(computation, [x])

    if isinstance(x, list):  # pragma: no cover
        try:
            x = np.array(x)
        except Exception:  # pylint: disable=broad-except
            pass

    if isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(
                f"Expected input elements to be integers but they are {type(x.dtype).__name__}"
            )
    elif not isinstance(x, int):
        raise TypeError(f"Expected input to be an int or a numpy array but it's {type(x).__name__}")

    return evaluator(x, lsbs_to_remove)
