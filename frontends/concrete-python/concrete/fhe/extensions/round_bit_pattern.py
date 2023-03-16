"""
Declaration of `round_bit_pattern` function, to provide an interface for rounded table lookups.
"""

import threading
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np

from ..dtypes import Integer
from ..mlir.utils import MAXIMUM_TLU_BIT_WIDTH
from ..representation import Node
from ..tracing import Tracer
from ..values import Value

local = threading.local()

# pylint: disable=protected-access
local._is_adjusting = False
# pylint: enable=protected-access


class Adjusting(BaseException):
    """
    Adjusting class, to be used as early stop signal during adjustment.
    """

    rounder: "AutoRounder"
    input_min: int
    input_max: int

    def __init__(self, rounder: "AutoRounder", input_min: int, input_max: int):
        super().__init__()
        self.rounder = rounder
        self.input_min = input_min
        self.input_max = input_max


class AutoRounder:
    """
    AutoRounder class, to optimize for number of msbs to keep druing round bit pattern operation.
    """

    target_msbs: int

    is_adjusted: bool
    input_min: int
    input_max: int
    input_bit_width: int
    lsbs_to_remove: int

    def __init__(self, target_msbs: int = MAXIMUM_TLU_BIT_WIDTH):
        # pylint: disable=protected-access
        if local._is_adjusting:
            message = (
                "AutoRounders cannot be constructed during adjustment, "
                "please construct AutoRounders outside the function and reference it"
            )
            raise RuntimeError(message)
        # pylint: enable=protected-access

        self.target_msbs = target_msbs

        self.is_adjusted = False
        self.input_min = 0
        self.input_max = 0
        self.input_bit_width = 0
        self.lsbs_to_remove = 0

    @staticmethod
    def adjust(function: Callable, inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]]]):
        """
        Adjust AutoRounders in a function using an inputset.
        """

        # pylint: disable=protected-access,too-many-branches

        try:  # extract underlying function for decorators
            function = function.function  # type: ignore
            assert callable(function)
        except AttributeError:
            pass

        if local._is_adjusting:
            message = "AutoRounders cannot be adjusted recursively"
            raise RuntimeError(message)

        try:
            local._is_adjusting = True
            while True:
                rounder = None

                for sample in inputset:
                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    try:
                        function(*sample)
                    except Adjusting as adjuster:
                        rounder = adjuster.rounder

                        rounder.input_min = min(rounder.input_min, adjuster.input_min)
                        rounder.input_max = max(rounder.input_max, adjuster.input_max)

                        input_value = Value.of([rounder.input_min, rounder.input_max])
                        assert isinstance(input_value.dtype, Integer)
                        rounder.input_bit_width = input_value.dtype.bit_width

                        if rounder.input_bit_width - rounder.lsbs_to_remove > rounder.target_msbs:
                            rounder.lsbs_to_remove = rounder.input_bit_width - rounder.target_msbs
                    else:
                        return

                if rounder is None:
                    message = "AutoRounders cannot be adjusted with an empty inputset"
                    raise ValueError(message)

                rounder.is_adjusted = True

        finally:
            local._is_adjusting = False

        # pylint: enable=protected-access,too-many-branches


def round_bit_pattern(
    x: Union[int, np.integer, List, np.ndarray, Tracer],
    lsbs_to_remove: Union[int, AutoRounder],
) -> Union[int, np.integer, List, np.ndarray, Tracer]:
    """
    Round the bit pattern of an integer.

    If `lsbs_to_remove` is an `AutoRounder`:
        corresponding integer value will be determined by adjustment process.

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
         x (Union[int, np.integer, np.ndarray, Tracer]):
            input to round

         lsbs_to_remove (Union[int, AutoRounder]):
            number of the least significant bits to remove
            or an auto rounder object which will be used to determine the integer value

    Returns:
        Union[int, np.integer, np.ndarray, Tracer]:
            Tracer that respresents the operation during tracing
            rounded value(s) otherwise
    """

    # pylint: disable=protected-access,too-many-branches

    if isinstance(lsbs_to_remove, AutoRounder):
        if local._is_adjusting:
            if not lsbs_to_remove.is_adjusted:
                raise Adjusting(lsbs_to_remove, int(np.min(x)), int(np.max(x)))  # type: ignore

        elif not lsbs_to_remove.is_adjusted:
            message = (
                "AutoRounders cannot be used before adjustment, "
                "please call AutoRounder.adjust with the function that will be compiled "
                "and provide the exact inputset that will be used for compilation"
            )
            raise RuntimeError(message)

        lsbs_to_remove = lsbs_to_remove.lsbs_to_remove

    assert isinstance(lsbs_to_remove, int)

    def evaluator(
        x: Union[int, np.integer, np.ndarray],
        lsbs_to_remove: int,
    ) -> Union[int, np.integer, np.ndarray]:
        if lsbs_to_remove == 0:
            return x

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
            message = (
                f"Expected input elements to be integers but they are {type(x.dtype).__name__}"
            )
            raise TypeError(message)
    elif not isinstance(x, (int, np.integer)):
        message = f"Expected input to be an int or a numpy array but it's {type(x).__name__}"
        raise TypeError(message)

    return evaluator(x, lsbs_to_remove)

    # pylint: enable=protected-access,too-many-branches
