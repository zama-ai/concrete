"""
Declaration of `truncate_bit_pattern` extension.
"""

import threading
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np

from ..dtypes import Integer
from ..mlir.utils import MAXIMUM_TLU_BIT_WIDTH
from ..representation import Node
from ..tracing import Tracer
from ..values import ValueDescription

local = threading.local()

# pylint: disable=protected-access
local._is_adjusting = False
# pylint: enable=protected-access


class Adjusting(BaseException):
    """
    Adjusting class, to be used as early stop signal during adjustment.
    """

    truncator: "AutoTruncator"
    input_min: int
    input_max: int

    def __init__(self, truncator: "AutoTruncator", input_min: int, input_max: int):
        super().__init__()
        self.truncator = truncator
        self.input_min = input_min
        self.input_max = input_max


class AutoTruncator:
    """
    AutoTruncator class, to optimize for the number of msbs to keep during truncate operation.
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
                "AutoTruncators cannot be constructed during adjustment, "
                "please construct AutoTruncators outside the function and reference it"
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
        Adjust AutoTruncators in a function using an inputset.
        """

        # pylint: disable=protected-access,too-many-branches

        try:  # extract underlying function for decorators
            function = function.function  # type: ignore
            assert callable(function)
        except AttributeError:
            pass

        if local._is_adjusting:
            message = "AutoTruncators cannot be adjusted recursively"
            raise RuntimeError(message)

        try:
            local._is_adjusting = True

            # adjust the truncator using the inputset
            # this loop continues until the return is reached in the loop body
            # which only happens when ALL truncators are adjusted
            # this condition is met if the function can be executed fully
            # without `Adjusting` exception is raised

            while True:
                truncator = None

                for sample in inputset:
                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    try:
                        function(*sample)
                    except Adjusting as adjuster:
                        truncator = adjuster.truncator

                        truncator.input_min = min(truncator.input_min, adjuster.input_min)
                        truncator.input_max = max(truncator.input_max, adjuster.input_max)

                        input_value = ValueDescription.of(
                            [truncator.input_min, truncator.input_max]
                        )
                        assert isinstance(input_value.dtype, Integer)
                        truncator.input_bit_width = input_value.dtype.bit_width

                        if (
                            truncator.input_bit_width - truncator.lsbs_to_remove
                            > truncator.target_msbs
                        ):
                            truncator.lsbs_to_remove = (
                                truncator.input_bit_width - truncator.target_msbs
                            )
                    else:
                        # this branch will be executed if there were no exceptions in the try block
                        return

                if truncator is None:
                    message = "AutoTruncators cannot be adjusted with an empty inputset"
                    raise ValueError(message)

                truncator.is_adjusted = True

        finally:
            local._is_adjusting = False

        # pylint: enable=protected-access,too-many-branches

    def dump_dict(self) -> Dict:
        """
        Dump properties of the truncator to a dict.
        """

        return {
            "target_msbs": self.target_msbs,
            "is_adjusted": self.is_adjusted,
            "input_min": self.input_min,
            "input_max": self.input_max,
            "input_bit_width": self.input_bit_width,
            "lsbs_to_remove": self.lsbs_to_remove,
        }

    @classmethod
    def load_dict(cls, properties: Dict) -> "AutoTruncator":
        """
        Load previously dumped truncator.
        """

        result = AutoTruncator(target_msbs=properties["target_msbs"])

        result.is_adjusted = properties["is_adjusted"]
        result.input_min = properties["input_min"]
        result.input_max = properties["input_max"]
        result.lsbs_to_remove = properties["lsbs_to_remove"]
        result.input_bit_width = properties["input_bit_width"]

        return result


def truncate_bit_pattern(
    x: Union[int, np.integer, List, np.ndarray, Tracer],
    lsbs_to_remove: Union[int, AutoTruncator],
) -> Union[int, np.integer, List, np.ndarray, Tracer]:
    """
    Round the bit pattern of an integer.

    If `lsbs_to_remove` is an `AutoTruncator`:
        corresponding integer value will be determined by adjustment process.

    x = 0b_0000 , lsbs_to_remove = 2 => 0b_0000
    x = 0b_0001 , lsbs_to_remove = 2 => 0b_0000
    x = 0b_0010 , lsbs_to_remove = 2 => 0b_0000
    x = 0b_0100 , lsbs_to_remove = 2 => 0b_0100
    x = 0b_0110 , lsbs_to_remove = 2 => 0b_0100
    x = 0b_1100 , lsbs_to_remove = 2 => 0b_1100
    x = 0b_abcd , lsbs_to_remove = 2 => 0b_ab00

    Args:
         x (Union[int, np.integer, np.ndarray, Tracer]):
            input to truncate

         lsbs_to_remove (Union[int, AutoTruncator]):
            number of the least significant bits to clear
            or an auto truncator object which will be used to determine the integer value

    Returns:
        Union[int, np.integer, np.ndarray, Tracer]:
            Tracer that represents the operation during tracing
            truncated value(s) otherwise
    """

    # pylint: disable=protected-access,too-many-branches

    if isinstance(lsbs_to_remove, AutoTruncator):
        if local._is_adjusting:
            if not lsbs_to_remove.is_adjusted:
                raise Adjusting(lsbs_to_remove, int(np.min(x)), int(np.max(x)))  # type: ignore

        elif not lsbs_to_remove.is_adjusted:
            message = (
                "AutoTruncators cannot be used before adjustment, "
                "please call AutoTruncator.adjust with the function that will be compiled "
                "and provide the exact inputset that will be used for compilation"
            )
            raise RuntimeError(message)

        lsbs_to_remove = lsbs_to_remove.lsbs_to_remove

    assert isinstance(lsbs_to_remove, int)

    def evaluator(
        x: Union[int, np.integer, np.ndarray],
        lsbs_to_remove: int,
    ) -> Union[int, np.integer, np.ndarray]:
        return (x >> lsbs_to_remove) << lsbs_to_remove

    if isinstance(x, Tracer):
        computation = Node.generic(
            "truncate_bit_pattern",
            [deepcopy(x.output)],
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
