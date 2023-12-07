"""
Bit extraction extensions.
"""

from copy import deepcopy
from typing import Union

import numpy as np

from ..dtypes import Integer
from ..representation import Node
from ..representation.utils import format_indexing_element
from ..tracing import Tracer

MIN_EXTRACTABLE_BIT = 0
MAX_EXTRACTABLE_BIT = 63


class Bits:
    """
    Bits class, to provide indexing into the bits of integers.
    """

    value: Union[int, np.integer, np.ndarray, Tracer]

    def __init__(self, value: Union[int, np.integer, list, np.ndarray, Tracer]):
        if isinstance(value, list):  # pragma: no cover
            value = np.array(value)
        self.value = value

    def __getitem__(self, index: Union[int, np.integer, slice]) -> Tracer:
        if isinstance(index, slice):
            if index.start is not None:
                if not isinstance(index.start, (int, np.integer)):
                    message = (
                        f"Extracting bits using a non integer start "
                        f"(e.g., {index.start}) isn't supported"
                    )
                    raise ValueError(message)

                if index.start < 0:
                    message = (
                        f"Extracting bits using a negative start "
                        f"(e.g., {index.start}) isn't supported"
                    )
                    raise ValueError(message)

            if index.stop is not None:
                if not isinstance(index.stop, (int, np.integer)):
                    message = (
                        f"Extracting bits using a non integer stop "
                        f"(e.g., {index.stop}) isn't supported"
                    )
                    raise ValueError(message)

                if index.stop < 0:
                    message = (
                        f"Extracting bits using a negative stop "
                        f"(e.g., {index.stop}) isn't supported"
                    )
                    raise ValueError(message)

            if index.step is not None:
                if not isinstance(index.step, (int, np.integer)):
                    message = (
                        f"Extracting bits using a non integer step "
                        f"(e.g., {index.step}) isn't supported"
                    )
                    raise ValueError(message)

                if index.step == 0:
                    message = "Extracting bits using zero step isn't supported"
                    raise ValueError(message)

                if index.step < 0 and index.start is None:
                    message = (
                        "Extracting bits in reverse (step < 0) "
                        "isn't supported without providing the start bit"
                    )
                    raise ValueError(message)

        elif isinstance(index, (int, np.integer)):
            if index < 0:
                message = f"Extracting bits from the back (index == {index} < 0) isn't supported"
                raise ValueError(message)

        else:
            message = (
                f"Bits of {self.value} cannot be extracted "
                f"using {format_indexing_element(index)} "
                f"since it's not an integer or a slice"
            )
            raise ValueError(message)

        input_is_integer = isinstance(self.value, (int, np.integer, np.ndarray, Tracer))

        if isinstance(self.value, Tracer):
            input_is_integer = isinstance(self.value.output.dtype, Integer)
        elif isinstance(self.value, np.ndarray):
            input_is_integer = np.issubdtype(self.value.dtype, np.integer)

        if not input_is_integer:
            message = f"Bits of {self.value} cannot be extracted since it's not an integer"
            raise ValueError(message)

        def evaluator(x, bits):  # pylint: disable=redefined-outer-name
            if isinstance(bits, (int, np.integer)):
                return (x & (1 << bits)) >> bits

            assert isinstance(bits, slice)

            step = bits.step or 1

            assert step != 0
            assert step > 0 or bits.start is not None

            if np.any(x < 0) and bits.stop is None and step > 0:
                message = (
                    f"Extracting bits without an upper bound (stop is None) "
                    f"isn't supported on signed values (e.g., {x})"
                )
                raise ValueError(message)

            start = bits.start or MIN_EXTRACTABLE_BIT
            stop = bits.stop or (MAX_EXTRACTABLE_BIT if step > 0 else (MIN_EXTRACTABLE_BIT - 1))

            result = 0
            for i, bit in enumerate(range(start, stop, step)):
                value = (x & (1 << bit)) >> bit
                result += value << i

            return result

        if isinstance(self.value, Tracer):
            computation = Node.generic(
                "extract_bit_pattern",
                [deepcopy(self.value.output)],
                deepcopy(self.value.output),
                evaluator,
                kwargs={"bits": index},
            )
            return Tracer(computation, [self.value])

        return evaluator(self.value, bits=index)


def bits(x: Union[int, np.integer, list, np.ndarray, Tracer]) -> Bits:
    """
    Extract bits of integers.

    Args:
         x (Union[int, np.integer, list, np.ndarray, Tracer]):
            input to extract bits from

    Returns:
        Bits:
            Auxiliary class to represent bits of the integer
    """

    return Bits(x)
