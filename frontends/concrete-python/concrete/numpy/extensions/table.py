"""
Declaration of `LookupTable` class.
"""

from copy import deepcopy
from typing import Any, Union

import numpy as np

from ..dtypes import BaseDataType, Integer
from ..representation import Node
from ..tracing import Tracer


class LookupTable:
    """
    LookupTable class, to provide a way to do direct table lookups.
    """

    table: np.ndarray
    output_dtype: BaseDataType

    def __init__(self, table: Any):
        is_valid = True
        try:
            self.table = table if isinstance(table, np.ndarray) else np.array(table)
        except Exception:  # pragma: no cover  # pylint: disable=broad-except
            # here we try our best to convert the table to np.ndarray
            # if it fails we raise the exception at the end of the function
            is_valid = False

        if is_valid:
            is_valid = self.table.size > 0

            if is_valid:
                minimum: int = 0
                maximum: int = 0

                if np.issubdtype(self.table.dtype, np.integer):
                    minimum = int(self.table.min())
                    maximum = int(self.table.max())
                    if self.table.ndim != 1:
                        is_valid = False
                else:
                    is_valid = all(isinstance(item, LookupTable) for item in self.table.flat)
                    if is_valid:
                        minimum = int(self.table.flat[0].table.min())
                        maximum = int(self.table.flat[0].table.max())
                        for item in self.table.flat:
                            minimum = min(minimum, item.table.min())
                            maximum = max(maximum, item.table.max())

                self.output_dtype = Integer.that_can_represent([minimum, maximum])

        if not is_valid:
            message = f"LookupTable cannot be constructed with {repr(table)}"
            raise ValueError(message)

    def __repr__(self):
        return str(list(self.table))

    def __getitem__(self, key: Union[int, np.integer, np.ndarray, Tracer]):
        if not isinstance(key, Tracer):
            return LookupTable.apply(key, self.table)

        if not isinstance(key.output.dtype, Integer):
            message = f"LookupTable cannot be looked up with {key.output}"
            raise ValueError(message)

        table = self.table
        if not np.issubdtype(self.table.dtype, np.integer):
            try:
                table = np.broadcast_to(table, key.output.shape)
            except Exception as error:
                message = (
                    f"LookupTable of shape {self.table.shape} "
                    f"cannot be looked up with {key.output}"
                )
                raise ValueError(message) from error

        output = deepcopy(key.output)
        output.dtype = self.output_dtype

        computation = Node.generic(
            "tlu",
            [key.output],
            output,
            LookupTable.apply,
            kwargs={"table": table},
        )
        return Tracer(computation, [key])

    @staticmethod
    def apply(
        key: Union[int, np.integer, np.ndarray],
        table: np.ndarray,
    ) -> Union[int, np.integer, np.ndarray]:
        """
        Apply lookup table.

        Args:
            key (Union[int, np.integer, np.ndarray]):
                lookup key

            table (np.ndarray):
                lookup table

        Returns:
            Union[int, np.integer, np.ndarray]:
                lookup result

        Raises:
            ValueError:
                if `table` cannot be looked up with `key`
        """

        if not isinstance(key, (int, np.integer, np.ndarray)) or (
            isinstance(key, np.ndarray) and not np.issubdtype(key.dtype, np.integer)
        ):
            message = f"LookupTable cannot be looked up with {key}"
            raise ValueError(message)

        if np.issubdtype(table.dtype, np.integer):
            return table[key]

        if not isinstance(key, np.ndarray) or key.shape != table.shape:
            message = f"LookupTable of shape {table.shape} cannot be looked up with {key}"
            raise ValueError(message)

        flat_result = np.fromiter(
            (lt.table[k] for lt, k in zip(table.flat, key.flat)),
            dtype=np.longlong,
        )
        return flat_result.reshape(table.shape)
