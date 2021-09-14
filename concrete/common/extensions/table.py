"""This file contains a wrapper class for direct table lookups."""

from copy import deepcopy
from typing import Iterable, Tuple, Union

from ..common_helpers import is_a_power_of_2
from ..data_types.base import BaseDataType
from ..data_types.integers import make_integer_to_hold
from ..representation import intermediate as ir
from ..tracing.base_tracer import BaseTracer


class LookupTable:
    """Class representing a lookup table."""

    # lookup table itself, has 2^N entries
    table: Tuple[int, ...]

    # type of the result of the lookup
    output_dtype: BaseDataType

    def __init__(self, table: Iterable[int]):
        table = tuple(table)

        if not is_a_power_of_2(len(table)):
            raise ValueError(
                f"Desired lookup table has inappropriate number of entries ({len(table)})"
            )

        self.table = table
        self.output_dtype = make_integer_to_hold(table, force_signed=False)

    def __getitem__(self, key: Union[int, BaseTracer]):
        # if a tracer is used for indexing,
        # we need to create an `ArbitraryFunction` node
        # because the result will be determined during the runtime
        if isinstance(key, BaseTracer):
            traced_computation = ir.ArbitraryFunction(
                input_base_value=key.output,
                arbitrary_func=LookupTable._checked_indexing,
                output_dtype=self.output_dtype,
                op_kwargs={"table": deepcopy(self.table)},
                op_name="TLU",
            )
            return key.__class__(
                inputs=[key],
                traced_computation=traced_computation,
                output_index=0,
            )

        # if not, it means table is indexed with a constant
        # thus, the result of the lookup is a constant
        # so, we can propagate it directly
        return LookupTable._checked_indexing(key, self.table)

    @staticmethod
    def _checked_indexing(x, table):
        if x < 0 or x >= len(table):
            raise ValueError(
                f"Lookup table with {len(table)} entries cannot be indexed with {x} "
                f"(you should check your inputset)",
            )

        return table[x]
