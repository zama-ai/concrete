"""This file contains a wrapper class for direct table lookups."""

from copy import deepcopy
from typing import Iterable, Tuple, Union

from ..common_helpers import is_a_power_of_2
from ..data_types.base import BaseDataType
from ..data_types.integers import make_integer_to_hold_ints
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
        self.output_dtype = make_integer_to_hold_ints(table, force_signed=False)

    def __getitem__(self, item: Union[int, BaseTracer]):
        # if a tracer is used for indexing,
        # we need to create an `ArbitraryFunction` node
        # because the result will be determined during the runtime
        if isinstance(item, BaseTracer):
            traced_computation = ir.ArbitraryFunction(
                input_base_value=item.output,
                arbitrary_func=lambda x, table: table[x],
                output_dtype=self.output_dtype,
                op_kwargs={"table": deepcopy(self.table)},
            )
            return item.__class__(
                inputs=[item],
                traced_computation=traced_computation,
                output_index=0,
            )

        # if not, it means table is indexed with a constant
        # thus, the result of the lookup is a constant
        # so, we can propagate it directly
        return self.table[item]
