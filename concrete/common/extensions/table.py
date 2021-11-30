"""This file contains a wrapper class for direct table lookups."""

from copy import deepcopy
from typing import Any, Iterable, List, Tuple, Union

from ..common_helpers import is_a_power_of_2
from ..data_types.base import BaseDataType
from ..data_types.integers import make_integer_to_hold
from ..representation.intermediate import GenericFunction
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

    def __repr__(self):
        return str(list(self.table))

    def __getitem__(self, key: Union[int, Iterable, BaseTracer]):
        # if a tracer is used for indexing,
        # we need to create an `GenericFunction` node
        # because the result will be determined during the runtime
        if isinstance(key, BaseTracer):
            generic_function_output_value = deepcopy(key.output)
            generic_function_output_value.dtype = self.output_dtype

            traced_computation = GenericFunction(
                inputs=[key.output],
                arbitrary_func=LookupTable._checked_indexing,
                output_value=generic_function_output_value,
                op_kind="TLU",
                op_kwargs={"table": deepcopy(self.table)},
                op_name="TLU",
            )
            return key.__class__(
                inputs=[key],
                traced_computation=traced_computation,
                output_idx=0,
            )

        # if not, it means table is indexed with a constant
        # thus, the result of the lookup is a constant
        # so, we can propagate it directly
        return LookupTable._checked_indexing(key, self.table)

    @staticmethod
    def _check_index_out_of_range(x, table):
        if not -len(table) <= x < len(table):
            raise ValueError(
                f"Lookup table with {len(table)} entries cannot be indexed with {x} "
                f"(you should check your inputset)",
            )

    @staticmethod
    def _checked_indexing(x, table):
        """Index `table` using `x`.

        There is a single table and the indexing works with the following semantics:
        - when x == c
            - table[x] == table[c]
        - when x == [c1, c2]
            - table[x] == [table[c1], table[c2]]
        - when x == [[c1, c2], [c3, c4], [c5, c6]]
            - table[x] == [[table[c1], table[c2]], [table[c3], table[c4]], [table[c5], table[c6]]]

        Args:
            x (Union[int, Iterable]): index to use
            table (Tuple[int, ...]): table to index

        Returns:
            Union[int, List[int]]: result of indexing
        """

        if not isinstance(x, Iterable):
            LookupTable._check_index_out_of_range(x, table)
            return table[x]

        def fill_result(partial_result: List[Any], partial_x: Iterable[Any]):
            """Fill partial result with partial x.

            This function implements the recursive indexing of nested iterables.

            Args:
                partial_result (List[Any]): currently accumulated result
                partial_x (Iterable[Any]): current index to use

            Returns:
                None
            """

            for item in partial_x:
                if isinstance(item, Iterable):
                    partial_result.append([])
                    fill_result(partial_result[-1], item)
                else:
                    LookupTable._check_index_out_of_range(item, table)
                    partial_result.append(table[item])

        result = []
        fill_result(result, x)
        return result
