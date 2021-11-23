"""This file contains a wrapper class for direct multi table lookups."""

import itertools
from copy import deepcopy
from typing import List, Tuple, Union

from ..data_types.base import BaseDataType
from ..data_types.dtypes_helpers import find_type_to_hold_both_lossy
from ..representation.intermediate import GenericFunction
from ..tracing.base_tracer import BaseTracer
from ..values import TensorValue
from .table import LookupTable


class MultiLookupTable:
    """Class representing a multi lookup table."""

    # Multi table lookup is needed when you want to perform a lookup on a tensor,
    # but you want each element to be used with a different lookup table.
    #
    # Here is an example:
    #
    # You have x which is of shape (2, 3),
    # you want the first row to be indexed with `table1 = LookupTable([2, 3, 1, 0])`
    # and the second row to be indexed with `table1 = LookupTable([0, 1, 3, 2])`
    #
    # You can create such a multi lookup table
    #     multitable = MultiLookupTable(
    #         [
    #             [table1, table1, table1],
    #             [table2, table2, table2],
    #         ],
    #     )
    # (notice the shape of multitable matches with the shape of x)
    #
    # and use multitable[x] toget the following result
    #     assert multitable[x] == [
    #         [table1[x[0, 0]], table1[x[0, 1]], table1[x[0, 2]]],
    #         [table2[x[1, 0]], table2[x[1, 1]], table2[x[1, 2]]],
    #     ]

    # underlying lookup tables
    tables: List

    # shape of the input of the lookup
    input_shape: Tuple[int, ...]

    # type of the result of the lookup
    output_dtype: BaseDataType

    def __init__(self, tables: List):
        input_shape_list: List[int] = []
        MultiLookupTable._extract_shape_using_first_elements_only(tables, input_shape_list)
        input_shape: Tuple[int, ...] = tuple(input_shape_list)

        table_sizes: List[int] = []
        table_output_dtypes: List[BaseDataType] = []
        MultiLookupTable._check_shape_and_record_luts(
            tables,
            0,
            input_shape,
            table_sizes,
            table_output_dtypes,
        )

        for i in range(1, len(table_sizes)):
            if table_sizes[i - 1] != table_sizes[i]:
                # this branch is for such a case:
                #
                #     table1 = hnp.LookupTable([1, 3])
                #     table2 = hnp.LookupTable([0, 2, 3, 1])
                #
                #     multitable = hnp.MultiLookupTable(
                #         [
                #             [table1, table2, table1],
                #             [table2, table1, table2],
                #         ],
                #     )
                raise ValueError(
                    f"LookupTables within a MultiLookupTable "
                    f"should have the same size but they do not "
                    f"(there was a table with the size of {table_sizes[i - 1]} "
                    f"and another with the size of {table_sizes[i]})"
                )

        output_dtype = table_output_dtypes[0]
        for table_output_dtype in table_output_dtypes:
            output_dtype = find_type_to_hold_both_lossy(output_dtype, table_output_dtype)

        self.tables = tables
        self.input_shape = input_shape
        self.output_dtype = output_dtype

    def __getitem__(self, key: Union[int, BaseTracer]):
        # this branch is used during tracing and the regular flow is used during evaluation
        if isinstance(key, BaseTracer):
            out_dtype = deepcopy(key.output.dtype)
            out_shape = deepcopy(self.input_shape)

            generic_function_output_value = TensorValue(
                out_dtype,
                key.output.is_encrypted,
                out_shape,
            )

            traced_computation = GenericFunction(
                inputs=[key.output],
                arbitrary_func=MultiLookupTable._checked_indexing,
                output_value=generic_function_output_value,
                op_kind="TLU",
                op_kwargs={
                    "input_shape": deepcopy(self.input_shape),
                    "tables": deepcopy(self.tables),
                },
                op_name="MultiTLU",
            )
            return key.__class__(
                inputs=[key],
                traced_computation=traced_computation,
                output_idx=0,
            )

        # if not, it means table is indexed with a constant
        # thus, the result of the lookup is a constant
        # so, we can propagate it directly
        return MultiLookupTable._checked_indexing(key, self.input_shape, self.tables)

    @staticmethod
    def _extract_shape_using_first_elements_only(array, shape):
        if not isinstance(array, list):
            # base case for recursion
            # the shape is already accumulated up to this point
            # so we just return
            return

        if len(array) == 0:
            # this branch is for such a case:
            #
            #     table1 = hnp.LookupTable([1, 3, 2, 0])
            #     table2 = hnp.LookupTable([0, 2, 3, 1])
            #
            #     multitable = hnp.MultiLookupTable(
            #         [
            #             [],
            #             [table1, table2, table1],
            #             [table2, table1, table2],
            #         ],
            #     )

            raise ValueError("MultiLookupTable cannot have an empty array within it")

        shape.append(len(array))
        MultiLookupTable._extract_shape_using_first_elements_only(array[0], shape)

    @staticmethod
    def _check_shape_and_record_luts(array, dimension, shape, table_sizes, table_output_dtypes):
        if dimension == len(shape):
            if not isinstance(array, LookupTable):
                # this branch is for such a case:
                #
                #     table1 = hnp.LookupTable([1, 3, 2, 0])
                #     table2 = hnp.LookupTable([0, 2, 3, 1])
                #
                #     multitable = hnp.MultiLookupTable(
                #         [
                #             [table1, table2, 4],
                #             [table2, table1, table2],
                #         ],
                #     )
                raise ValueError(
                    f"MultiLookupTable should have been made out of LookupTables "
                    f"but it had an object of type {array.__class__.__name__} within it"
                )

            table_sizes.append(len(array.table))
            table_output_dtypes.append(array.output_dtype)
            return

        if not isinstance(array, list) or len(array) != shape[dimension]:
            # this branch is for such a case:
            #
            #     table1 = hnp.LookupTable([1, 3, 2, 0])
            #     table2 = hnp.LookupTable([0, 2, 3, 1])
            #
            #     multitable = hnp.MultiLookupTable(
            #         [
            #             [table1, table2],
            #             [table2, table1, table2],
            #         ],
            #     )
            raise ValueError(
                f"MultiLookupTable should have the shape {shape} but it does not "
                f"(an array on dimension {dimension} has the size {len(array)} "
                f"but its size should have been {shape[dimension]} "
                f"as the expected shape is {shape})"
            )

        for item in array:
            MultiLookupTable._check_shape_and_record_luts(
                item,
                dimension + 1,
                shape,
                table_sizes,
                table_output_dtypes,
            )

    @staticmethod
    def _checked_indexing(x, input_shape, tables):
        try:
            result = []
            for indices in itertools.product(*[range(dimension) for dimension in input_shape]):
                which_table_to_use = tables
                what_value_to_use = x
                where_to_append = result

                for index in indices[:-1]:
                    which_table_to_use = tables[index]
                    what_value_to_use = x[index]

                    if len(where_to_append) == index:
                        where_to_append.append([])
                    where_to_append = result[index]

                which_table_to_use = which_table_to_use[indices[-1]]
                what_value_to_use = what_value_to_use[indices[-1]]
                where_to_append.append(which_table_to_use[what_value_to_use])
        except Exception as error:
            raise ValueError(
                f"Multiple Lookup Table of shape {input_shape} cannot be looked up with {x} "
                f"(you should check your inputset)",
            ) from error

        return result
