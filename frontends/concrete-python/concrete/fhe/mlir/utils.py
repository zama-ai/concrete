"""
Declaration of various functions and constants related to MLIR conversion.
"""

# pylint: disable=import-error

from collections import defaultdict, deque
from copy import deepcopy
from itertools import chain, product
from typing import Any, DefaultDict, List, Optional, Tuple, Union, cast

import numpy as np
from mlir.dialects import tensor
from mlir.dialects._ods_common import get_op_results_or_values
from mlir.ir import OpResult as MlirOperation
from mlir.ir import Type as MlirType
from mlir.ir import Value as MlirValue

from ..dtypes import Integer
from ..internal.utils import assert_that
from ..representation import Node, Operation

# pylint: enable=import-error

MAXIMUM_TLU_BIT_WIDTH = 16


class HashableNdarray:
    """
    HashableNdarray class, to use numpy arrays in dictionaries.
    """

    array: np.ndarray

    def __init__(self, array: np.ndarray):
        self.array = array

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HashableNdarray) and np.array_equal(self.array, other.array)

    def __hash__(self) -> int:
        return hash(self.array.tobytes())


def flood_replace_none_values(table: list):
    """
    Use flooding algorithm to replace `None` values.

    Args:
        table (list):
            the list in which there are `None` values that need to be replaced
            with copies of the closest non `None` data from the list
    """

    assert_that(any(value is not None for value in table))

    not_none_values_idx = deque(idx for idx, value in enumerate(table) if value is not None)
    while not_none_values_idx:
        current_idx = not_none_values_idx.popleft()
        current_value = table[current_idx]

        previous_idx = current_idx - 1
        next_idx = current_idx + 1

        if previous_idx >= 0 and table[previous_idx] is None:  # pragma: no cover
            table[previous_idx] = deepcopy(current_value)
            not_none_values_idx.append(previous_idx)

        if next_idx < len(table) and table[next_idx] is None:  # pragma: no cover
            table[next_idx] = deepcopy(current_value)
            not_none_values_idx.append(next_idx)

    assert_that(all(value is not None for value in table))


def construct_table(node: Node, preds: List[Node]) -> List[Any]:
    """
    Construct the lookup table for an Operation.Generic node.

    Args:
        node (Node):
            Operation.Generic to construct the table

        preds (List[Node]):
            ordered predecessors to `node`

    Returns:
        List[Any]:
            lookup table corresponding to `node` and its input value
    """

    variable_input_index = -1
    for index, pred in enumerate(preds):
        if pred.operation != Operation.Constant:
            variable_input_index = index
            break
    assert_that(variable_input_index != -1)

    variable_input_dtype = node.inputs[variable_input_index].dtype
    variable_input_shape = node.inputs[variable_input_index].shape

    assert_that(isinstance(variable_input_dtype, Integer))
    variable_input_dtype = deepcopy(cast(Integer, variable_input_dtype))

    variable_input = preds[variable_input_index]
    if (
        variable_input.operation == Operation.Generic
        and variable_input.properties["name"] == "round_bit_pattern"
    ):
        resulting_bit_width = variable_input.properties["resulting_bit_width"]
        expected_number_of_elements = 2**resulting_bit_width

        overflow_protection = variable_input.properties["overflow_protection"]
        overflow_detected = variable_input.properties["overflow_detected"]

        variable_input_dtype.bit_width = variable_input.properties["original_input_bit_width"]
        if overflow_protection and overflow_detected:
            variable_input_dtype.bit_width += 1

        step = (2**variable_input_dtype.bit_width) // expected_number_of_elements
    else:
        step = 1

    values = chain(
        range(0, variable_input_dtype.max() + 1, step),
        range(variable_input_dtype.min(), 0, step),
    )

    np.seterr(divide="ignore")

    inputs: List[Any] = [pred() if pred.operation == Operation.Constant else None for pred in preds]
    table: List[Optional[Union[np.bool_, np.integer, np.floating, np.ndarray]]] = []
    for value in values:
        try:
            inputs[variable_input_index] = np.ones(variable_input_shape, dtype=np.int64) * value
            table.append(node(*inputs))
        except Exception:  # pylint: disable=broad-except
            # here we try our best to fill the table
            # if it fails, we append None and let flooding algoritm replace None values below
            table.append(None)

    np.seterr(divide="warn")

    flood_replace_none_values(table)

    return table


def construct_deduplicated_tables(
    node: Node,
    preds: List[Node],
) -> Tuple[Tuple[np.ndarray, List[Tuple[int, ...]]], ...]:
    """
    Construct lookup tables for each cell of the input for an Operation.Generic node.

    Args:
        node (Node):
            Operation.Generic to construct the table

        preds (List[Node]):
            ordered predecessors to `node`

    Returns:
        Tuple[Tuple[numpy.ndarray, List[Tuple[int, ...]]], ...]:
            tuple containing tuples of 2 for
                - constructed table
                - list of indices of the input that use the constructed table

            e.g.,

            .. code-block:: python

                (
                    (np.array([3, 1, 2, 4]), [(1, 0), (2, 1)]),
                    (np.array([5, 8, 6, 7]), [(0, 0), (0, 1), (1, 1), (2, 0)]),
                )

            means the lookup on 3x2 input will result in

            .. code-block:: python

                [ [5, 8, 6, 7][input[0, 0]] , [5, 8, 6, 7][input[0, 1]] ]
                [ [3, 1, 2, 4][input[1, 0]] , [5, 8, 6, 7][input[1, 1]] ]
                [ [5, 8, 6, 7][input[2, 0]] , [3, 1, 2, 4][input[2, 1]] ]
    """

    node_complete_table = np.concatenate(
        tuple(np.expand_dims(array, -1) for array in construct_table(node, preds)),
        axis=-1,
    )

    all_cells_idx = product(*tuple(range(max_val) for max_val in node_complete_table.shape[:-1]))
    tables_to_cell_idx: DefaultDict[HashableNdarray, List[Tuple[int, ...]]] = defaultdict(list)

    idx: Tuple[int, ...]
    all_idx_set = set()
    for idx in all_cells_idx:
        hashable_array = HashableNdarray(node_complete_table[idx])
        tables_to_cell_idx[hashable_array].append(idx)
        all_idx_set.add(idx)

    assert_that(len(all_idx_set) == np.prod(node_complete_table.shape[:-1]))

    return tuple(
        (hashable_array.array, indices) for hashable_array, indices in tables_to_cell_idx.items()
    )


class _FromElementsOp(tensor.FromElementsOp):
    """Replace missing tensor.FromElementsOp.__init__."""

    def __init__(self, result: MlirType, *elements: MlirOperation):
        assert isinstance(result, MlirType)
        elements = get_op_results_or_values(list(elements))
        assert all(isinstance(element, (MlirOperation, MlirValue)) for element in elements)
        super(tensor.FromElementsOp, self).__init__(
            self.build_generic(
                attributes={},
                results=[result],
                operands=elements,
                successors=None,
                regions=None,
                loc=None,
                ip=None,
            )
        )
