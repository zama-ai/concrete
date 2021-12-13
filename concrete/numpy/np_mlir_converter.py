"""Numpy-specific MLIR converter."""

import math
from collections import defaultdict
from itertools import product
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy

from ..common.debugging import assert_true
from ..common.mlir.graph_converter import OPGraphConverter
from ..common.operator_graph import OPGraph
from ..common.representation.intermediate import GenericFunction, IntermediateNode


class HashableNPArray:
    """Class to easily manipulate numpy arrays for hashing.

    Note that the hash behavior won't work if the array is modified after being hashed, as it will
    have been hashed to a certain value and the new array content will be hashed to a different one.
    """

    array: numpy.ndarray

    def __init__(self, array: numpy.ndarray) -> None:
        self.array = array

    def __hash__(self) -> int:
        return hash(self.array.tobytes())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HashableNPArray) and numpy.array_equal(self.array, other.array)


def generate_deduplicated_tables(
    node: GenericFunction, ordered_preds: List[IntermediateNode]
) -> Tuple[Tuple[numpy.ndarray, List[Tuple[int, ...]]], ...]:
    """Deduplicate the tables for the different cells of a tensor if needed.

    Args:
        node (GenericFunction): the node for which to deduplicate the table.
        ordered_preds (List[IntermediateNode]): ordered list of predecessors of the node.

    Returns:
        Tuple[Tuple[numpy.ndarray, List[Tuple[int, ...]]], ...]: A tuple containing tuples whose
            first element is a table and the second element is a list of tuples indicating which
            cells in the tensor will use that table.
    """
    # This is the tensor containing the tables for each cell of the tensor for node
    node_complete_table = numpy.concatenate(
        tuple(numpy.expand_dims(array, -1) for array in node.get_table(ordered_preds)), axis=-1
    )

    all_cells_idx = product(*tuple(range(max_val) for max_val in node_complete_table.shape[:-1]))
    tables_to_cell_idx: DefaultDict[HashableNPArray, List[Tuple[int, ...]]] = defaultdict(list)
    idx: Tuple[int, ...]
    all_idx_set = set()
    for idx in all_cells_idx:
        hashable_array = HashableNPArray(node_complete_table[idx])
        tables_to_cell_idx[hashable_array].append(idx)
        all_idx_set.add(idx)

    assert_true(len(all_idx_set) == math.prod(node_complete_table.shape[:-1]))

    return tuple(
        (hashable_array.array, indices) for hashable_array, indices in tables_to_cell_idx.items()
    )


class NPMLIRConverter(OPGraphConverter):
    """Numpy-specific MLIR converter."""

    @staticmethod
    def _generate_additional_info_dict(op_graph: OPGraph) -> Dict[str, Any]:
        additional_conversion_info = {}

        # Disable numpy warnings during conversion to avoid issues during TLU generation
        with numpy.errstate(all="ignore"):
            additional_conversion_info["tables"] = {
                node: generate_deduplicated_tables(node, op_graph.get_ordered_preds(node))
                for node in op_graph.graph.nodes()
                if isinstance(node, GenericFunction) and node.op_kind == "TLU"
            }

        return additional_conversion_info
