"""Utilities for MLIR conversion."""
from typing import cast

from ..data_types import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_encrypted_scalar_integer,
    value_is_scalar_integer,
)
from ..operator_graph import OPGraph


def is_graph_values_compatible_with_mlir(op_graph: OPGraph) -> bool:
    """Make sure the graph outputs are unsigned integers, which is what the compiler supports.

    Args:
        op_graph: computation graph to check

    Returns:
        bool: is the graph compatible with the expected MLIR representation
    """
    return all(
        all(
            value_is_scalar_integer(out) and not cast(Integer, out.data_type).is_signed
            for out in out_node.outputs
        )
        for out_node in op_graph.output_nodes.values()
    )


def _set_all_bit_width(op_graph: OPGraph, p: int):
    """Set all bit_width in the graph to `p` and `p+1` for clear and encrypted values respectively.

    Args:
        op_graph: graph to set bit_width for
        p: bit_width to set everywhere
    """
    for node in op_graph.graph.nodes:
        for value in node.outputs + node.inputs:
            if value_is_clear_scalar_integer(value):
                value.data_type.bit_width = p + 1
            elif value_is_encrypted_scalar_integer(value):
                value.data_type.bit_width = p


def update_bit_width_for_mlir(op_graph: OPGraph):
    """Prepare bit_width of all nodes to be the same, set to the maximum value in the graph.

    Args:
        op_graph: graph to update bit_width for
    """
    max_bit_width = 0
    for node in op_graph.graph.nodes:
        for value_out in node.outputs:
            if value_is_clear_scalar_integer(value_out):
                max_bit_width = max(max_bit_width, value_out.data_type.bit_width - 1)
            elif value_is_encrypted_scalar_integer(value_out):
                max_bit_width = max(max_bit_width, value_out.data_type.bit_width)
    _set_all_bit_width(op_graph, max_bit_width)
