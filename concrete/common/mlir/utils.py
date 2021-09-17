"""Utilities for MLIR conversion."""
from typing import cast

from ..data_types import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_clear_tensor_integer,
    value_is_encrypted_scalar_integer,
    value_is_encrypted_tensor_integer,
    value_is_scalar_integer,
)
from ..debugging.custom_assert import assert_true
from ..operator_graph import OPGraph
from ..representation.intermediate import ArbitraryFunction

# TODO: should come from compiler, through an API, #402
ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB = 7


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
            if value_is_clear_scalar_integer(value) or value_is_clear_tensor_integer(value):
                value.data_type.bit_width = p + 1
            elif value_is_encrypted_scalar_integer(value) or value_is_encrypted_tensor_integer(
                value
            ):
                value.data_type.bit_width = p


def update_bit_width_for_mlir(op_graph: OPGraph):
    """Prepare bit_width of all nodes to be the same, set to the maximum value in the graph.

    Args:
        op_graph: graph to update bit_width for
    """
    max_bit_width = 0
    offending_list = []
    for node in op_graph.graph.nodes:
        for value_out in node.outputs:
            if value_is_clear_scalar_integer(value_out) or value_is_clear_tensor_integer(value_out):
                current_node_out_bit_width = value_out.data_type.bit_width - 1
            else:

                assert_true(
                    value_is_encrypted_scalar_integer(value_out)
                    or value_is_encrypted_tensor_integer(value_out)
                )

                current_node_out_bit_width = value_out.data_type.bit_width

            max_bit_width = max(max_bit_width, current_node_out_bit_width)

            # Check that current_node_out_bit_width is supported by the compiler
            if current_node_out_bit_width > ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB:
                offending_list.append((node, current_node_out_bit_width))

    # TODO: remove this workaround, which was for #279, once the compiler can handle
    # smaller tables, #412
    has_a_table = any(isinstance(node, ArbitraryFunction) for node in op_graph.graph.nodes)

    if has_a_table:
        max_bit_width = ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB

    _set_all_bit_width(op_graph, max_bit_width)

    # Check that the max_bit_width is supported by the compiler
    if len(offending_list) != 0:
        raise RuntimeError(
            f"max_bit_width of some nodes is too high for the current version of "
            f"the compiler (maximum must be {ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB} "
            f"which is not compatible with {offending_list})"
        )


def extend_direct_lookup_tables(op_graph: OPGraph):
    """Extend direct lookup tables to the maximum length the input bit width can support.

    Args:
        op_graph: graph to update lookup tables for
    """
    for node in op_graph.graph.nodes:
        if isinstance(node, ArbitraryFunction) and node.op_name == "TLU":
            table = node.op_kwargs["table"]
            bit_width = cast(Integer, node.inputs[0].data_type).bit_width
            expected_length = 2 ** bit_width
            if len(table) > expected_length:
                node.op_kwargs["table"] = table[:expected_length]
            else:
                repeat = expected_length // len(table)
                node.op_kwargs["table"] = (table * repeat)[:expected_length]
