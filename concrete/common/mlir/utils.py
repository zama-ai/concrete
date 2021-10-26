"""Utilities for MLIR conversion."""
from typing import Dict, List, Optional, cast

from ..data_types import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_clear_tensor_integer,
    value_is_encrypted_scalar_integer,
    value_is_encrypted_tensor_integer,
    value_is_integer,
    value_is_scalar,
    value_is_unsigned_integer,
)
from ..debugging import get_printable_graph
from ..debugging.custom_assert import assert_not_reached, assert_true
from ..operator_graph import OPGraph
from ..representation import intermediate
from ..representation.intermediate import IntermediateNode, UnivariateFunction

# TODO: should come from compiler, through an API, #402
ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB = 7


def check_node_compatibility_with_mlir(node: IntermediateNode, is_output: bool) -> Optional[str]:
    """Check if node is compatible with MLIR.

    Args:
        node (IntermediateNode): node to check
        is_output (bool): whether the node is an output node or not

    Returns:
        Optional[str]: None if the node is compatible else reason for incompatibility
    """

    # pylint: disable=too-many-branches,too-many-return-statements

    inputs = node.inputs
    outputs = node.outputs

    if isinstance(node, intermediate.Add):  # constraints for addition
        for inp in inputs:
            if not value_is_scalar(inp):
                return "only scalar addition is supported"

    elif isinstance(node, intermediate.Sub):  # constraints for subtraction
        for inp in inputs:
            if not value_is_scalar(inp):
                return "only scalar subtraction is supported"

    elif isinstance(node, intermediate.Mul):  # constraints for multiplication
        for inp in inputs:
            if not value_is_scalar(inp):
                return "only scalar multiplication is supported"

    elif isinstance(node, intermediate.Input):  # constraints for inputs
        assert_true(len(outputs) == 1)
        if not value_is_unsigned_integer(outputs[0]):
            return "only unsigned integer inputs are supported"

    elif isinstance(node, intermediate.Constant):  # constraints for constants
        assert_true(len(outputs) == 1)
        # We currently can't fail on the following assert, but let it for possible changes in the
        # future
        if not value_is_integer(outputs[0]):
            return "only integer constants are supported"  # pragma: no cover

    elif isinstance(node, intermediate.UnivariateFunction):  # constraints for univariate functions
        assert_true(len(inputs) == 1)
        if not value_is_scalar(inputs[0]) or not value_is_unsigned_integer(inputs[0]):
            return "only unsigned integer scalar lookup tables are supported"

    elif isinstance(node, intermediate.Dot):  # constraints for dot product
        assert_true(len(inputs) == 2)
        if not value_is_unsigned_integer(inputs[0]) or not value_is_unsigned_integer(inputs[1]):
            return "only unsigned integer dot product is supported"

    elif isinstance(node, intermediate.IndexConstant):  # constraints for constant indexing
        assert_true(len(outputs) == 1)
        return "indexing is not supported for the time being"

    elif isinstance(node, intermediate.MatMul):  # constraints for matrix multiplication
        return "matrix multiplication is not supported for the time being"

    else:  # pragma: no cover
        assert_not_reached("Non IntermediateNode object in the OPGraph")

    if is_output:
        for out in outputs:
            if not value_is_scalar(out) or not value_is_unsigned_integer(out):
                return "only scalar unsigned integer outputs are supported"
    else:
        for out in outputs:
            # We currently can't fail on the following assert, but let it for possible changes in
            # the future
            if not value_is_integer(out):
                return "only integer intermediates are supported"  # pragma: no cover

    # pylint: enable=too-many-branches,too-many-return-statements

    return None


def check_graph_values_compatibility_with_mlir(
    op_graph: OPGraph,
) -> Optional[Dict[IntermediateNode, List[str]]]:
    """Make sure the graph outputs are unsigned integers, which is what the compiler supports.

    Args:
        op_graph: computation graph to check

    Returns:
        Dict[IntermediateNode, str]: None if the graph is compatible
            information about offending nodes otherwise
    """

    offending_nodes = {}

    for node in op_graph.graph.nodes:
        is_output = node in op_graph.output_nodes.values()
        if (reason := check_node_compatibility_with_mlir(node, is_output)) is not None:
            offending_nodes[node] = [reason]

    return None if len(offending_nodes) == 0 else offending_nodes


def _set_all_bit_width(op_graph: OPGraph, p: int):
    """Set all bit_width in the graph to `p` and `p+1` for clear and encrypted values respectively.

    Args:
        op_graph: graph to set bit_width for
        p: bit_width to set everywhere
    """
    for node in op_graph.graph.nodes:
        for value in node.outputs + node.inputs:
            if value_is_clear_scalar_integer(value) or value_is_clear_tensor_integer(value):
                value.dtype.bit_width = p + 1
            elif value_is_encrypted_scalar_integer(value) or value_is_encrypted_tensor_integer(
                value
            ):
                value.dtype.bit_width = p


def update_bit_width_for_mlir(op_graph: OPGraph):
    """Prepare bit_width of all nodes to be the same, set to the maximum value in the graph.

    Args:
        op_graph: graph to update bit_width for
    """
    max_bit_width = 0
    offending_nodes = {}
    for node in op_graph.graph.nodes:
        for value_out in node.outputs:
            if value_is_clear_scalar_integer(value_out) or value_is_clear_tensor_integer(value_out):
                current_node_out_bit_width = value_out.dtype.bit_width - 1
            else:

                assert_true(
                    value_is_encrypted_scalar_integer(value_out)
                    or value_is_encrypted_tensor_integer(value_out)
                )

                current_node_out_bit_width = value_out.dtype.bit_width

            max_bit_width = max(max_bit_width, current_node_out_bit_width)

            # Check that current_node_out_bit_width is supported by the compiler
            if current_node_out_bit_width > ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB:
                offending_nodes[node] = [
                    f"{current_node_out_bit_width} bits is not supported for the time being"
                ]

    if len(offending_nodes) != 0:
        raise RuntimeError(
            f"max_bit_width of some nodes is too high for the current version of "
            f"the compiler (maximum must be {ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB}) "
            f"which is not compatible with:\n"
            + get_printable_graph(op_graph, show_data_types=True, highlighted_nodes=offending_nodes)
        )

    _set_all_bit_width(op_graph, max_bit_width)


def extend_direct_lookup_tables(op_graph: OPGraph):
    """Extend direct lookup tables to the maximum length the input bit width can support.

    Args:
        op_graph: graph to update lookup tables for
    """
    for node in op_graph.graph.nodes:
        if isinstance(node, UnivariateFunction) and node.op_name == "TLU":
            table = node.op_kwargs["table"]
            bit_width = cast(Integer, node.inputs[0].dtype).bit_width
            expected_length = 2 ** bit_width

            # TODO: remove no cover once the table length workaround is removed
            # (https://github.com/zama-ai/concretefhe-internal/issues/359)
            if len(table) > expected_length:  # pragma: no cover
                node.op_kwargs["table"] = table[:expected_length]
            else:
                repeat = expected_length // len(table)
                node.op_kwargs["table"] = (table * repeat)[:expected_length]
