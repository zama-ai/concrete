"""Utilities for MLIR conversion."""
from typing import Dict, List, Optional, cast

import networkx as nx

from ..data_types import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_clear_tensor_integer,
    value_is_encrypted_scalar_integer,
    value_is_encrypted_tensor_integer,
    value_is_integer,
    value_is_unsigned_integer,
)
from ..debugging import format_operation_graph
from ..debugging.custom_assert import assert_not_reached, assert_true
from ..operator_graph import OPGraph
from ..representation import intermediate
from ..representation.intermediate import GenericFunction, IntermediateNode

# TODO: should come from compiler, through an API, #402
ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB = 7


def check_node_compatibility_with_mlir(
    node: IntermediateNode,
    nx_graph: nx.MultiDiGraph,
    is_output: bool,
) -> Optional[str]:
    """Check if node is compatible with MLIR.

    Args:
        node (IntermediateNode): node to check
        nx_graph (nx.MultiDiGraph): the networkx graph to which node belongs
        is_output (bool): whether the node is an output node or not

    Returns:
        Optional[str]: None if the node is compatible else reason for incompatibility
    """

    # pylint: disable=too-many-branches,too-many-return-statements

    inputs = node.inputs
    outputs = node.outputs

    if isinstance(node, intermediate.Add):  # constraints for addition
        for inp in inputs:
            if not value_is_integer(inp):
                return "only integer addition is supported"

    elif isinstance(node, intermediate.Sub):  # constraints for subtraction
        for inp in inputs:
            if not value_is_integer(inp):
                return "only integer subtraction is supported"

    elif isinstance(node, intermediate.Mul):  # constraints for multiplication
        for inp in inputs:
            if not value_is_integer(inp):
                return "only integer multiplication is supported"

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

    elif isinstance(node, intermediate.GenericFunction):  # constraints for univariate functions
        for inp in inputs:
            if not value_is_integer(inp):
                return (
                    f"{node.op_name} with floating-point inputs "
                    f"is required to be fused to be supported"
                )

        if node.op_kind == "TLU":
            assert_true(
                len(
                    [
                        pred_node
                        for pred_node in nx_graph.pred[node]
                        if not isinstance(pred_node, intermediate.Constant)
                    ]
                )
                == 1
            )

            if not value_is_unsigned_integer(inputs[0]):
                # this branch is not reachable because compilation fails during inputset evaluation
                if node.op_name == "TLU":  # pragma: no cover
                    return "only unsigned integer lookup tables are supported"

                if node.op_name == "MultiTLU":  # pragma: no cover
                    return "only unsigned integer multi lookup tables are supported"

                # e.g., `np.absolute is not supported for the time being`
                return f"{node.op_name} is not supported for the time being"
        else:
            return f"{node.op_name} is not supported for the time being"

    elif isinstance(node, intermediate.Dot):  # constraints for dot product
        assert_true(len(inputs) == 2)
        if not value_is_unsigned_integer(inputs[0]) or not value_is_unsigned_integer(inputs[1]):
            return "only unsigned integer dot product is supported"

    elif isinstance(node, intermediate.IndexConstant):  # constraints for constant indexing
        assert_true(len(outputs) == 1)
        return "indexing is not supported for the time being"

    elif isinstance(node, intermediate.MatMul):  # constraints for matrix multiplication
        assert_true(len(inputs) == 2)
        if not value_is_unsigned_integer(inputs[0]) or not value_is_unsigned_integer(inputs[1]):
            return "only unsigned integer matrix multiplication is supported"

    else:  # pragma: no cover
        assert_not_reached("Non IntermediateNode object in the OPGraph")

    if is_output:
        for out in outputs:
            # For signed values and waiting for a real fix (#845): what is returned by the compiler
            # is not the (possibly negative) result r, but the always-positive (r mod 2**t), where t
            # is the bitwidth of r

            # We currently can't fail on the following assert, but let it for possible changes in
            # the future
            if not value_is_integer(out):
                return "only integer outputs are supported"  # pragma: no cover
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
        if (
            reason := check_node_compatibility_with_mlir(node, op_graph.graph, is_output)
        ) is not None:
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
            f"which is not compatible with:\n\n"
            + format_operation_graph(op_graph, highlighted_nodes=offending_nodes)
        )

    _set_all_bit_width(op_graph, max_bit_width)


def extend_direct_lookup_tables(op_graph: OPGraph):
    """Extend direct lookup tables to the maximum length the input bit width can support.

    Args:
        op_graph: graph to update lookup tables for
    """
    for node in op_graph.graph.nodes:
        if isinstance(node, GenericFunction) and node.op_name == "TLU":
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
