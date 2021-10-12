"""File to hold some helper code."""

from typing import List, Optional

from .data_types.integers import Integer
from .debugging import assert_true
from .operator_graph import OPGraph
from .representation.intermediate import IntermediateNode


def is_a_power_of_2(x: int) -> bool:
    """Check if an integer is a power of two.

    Args:
        x (int): Number to check

    Returns:
        bool: True if the number is a power of two
    """
    # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two

    return x > 0 and (x & (x - 1)) == 0


def ir_nodes_has_integer_input_and_output(node: IntermediateNode) -> bool:
    """Check if an ir node has Integer inputs and outputs.

    Args:
        node (IntermediateNode): Node to check

    Returns:
        bool: True if all input and output values hold Integers
    """
    return all(isinstance(x.dtype, Integer) for x in node.inputs) and all(
        isinstance(x.dtype, Integer) for x in node.outputs
    )


# This check makes sense as long as the compiler backend only manages integers, to be removed in the
# long run probably
def check_op_graph_is_integer_program(
    op_graph: OPGraph,
    offending_nodes_out: Optional[List[IntermediateNode]] = None,
) -> bool:
    """Check if an op_graph inputs, outputs and intermediate values are Integers.

    Args:
        op_graph (OPGraph): The OPGraph to check
        offending_nodes_out (Optional[List[IntermediateNode]]): Optionally pass a list that will
            be populated with offending nodes, the list will be cleared before being filled

    Returns:
        bool: True if inputs, outputs and intermediate values are Integers, False otherwise
    """
    offending_nodes = [] if offending_nodes_out is None else offending_nodes_out

    assert_true(
        isinstance(offending_nodes, list),
        f"offending_nodes_out must be a list, got {type(offending_nodes_out)}",
    )

    offending_nodes.clear()
    offending_nodes.extend(
        node for node in op_graph.graph.nodes() if not ir_nodes_has_integer_input_and_output(node)
    )

    return len(offending_nodes) == 0
