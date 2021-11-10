"""Functions to format operation graphs for debugging purposes."""

from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..debugging.custom_assert import assert_true
from ..operator_graph import OPGraph
from ..representation.intermediate import IntermediateNode


def format_operation_graph(
    opgraph: OPGraph,
    maximum_constant_length: int = 25,
    highlighted_nodes: Optional[Dict[IntermediateNode, List[str]]] = None,
) -> str:
    """Format an operation graph.

    Args:
        opgraph (OPGraph):
            the operation graph to format

        maximum_constant_length (int):
            maximum length of the constant throughout the formatting

        highlighted_nodes (Optional[Dict[IntermediateNode, List[str]]] = None):
            the dict of nodes and their corresponding messages which will be highlighted

    Returns:
        str: formatted operation graph
    """
    assert_true(isinstance(opgraph, OPGraph))

    # (node, output_index) -> identifier
    # e.g., id_map[(node1, 0)] = 2 and id_map[(node1, 1)] = 3
    # means line for node1 is in this form (%2, %3) = node1.format(...)
    id_map: Dict[Tuple[IntermediateNode, int], int] = {}

    # lines that will be merged at the end
    lines: List[str] = []

    # type information to add to each line (for alingment, this is done after lines are determined)
    type_informations: List[str] = []

    # default highlighted nodes is empty
    highlighted_nodes = highlighted_nodes if highlighted_nodes is not None else {}

    # highlight information for lines, this is required because highlights are added to lines
    # after their type information is added and we only have line numbers, not nodes
    highlighted_lines: Dict[int, List[str]] = {}

    for node in nx.topological_sort(opgraph.graph):
        # assign a unique id to outputs of node
        assert_true(len(node.outputs) > 0)
        for i in range(len(node.outputs)):
            id_map[(node, i)] = len(id_map)

        # remember highlights of the node
        if node in highlighted_nodes:
            highlighted_lines[len(lines)] = highlighted_nodes[node]

        # extract predecessors and their ids
        predecessors = []
        for predecessor, output_idx in opgraph.get_ordered_inputs_of(node):
            predecessors.append(f"%{id_map[(predecessor, output_idx)]}")

        # start the build the line for the node
        line = ""

        # add output information to the line
        outputs = ", ".join(f"%{id_map[(node, i)]}" for i in range(len(node.outputs)))
        line += outputs if len(node.outputs) == 1 else f"({outputs})"

        # add node information to the line
        line += " = "
        line += node.text_for_formatting(predecessors, maximum_constant_length)

        # append line to list of lines
        lines.append(line)

        # remember type information of the node
        types = ", ".join(str(output) for output in node.outputs)
        type_informations.append(types if len(node.outputs) == 1 else f"({types})")

    # align = signs
    #
    # e.g.,
    #
    #  %1 = ...
    #  %2 = ...
    #  ...
    #  %8 = ...
    #  %9 = ...
    # %10 = ...
    # %11 = ...
    # ...
    longest_length_before_equals_sign = max(len(line.split("=")[0]) for line in lines)
    for i, line in enumerate(lines):
        length_before_equals_sign = len(line.split("=")[0])
        lines[i] = (" " * (longest_length_before_equals_sign - length_before_equals_sign)) + line

    # add type informations
    longest_line_length = max(len(line) for line in lines)
    for i, line in enumerate(lines):
        lines[i] += " " * (longest_line_length - len(line))
        lines[i] += f"        # {type_informations[i]}"

    # add highlights (this is done in reverse to keep indices consistent)
    for i in reversed(range(len(lines))):
        if i in highlighted_lines:
            for j, message in enumerate(highlighted_lines[i]):
                highlight = "^" if j == 0 else " "
                lines.insert(i + 1 + j, f"{highlight * len(lines[i])} {message}")

    # add return information
    # (if there is a single return, it's in the form `return %id`
    # (otherwise, it's in the form `return (%id1, %id2, ..., %idN)`
    returns: List[str] = []
    for node in opgraph.output_nodes.values():
        outputs = ", ".join(f"%{id_map[(node, i)]}" for i in range(len(node.outputs)))
        returns.append(outputs if len(node.outputs) == 1 else f"({outputs})")
    lines.append("return " + returns[0] if len(returns) == 1 else f"({', '.join(returns)})")

    return "\n".join(lines)
