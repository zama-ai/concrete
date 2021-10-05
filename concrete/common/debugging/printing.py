"""functions to print the different graphs we can generate in the package, eg to debug."""

from typing import Any, Dict

import networkx as nx

from ..debugging.custom_assert import custom_assert
from ..operator_graph import OPGraph
from ..representation.intermediate import ArbitraryFunction, Constant, Input


def output_data_type_to_string(node):
    """Return the datatypes of the outputs of the node.

    Args:
        node: a graph node

    Returns:
        str: a string representing the datatypes of the outputs of the node

    """
    return ", ".join([str(o) for o in node.outputs])


def get_printable_graph(opgraph: OPGraph, show_data_types: bool = False) -> str:
    """Return a string representing a graph.

    Args:
        opgraph (OPGraph): The graph that we want to draw
        show_data_types (bool): Whether or not showing data_types of nodes, eg
            to see their width

    Returns:
        str: a string to print or save in a file
    """
    custom_assert(isinstance(opgraph, OPGraph))
    list_of_nodes_which_are_outputs = list(opgraph.output_nodes.values())
    graph = opgraph.graph

    returned_str = ""

    i = 0
    map_table: Dict[Any, int] = {}

    for node in nx.topological_sort(graph):

        # This code doesn't work with more than a single output. For more outputs,
        # we would need to change the way the destination are created: currently,
        # they only are done by incrementing i
        custom_assert(len(node.outputs) == 1)

        if isinstance(node, Input):
            what_to_print = node.input_name
        elif isinstance(node, Constant):
            content = str(node.constant_data).replace("\n", "")
            # if content is longer than 25 chars, only show the first and the last 10 chars of it
            # 25 is selected using the spaces available before data type information
            to_show = f"{content[:10]} ... {content[-10:]}" if len(content) > 25 else content
            what_to_print = f"Constant({to_show})"
        else:

            base_name = node.__class__.__name__

            if isinstance(node, ArbitraryFunction):
                base_name = node.op_name

            what_to_print = base_name + "("

            # Find all the names of the current predecessors of the node
            list_of_arg_name = []

            for pred, index_list in graph.pred[node].items():
                for index in index_list.values():
                    # Remark that we keep the index of the predecessor and its
                    # name, to print sources in the right order, which is
                    # important for eg non commutative operations
                    list_of_arg_name += [(index["input_idx"], str(map_table[pred]))]

            # Some checks, because the previous algorithm is not clear
            custom_assert(len(list_of_arg_name) == len(set(x[0] for x in list_of_arg_name)))
            list_of_arg_name.sort()
            custom_assert([x[0] for x in list_of_arg_name] == list(range(len(list_of_arg_name))))

            # Then, just print the predecessors in the right order
            what_to_print += ", ".join([x[1] for x in list_of_arg_name]) + ")"

        # This code doesn't work with more than a single output
        new_line = f"%{i} = {what_to_print}"

        # Manage datatypes
        if show_data_types:
            new_line = f"{new_line: <40s} # {output_data_type_to_string(node)}"

        returned_str += f"{new_line}\n"

        map_table[node] = i
        i += 1

    return_part = ", ".join(["%" + str(map_table[n]) for n in list_of_nodes_which_are_outputs])
    returned_str += f"return({return_part})\n"

    return returned_str
