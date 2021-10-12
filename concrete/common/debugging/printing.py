"""functions to print the different graphs we can generate in the package, eg to debug."""

from typing import Any, Dict

import networkx as nx

from ..debugging.custom_assert import assert_true
from ..operator_graph import OPGraph
from ..representation.intermediate import Constant, Input, UnivariateFunction


def output_data_type_to_string(node):
    """Return the datatypes of the outputs of the node.

    Args:
        node: a graph node

    Returns:
        str: a string representing the datatypes of the outputs of the node

    """
    return ", ".join([str(o) for o in node.outputs])


def shorten_a_constant(constant_data: str):
    """Return a constant (if small) or an extra of the constant (if too large).

    Args:
        constant (str): The constant we want to shorten

    Returns:
        str: a string to represent the constant
    """

    content = str(constant_data).replace("\n", "")
    # if content is longer than 25 chars, only show the first and the last 10 chars of it
    # 25 is selected using the spaces available before data type information
    short_content = f"{content[:10]} ... {content[-10:]}" if len(content) > 25 else content
    return short_content


def get_printable_graph(opgraph: OPGraph, show_data_types: bool = False) -> str:
    """Return a string representing a graph.

    Args:
        opgraph (OPGraph): The graph that we want to draw
        show_data_types (bool): Whether or not showing data_types of nodes, eg
            to see their width

    Returns:
        str: a string to print or save in a file
    """
    assert_true(isinstance(opgraph, OPGraph))
    list_of_nodes_which_are_outputs = list(opgraph.output_nodes.values())
    graph = opgraph.graph

    returned_str = ""

    i = 0
    map_table: Dict[Any, int] = {}

    for node in nx.topological_sort(graph):

        # TODO: #640
        # This code doesn't work with more than a single output. For more outputs,
        # we would need to change the way the destination are created: currently,
        # they only are done by incrementing i
        assert_true(len(node.outputs) == 1)

        if isinstance(node, Input):
            what_to_print = node.input_name
        elif isinstance(node, Constant):
            to_show = shorten_a_constant(node.constant_data)
            what_to_print = f"Constant({to_show})"
        else:

            base_name = node.__class__.__name__

            if isinstance(node, UnivariateFunction):
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
            assert_true(len(list_of_arg_name) == len(set(x[0] for x in list_of_arg_name)))
            list_of_arg_name.sort()
            assert_true([x[0] for x in list_of_arg_name] == list(range(len(list_of_arg_name))))

            prefix_to_add_to_what_to_print = ""
            suffix_to_add_to_what_to_print = ""

            # Print constant that may be in the UnivariateFunction. For the moment, it considers
            # there is a single constant maximally and that there is 2 inputs maximally
            if isinstance(node, UnivariateFunction) and "baked_constant" in node.op_kwargs:
                baked_constant = node.op_kwargs["baked_constant"]
                if node.op_attributes["in_which_input_is_constant"] == 0:
                    prefix_to_add_to_what_to_print = f"{shorten_a_constant(baked_constant)}, "
                else:
                    assert_true(
                        node.op_attributes["in_which_input_is_constant"] == 1,
                        "'in_which_input_is_constant' should be a key of node.op_attributes",
                    )
                    suffix_to_add_to_what_to_print = f", {shorten_a_constant(baked_constant)}"

            # Then, just print the predecessors in the right order
            what_to_print += prefix_to_add_to_what_to_print
            what_to_print += ", ".join(["%" + x[1] for x in list_of_arg_name])
            what_to_print += suffix_to_add_to_what_to_print
            what_to_print += ")"

        # This code doesn't work with more than a single output
        new_line = f"%{i} = {what_to_print}"

        # Manage datatypes
        if show_data_types:
            new_line = f"{new_line: <50s} # {output_data_type_to_string(node)}"

        returned_str += f"{new_line}\n"

        map_table[node] = i
        i += 1

    return_part = ", ".join(["%" + str(map_table[n]) for n in list_of_nodes_which_are_outputs])
    returned_str += f"return({return_part})\n"

    return returned_str
