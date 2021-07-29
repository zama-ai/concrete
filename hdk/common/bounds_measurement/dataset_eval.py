"""Code to evaluate the IR graph on datasets"""

from typing import Iterator

from ..operator_graph import OPGraph


def eval_op_graph_bounds_on_dataset(op_graph: OPGraph, data_generator: Iterator):
    """Evaluate the bounds for all output values of the operators in the graph op_graph over data
        coming from the data_generator

    Args:
        op_graph (OPGraph): The graph for which we want to determine the bounds
        data_generator (Iterator): The dataset over which op_graph is evaluated

    Returns:
        Dict: dict containing the bounds for each node from op_graph, stored with the node as key
            and a dict with keys "min" and "max" as value
    """
    first_input_data = dict(enumerate(next(data_generator)))
    first_output = op_graph.evaluate(first_input_data)

    node_bounds = {
        node: {"min": first_output[node], "max": first_output[node]}
        for node in op_graph.graph.nodes()
    }

    for input_data in data_generator:
        current_output = op_graph.evaluate(dict(enumerate(input_data)))
        for node, value in current_output.items():
            node_bounds[node]["min"] = min(node_bounds[node]["min"], value)
            node_bounds[node]["max"] = max(node_bounds[node]["max"], value)

    return node_bounds
