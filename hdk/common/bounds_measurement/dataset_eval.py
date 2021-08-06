"""Code to evaluate the IR graph on datasets"""

from typing import Any, Iterator, Tuple

from ..operator_graph import OPGraph


def eval_op_graph_bounds_on_dataset(op_graph: OPGraph, dataset: Iterator[Tuple[Any, ...]]):
    """Evaluate the bounds for all output values of the operators in the graph op_graph over data
        coming from the dataset

    Args:
        op_graph (OPGraph): The graph for which we want to determine the bounds
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters

    Returns:
        Dict: dict containing the bounds for each node from op_graph, stored with the node as key
            and a dict with keys "min" and "max" as value
    """
    first_input_data = dict(enumerate(next(dataset)))

    # Check the dataset is well-formed
    assert len(first_input_data) == len(op_graph.input_nodes), (
        f"Got input data from dataset of len: {len(first_input_data)}, function being evaluated has"
        f" only {len(op_graph.input_nodes)} inputs, please make sure your data generator returns"
        f" valid tuples of input values"
    )

    first_output = op_graph.evaluate(first_input_data)

    node_bounds = {
        node: {"min": first_output[node], "max": first_output[node]}
        for node in op_graph.graph.nodes()
    }

    for input_data in dataset:

        next_input_data = dict(enumerate(input_data))

        # Check the dataset is well-formed
        assert len(next_input_data) == len(op_graph.input_nodes), (
            f"Got input data from dataset of len: {len(next_input_data)},"
            f" function being evaluated has"
            f" only {len(op_graph.input_nodes)} inputs, please make sure"
            f" your data generator returns"
            f" valid tuples of input values"
        )

        current_output = op_graph.evaluate(next_input_data)
        for node, value in current_output.items():
            node_bounds[node]["min"] = min(node_bounds[node]["min"], value)
            node_bounds[node]["max"] = max(node_bounds[node]["max"], value)

    return node_bounds
