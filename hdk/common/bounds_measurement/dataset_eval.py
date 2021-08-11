"""Code to evaluate the IR graph on datasets."""

from typing import Any, Iterator, Tuple

from ..operator_graph import OPGraph


def eval_op_graph_bounds_on_dataset(op_graph: OPGraph, dataset: Iterator[Tuple[Any, ...]]):
    """Evaluate the bounds with a dataset.

    Evaluate the bounds for all output values of the operators in the graph op_graph over data
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

    def check_dataset_input_is_valid(data_to_check):
        assert len(data_to_check) == len(op_graph.input_nodes), (
            f"Got input data from dataset of len: {len(data_to_check)}, "
            f"function being evaluated has {len(op_graph.input_nodes)} inputs, please make "
            f"sure your data generator returns valid tuples of input values"
        )
        # TODO: change this to be more generic and check coherence between the input data type and
        # the corresponding Input ir node expected data type
        assert all(
            isinstance(val, int) for val in data_to_check
        ), "For now dataset evaluation only support int as inputs, please check your dataset"

    first_input_data = dict(enumerate(next(dataset)))
    check_dataset_input_is_valid(first_input_data.values())
    first_output = op_graph.evaluate(first_input_data)

    node_bounds = {
        node: {"min": first_output[node], "max": first_output[node]}
        for node in op_graph.graph.nodes()
    }

    for input_data in dataset:
        current_input_data = dict(enumerate(input_data))
        check_dataset_input_is_valid(current_input_data.values())
        current_output = op_graph.evaluate(current_input_data)
        for node, value in current_output.items():
            node_bounds[node]["min"] = min(node_bounds[node]["min"], value)
            node_bounds[node]["max"] = max(node_bounds[node]["max"], value)

    return node_bounds
