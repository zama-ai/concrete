"""Code to evaluate the IR graph on datasets."""

from typing import Any, Callable, Dict, Iterator, Tuple

from ..debugging import custom_assert
from ..operator_graph import OPGraph
from ..representation.intermediate import IntermediateNode


def eval_op_graph_bounds_on_dataset(
    op_graph: OPGraph,
    dataset: Iterator[Tuple[Any, ...]],
    min_func: Callable[[Any, Any], Any] = min,
    max_func: Callable[[Any, Any], Any] = max,
) -> Dict[IntermediateNode, Dict[str, Any]]:
    """Evaluate the bounds with a dataset.

    Evaluate the bounds for all output values of the operators in the graph op_graph over data
        coming from the dataset

    Args:
        op_graph (OPGraph): The graph for which we want to determine the bounds
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        min_func (Callable[[Any, Any], Any], optional): custom function to compute a scalar minimum
            between two values that can be encountered during evaluation (for e.g. numpy or torch
            tensors). Defaults to min.
        max_func (Callable[[Any, Any], Any], optional): custom function to compute a scalar maximum
            between two values that can be encountered during evaluation (for e.g. numpy or torch
            tensors). Defaults to max.

    Returns:
        Dict[IntermediateNode, Dict[str, Any]]: dict containing the bounds for each node from
            op_graph, stored with the node as key and a dict with keys "min" and "max" as value.
    """

    def check_dataset_input_len_is_valid(data_to_check):
        custom_assert(
            len(data_to_check) == len(op_graph.input_nodes),
            (
                f"Got input data from dataset of len: {len(data_to_check)}, "
                f"function being evaluated has {len(op_graph.input_nodes)} inputs, please make "
                f"sure your data generator returns valid tuples of input values"
            ),
        )

    # TODO: do we want to check coherence between the input data type and the corresponding Input ir
    # node expected data type ? Not considering bit_width as they may not make sense at this stage

    first_input_data = dict(enumerate(next(dataset)))
    check_dataset_input_len_is_valid(first_input_data.values())
    first_output = op_graph.evaluate(first_input_data)

    # We evaluate the min and max func to be able to resolve the tensors min and max rather than
    # having the tensor itself as the stored min and max values.
    node_bounds = {
        node: {"min": min_func(value, value), "max": max_func(value, value)}
        for node, value in first_output.items()
    }

    for input_data in dataset:
        current_input_data = dict(enumerate(input_data))
        check_dataset_input_len_is_valid(current_input_data.values())
        current_output = op_graph.evaluate(current_input_data)
        for node, value in current_output.items():
            node_bounds[node]["min"] = min_func(node_bounds[node]["min"], value)
            node_bounds[node]["max"] = max_func(node_bounds[node]["max"], value)

    return node_bounds
