"""Code to evaluate the IR graph on inputsets."""

import sys
from typing import Any, Callable, Dict, Iterable, Tuple

from ..compilation import CompilationConfiguration
from ..data_types.dtypes_helpers import (
    get_base_value_for_python_constant_data,
    is_data_type_compatible_with,
)
from ..debugging import custom_assert
from ..operator_graph import OPGraph
from ..representation.intermediate import IntermediateNode


def _check_input_coherency(
    input_to_check: Dict[str, Any],
    parameters: Dict[str, Any],
    get_base_value_for_constant_data_func: Callable[[Any], Any],
):
    """Check whether `input_to_check` is coherent with `parameters`.

    This function works by iterating over each constant of the input,
    determining base value of the constant using `get_base_value_for_constant_data_func` and
    checking if the base value of the contant is compatible with the base value of the parameter.

    Args:
        input_to_check (Dict[str, Any]): input to check coherency of
        parameters (Dict[str, Any]): parameters and their expected base values
        get_base_value_for_constant_data_func (Callable[[Any], Any]):
            function to get the base value of python objects.

    Returns:
        List[str]: List of warnings about the coherency
    """

    warnings = []
    for parameter_name, value in input_to_check.items():
        parameter_base_value = parameters[parameter_name]

        base_value_class = get_base_value_for_constant_data_func(value)
        base_value = base_value_class(is_encrypted=parameter_base_value.is_encrypted)

        if base_value.shape != parameter_base_value.shape or not is_data_type_compatible_with(
            base_value.data_type, parameter_base_value.data_type
        ):
            warnings.append(
                f"expected {str(parameter_base_value)} "
                f"for parameter `{parameter_name}` "
                f"but got {str(base_value)} "
                f"which is not compatible"
            )
    return warnings


def _print_input_coherency_warnings(
    current_input_index: int,
    current_input_data: Dict[int, Any],
    parameters: Dict[str, Any],
    parameter_index_to_parameter_name: Dict[int, str],
    get_base_value_for_constant_data_func: Callable[[Any], Any],
):
    """Print coherency warning for `input_to_check` against `parameters`.

    Args:
        current_input_index (int): index of the current input on the inputset
        current_input_data (Dict[int, Any]): input to print coherency warnings of
        parameters (Dict[str, Any]): parameters and their expected base values
        parameter_index_to_parameter_name (Dict[int, str]):
            dict to get parameter names from parameter indices
        get_base_value_for_constant_data_func (Callable[[Any], Any]):
            function to get the base value of python objects.

    Returns:
        None
    """

    current_input_named_data = {
        parameter_index_to_parameter_name[index]: data for index, data in current_input_data.items()
    }

    problems = _check_input_coherency(
        current_input_named_data,
        parameters,
        get_base_value_for_constant_data_func,
    )
    for problem in problems:
        sys.stderr.write(
            f"Warning: Input #{current_input_index} (0-indexed) "
            f"is not coherent with the hinted parameters ({problem})\n",
        )


def eval_op_graph_bounds_on_inputset(
    op_graph: OPGraph,
    inputset: Iterable[Tuple[Any, ...]],
    compilation_configuration: CompilationConfiguration,
    min_func: Callable[[Any, Any], Any] = min,
    max_func: Callable[[Any, Any], Any] = max,
    get_base_value_for_constant_data_func: Callable[
        [Any], Any
    ] = get_base_value_for_python_constant_data,
) -> Tuple[int, Dict[IntermediateNode, Dict[str, Any]]]:
    """Evaluate the bounds with a inputset.

    Evaluate the bounds for all output values of the operators in the graph op_graph over data
        coming from the inputset

    Args:
        op_graph (OPGraph): The graph for which we want to determine the bounds
        inputset (Iterable[Tuple[Any, ...]]): The inputset over which op_graph is evaluated. It
            needs to be an iterable on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during determining input checking strategy
        min_func (Callable[[Any, Any], Any], optional): custom function to compute a scalar minimum
            between two values that can be encountered during evaluation (for e.g. numpy or torch
            tensors). Defaults to min.
        max_func (Callable[[Any, Any], Any], optional): custom function to compute a scalar maximum
            between two values that can be encountered during evaluation (for e.g. numpy or torch
            tensors). Defaults to max.
        get_base_value_for_constant_data_func (Callable[[Any], Any], optional): custom function
            to compute the base value of a python object.

    Returns:
        Tuple[int, Dict[IntermediateNode, Dict[str, Any]]]: number of inputs in the inputset and
            a dict containing the bounds for each node from op_graph, stored with the node
            as key and a dict with keys "min" and "max" as value.
    """

    def check_inputset_input_len_is_valid(data_to_check):
        custom_assert(
            len(data_to_check) == len(op_graph.input_nodes),
            (
                f"Got input data from inputset of len: {len(data_to_check)}, "
                f"function being evaluated has {len(op_graph.input_nodes)} inputs, please make "
                f"sure your data generator returns valid tuples of input values"
            ),
        )

    # TODO: do we want to check coherence between the input data type and the corresponding Input ir
    # node expected data type ? Not considering bit_width as they may not make sense at this stage

    parameter_index_to_parameter_name = {
        index: input_node.input_name for index, input_node in op_graph.input_nodes.items()
    }
    parameters = {
        input_node.input_name: input_node.inputs[0] for input_node in op_graph.input_nodes.values()
    }

    inputset_iterator = iter(inputset)
    inputset_size = 0

    current_input_data = dict(enumerate(next(inputset_iterator)))
    inputset_size += 1

    check_inputset_input_len_is_valid(current_input_data.values())
    _print_input_coherency_warnings(
        inputset_size - 1,
        current_input_data,
        parameters,
        parameter_index_to_parameter_name,
        get_base_value_for_constant_data_func,
    )

    first_output = op_graph.evaluate(current_input_data)

    # We evaluate the min and max func to be able to resolve the tensors min and max rather than
    # having the tensor itself as the stored min and max values.
    node_bounds = {
        node: {"min": min_func(value, value), "max": max_func(value, value)}
        for node, value in first_output.items()
    }

    for input_data in inputset_iterator:
        inputset_size += 1
        current_input_data = dict(enumerate(input_data))

        check_inputset_input_len_is_valid(current_input_data.values())
        if compilation_configuration.check_every_input_in_inputset:
            _print_input_coherency_warnings(
                inputset_size - 1,
                current_input_data,
                parameters,
                parameter_index_to_parameter_name,
                get_base_value_for_constant_data_func,
            )

        current_output = op_graph.evaluate(current_input_data)
        for node, value in current_output.items():
            node_bounds[node]["min"] = min_func(node_bounds[node]["min"], value)
            node_bounds[node]["max"] = max_func(node_bounds[node]["max"], value)

    return inputset_size, node_bounds
