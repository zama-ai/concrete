"""Code to evaluate the IR graph on inputsets."""

import sys
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from ..compilation import CompilationConfiguration
from ..data_types.dtypes_helpers import (
    get_base_value_for_python_constant_data,
    is_data_type_compatible_with,
)
from ..debugging import assert_true
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
            base_value.dtype, parameter_base_value.dtype
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
    treat_warnings_as_errors: bool,
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
    messages = [
        (
            f"Input #{current_input_index} (0-indexed) "
            f"is not coherent with the hinted parameters ({problem})\n"
        )
        for problem in problems
    ]

    if len(messages) > 0:
        if treat_warnings_as_errors:
            raise ValueError(", ".join(messages))

        for message in messages:
            sys.stderr.write(f"Warning: {message}")


def eval_op_graph_bounds_on_inputset(
    op_graph: OPGraph,
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]]],
    compilation_configuration: CompilationConfiguration,
    min_func: Callable[[Any, Any], Any] = min,
    max_func: Callable[[Any, Any], Any] = max,
    get_base_value_for_constant_data_func: Callable[
        [Any], Any
    ] = get_base_value_for_python_constant_data,
    prev_node_bounds_and_samples: Optional[Dict[IntermediateNode, Dict[str, Any]]] = None,
) -> Tuple[int, Dict[IntermediateNode, Dict[str, Any]]]:
    """Evaluate the bounds with a inputset.

    Evaluate the bounds for all output values of the operators in the graph op_graph over data
        coming from the inputset

    Args:
        op_graph (OPGraph): The graph for which we want to determine the bounds
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]]]): The inputset over which op_graph
            is evaluated. It needs to be an iterable on tuples (can be single values in case the
            function has only one argument) which are of the same length than the number of
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
        prev_node_bounds_and_samples (Optional[Dict[IntermediateNode, Dict[str, Any]]], optional):
            Bounds and samples from a previous run. Defaults to None.

    Returns:
        Tuple[int, Dict[IntermediateNode, Dict[str, Any]]]: number of inputs in the inputset and
            a dict containing the bounds for each node from op_graph, stored with the node
            as key and a dict with keys "min", "max" and "sample" as value.
    """

    num_input_nodes = len(op_graph.input_nodes)

    def check_inputset_input_len_is_valid(data_to_check):
        # Only check if there are more than one input node, otherwise accept the value as the sole
        # argument passed to the OPGraph for evaluation
        if num_input_nodes > 1:
            assert_true(
                len(data_to_check) == num_input_nodes,
                (
                    f"Got input data from inputset of len: {len(data_to_check)}, "
                    f"function being evaluated has {num_input_nodes} inputs, please make "
                    f"sure your data generator returns valid tuples of input values"
                ),
            )

    def generate_input_values_dict(input_data) -> Dict[int, Any]:
        if num_input_nodes > 1:
            return dict(enumerate(input_data))
        # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/772
        # update this to support tuple in case of 1-input functions accepting tuples
        assert_true(
            not isinstance(input_data, tuple),
            "Tuples are unsupported for single input inputset evaluation",
        )
        return {0: input_data}

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

    current_input_data = generate_input_values_dict(next(inputset_iterator))
    inputset_size += 1

    check_inputset_input_len_is_valid(current_input_data.values())
    _print_input_coherency_warnings(
        inputset_size - 1,
        current_input_data,
        parameters,
        parameter_index_to_parameter_name,
        get_base_value_for_constant_data_func,
        compilation_configuration.treat_warnings_as_errors,
    )

    first_output = op_graph.evaluate(current_input_data)

    prev_node_bounds_and_samples = (
        {} if prev_node_bounds_and_samples is None else prev_node_bounds_and_samples
    )

    def get_previous_value_for_key_or_default_for_dict(
        dict_: Dict[IntermediateNode, Dict[str, Any]],
        node: IntermediateNode,
        key: str,
        default: Any,
    ) -> Any:
        return_value = default

        previous_value_dict = dict_.get(node, None)

        if previous_value_dict is not None:
            return_value = previous_value_dict.get(key, default)

        return return_value

    get_previous_value_for_key_or_default = partial(
        get_previous_value_for_key_or_default_for_dict, prev_node_bounds_and_samples
    )

    # We evaluate the min and max func to be able to resolve the tensors min and max rather than
    # having the tensor itself as the stored min and max values.
    # As we don't know the integrity of prev_node_bounds_and_samples we make sure we can
    # populate the new node_bounds_and_samples
    node_bounds_and_samples = {
        node: {
            "min": min_func(value, get_previous_value_for_key_or_default(node, "min", value)),
            "max": max_func(value, get_previous_value_for_key_or_default(node, "max", value)),
            "sample": get_previous_value_for_key_or_default(node, "sample", value),
        }
        for node, value in first_output.items()
    }

    for input_data in inputset_iterator:
        inputset_size += 1
        current_input_data = generate_input_values_dict(input_data)

        check_inputset_input_len_is_valid(current_input_data.values())
        if compilation_configuration.check_every_input_in_inputset:
            _print_input_coherency_warnings(
                inputset_size - 1,
                current_input_data,
                parameters,
                parameter_index_to_parameter_name,
                get_base_value_for_constant_data_func,
                compilation_configuration.treat_warnings_as_errors,
            )

        current_output = op_graph.evaluate(current_input_data)
        for node, value in current_output.items():
            node_bounds_and_samples[node]["min"] = min_func(
                node_bounds_and_samples[node]["min"], value
            )
            node_bounds_and_samples[node]["max"] = max_func(
                node_bounds_and_samples[node]["max"], value
            )

    return inputset_size, node_bounds_and_samples
