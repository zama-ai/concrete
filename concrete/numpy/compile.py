"""numpy compilation function."""

import sys
import traceback
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, cast

import numpy
from concrete.compiler import CompilerEngine

from ..common.bounds_measurement.inputset_eval import eval_op_graph_bounds_on_inputset
from ..common.common_helpers import check_op_graph_is_integer_program
from ..common.compilation import CompilationArtifacts, CompilationConfiguration
from ..common.data_types import Integer
from ..common.debugging import format_operation_graph
from ..common.debugging.custom_assert import assert_true
from ..common.fhe_circuit import FHECircuit
from ..common.mlir.utils import (
    check_graph_values_compatibility_with_mlir,
    update_bit_width_for_mlir,
)
from ..common.operator_graph import OPGraph
from ..common.optimization.topological import fuse_float_operations
from ..common.representation.intermediate import Add, Constant, GenericFunction, IntermediateNode
from ..common.values import BaseValue, ClearScalar
from ..numpy.tracing import trace_numpy_function
from .np_dtypes_helpers import (
    get_base_data_type_for_numpy_or_python_constant_data,
    get_base_value_for_numpy_or_python_constant_data,
    get_constructor_for_numpy_or_python_constant_data,
)
from .np_inputset_helpers import _check_special_inputset_availability, _generate_random_inputset
from .np_mlir_converter import NPMLIRConverter

_COMPILE_FHE_INSECURE_KEY_CACHE_DIR: Optional[str] = None


def numpy_max_func(lhs: Any, rhs: Any) -> Any:
    """Compute the maximum value between two values which can be numpy classes (e.g. ndarray).

    Args:
        lhs (Any): lhs value to compute max from.
        rhs (Any): rhs value to compute max from.

    Returns:
        Any: maximum scalar value between lhs and rhs.
    """
    return numpy.maximum(lhs, rhs).max()


def numpy_min_func(lhs: Any, rhs: Any) -> Any:
    """Compute the minimum value between two values which can be numpy classes (e.g. ndarray).

    Args:
        lhs (Any): lhs value to compute min from.
        rhs (Any): rhs value to compute min from.

    Returns:
        Any: minimum scalar value between lhs and rhs.
    """
    return numpy.minimum(lhs, rhs).min()


def sanitize_compilation_configuration_and_artifacts(
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> Tuple[CompilationConfiguration, CompilationArtifacts]:
    """Return the proper compilation configuration and artifacts.

    Default values are returned if None is passed for each argument.

    Args:
        compilation_configuration (Optional[CompilationConfiguration], optional): the compilation
            configuration to sanitize. Defaults to None.
        compilation_artifacts (Optional[CompilationArtifacts], optional): the compilation artifacts
            to sanitize. Defaults to None.

    Returns:
        Tuple[CompilationConfiguration, CompilationArtifacts]: the tuple of sanitized configuration
            and artifacts.
    """
    # Create default configuration if custom configuration is not specified
    compilation_configuration = (
        CompilationConfiguration()
        if compilation_configuration is None
        else compilation_configuration
    )

    # Create temporary artifacts if custom artifacts is not specified (in case of exceptions)
    if compilation_artifacts is None:
        compilation_artifacts = CompilationArtifacts()

    return compilation_configuration, compilation_artifacts


def get_inputset_to_use(
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]], str],
    compilation_configuration: CompilationConfiguration,
) -> Union[Iterable[Any], Iterable[Tuple[Any, ...]]]:
    """Get the proper inputset to use for compilation.

    Args:
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]], str]): The inputset over which
            op_graph is evaluated. It needs to be an iterable on tuples which are of the same length
            than the number of parameters in the function, and in the same order than these same
            parameters
        compilation_configuration (CompilationConfiguration): Configuration object to use during
            compilation

    Returns:
        Union[Iterable[Any], Iterable[Tuple[Any, ...]]]: the inputset to use.
    """
    # Generate random inputset if it is requested and available
    if isinstance(inputset, str):
        _check_special_inputset_availability(inputset, compilation_configuration)
        return _generate_random_inputset(function_parameters, compilation_configuration)
    return inputset


def run_compilation_function_with_error_management(
    compilation_function: Callable,
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
) -> Any:
    """Call compilation_function() and manage exceptions that may occur.

    Args:
        compilation_function (Callable): the compilation function to call.
        compilation_configuration (CompilationConfiguration): the current compilation configuration.
        compilation_artifacts (CompilationArtifacts): the current compilation artifacts.

    Returns:
        Any: returns the result of the call to compilation_function
    """

    # Try to compile the function and save partial artifacts on failure
    try:
        # Use context manager to restore numpy error handling
        with numpy.errstate(**numpy.geterr()):
            return compilation_function()
    except Exception:  # pragma: no cover
        # This branch is reserved for unexpected issues and hence it shouldn't be tested.
        # If it could be tested, we would have fixed the underlying issue.

        # We need to export all the information we have about the compilation
        # If the user wants them to be exported

        if compilation_configuration.dump_artifacts_on_unexpected_failures:
            compilation_artifacts.export()

            traceback_path = compilation_artifacts.output_directory.joinpath("traceback.txt")
            with open(traceback_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

        raise


def _compile_numpy_function_into_op_graph_internal(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
) -> OPGraph:
    """Compile a function into an OPGraph without evaluating the intermediate nodes bounds.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation

    Returns:
        OPGraph: compiled function into a graph, node values are not representative of the values
            that can be observed during execution.
            Use _compile_numpy_function_into_op_graph_and_measure_bounds_internal if you need bounds
            estimation.
    """
    # Check function parameters
    wrong_inputs = {
        inp: function_parameters[inp]
        for inp in function_parameters.keys()
        if not isinstance(function_parameters[inp], BaseValue)
    }
    list_of_possible_basevalue = [
        "ClearTensor",
        "EncryptedTensor",
        "ClearScalar",
        "EncryptedScalar",
    ]
    assert_true(
        len(wrong_inputs.keys()) == 0,
        f"wrong type for inputs {wrong_inputs}, needs to be one of {list_of_possible_basevalue}",
    )

    # Add the function to compile as an artifact
    compilation_artifacts.add_function_to_compile(function_to_compile)

    # Add the parameters of function to compile as artifacts
    for name, value in function_parameters.items():
        compilation_artifacts.add_parameter_of_function_to_compile(name, str(value))

    # Trace the function
    op_graph = trace_numpy_function(function_to_compile, function_parameters)

    # Add the initial graph as an artifact
    compilation_artifacts.add_operation_graph("initial", op_graph)

    # Apply topological optimizations if they are enabled
    if compilation_configuration.enable_topological_optimizations:
        # Fuse float operations to have int to int GenericFunction
        if not check_op_graph_is_integer_program(op_graph):
            fuse_float_operations(op_graph, compilation_artifacts)

    return op_graph


def compile_numpy_function_into_op_graph(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation

    Returns:
        OPGraph: compiled function into a graph
    """

    (
        compilation_configuration,
        compilation_artifacts,
    ) = sanitize_compilation_configuration_and_artifacts(
        compilation_configuration, compilation_artifacts
    )

    def compilation_function():
        return _compile_numpy_function_into_op_graph_internal(
            function_to_compile,
            function_parameters,
            compilation_configuration,
            compilation_artifacts,
        )

    result = run_compilation_function_with_error_management(
        compilation_function, compilation_configuration, compilation_artifacts
    )

    # for mypy
    assert isinstance(result, OPGraph)
    return result


def _measure_op_graph_bounds_and_update_internal(
    op_graph: OPGraph,
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
    prev_node_bounds_and_samples: Optional[Dict[IntermediateNode, Dict[str, Any]]] = None,
    warn_on_inputset_length: bool = True,
) -> Dict[IntermediateNode, Dict[str, Any]]:
    """Measure the intermediate values and update the OPGraph accordingly for the given inputset.

    Args:
        op_graph (OPGraph): the OPGraph for which to measure bounds and update node values.
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]]]): The inputset over which op_graph
            is evaluated. It needs to be an iterable on tuples which are of the same length than the
            number of parameters in the function, and in the same order than these same parameters
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation
        prev_node_bounds_and_samples (Optional[Dict[IntermediateNode, Dict[str, Any]]], optional):
            Bounds and samples from a previous run. Defaults to None.
        warn_on_inputset_length (bool, optional): Set to True to get a warning if inputset is not
            long enough. Defaults to True.

    Raises:
        ValueError: Raises an error if the inputset is too small and the compilation configuration
            treats warnings as error.

    Returns:
        Dict[IntermediateNode, Dict[str, Any]]: a dict containing the bounds for each node from
            op_graph, stored with the node as key and a dict with keys "min", "max" and "sample" as
            value.
    """

    # Find bounds with the inputset
    inputset_size, node_bounds_and_samples = eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=compilation_configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
        prev_node_bounds_and_samples=prev_node_bounds_and_samples,
    )

    if warn_on_inputset_length:
        # Check inputset size
        inputset_size_upper_limit = 1

        # this loop will determine the number of possible inputs of the function
        # if a function have a single 3-bit input, for example, inputset_size_upper_limit will be 8
        for parameter_value in function_parameters.values():
            if isinstance(parameter_value.dtype, Integer):
                # multiple parameter bit-widths are multiplied as they can be combined into an input
                inputset_size_upper_limit *= 2 ** parameter_value.dtype.bit_width

                # if the upper limit of the inputset size goes above 10,
                # break the loop as we will require at least 10 inputs in this case
                if inputset_size_upper_limit > 10:
                    break

        minimum_required_inputset_size = min(inputset_size_upper_limit, 10)
        if inputset_size < minimum_required_inputset_size:
            message = (
                f"Provided inputset contains too few inputs "
                f"(it should have had at least {minimum_required_inputset_size} "
                f"but it only had {inputset_size})\n"
            )

            if compilation_configuration.treat_warnings_as_errors:
                raise ValueError(message)

            sys.stderr.write(f"Warning: {message}")

    # Add the bounds as an artifact
    compilation_artifacts.add_final_operation_graph_bounds(node_bounds_and_samples)

    # Update the graph accordingly: after that, we have the compilable graph
    op_graph.update_values_with_bounds_and_samples(
        node_bounds_and_samples,
        get_base_data_type_for_numpy_or_python_constant_data,
        get_constructor_for_numpy_or_python_constant_data,
    )

    return node_bounds_and_samples


def measure_op_graph_bounds_and_update(
    op_graph: OPGraph,
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]], str],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    prev_node_bounds_and_samples: Optional[Dict[IntermediateNode, Dict[str, Any]]] = None,
    warn_on_inputset_length: bool = True,
) -> Dict[IntermediateNode, Dict[str, Any]]:
    """Measure the intermediate values and update the OPGraph accordingly for the given inputset.

    Args:
        op_graph (OPGraph): the OPGraph for which to measure bounds and update node values.
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]], str]): The inputset over which
            op_graph is evaluated. It needs to be an iterable on tuples which are of the same length
            than the number of parameters in the function, and in the same order than these same
            parameters
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation
        prev_node_bounds_and_samples (Optional[Dict[IntermediateNode, Dict[str, Any]]], optional):
            Bounds and samples from a previous run. Defaults to None.
        warn_on_inputset_length (bool, optional): Set to True to get a warning if inputset is not
            long enough. Defaults to True.

    Raises:
        ValueError: Raises an error if the inputset is too small and the compilation configuration
            treats warnings as error.

    Returns:
        Dict[IntermediateNode, Dict[str, Any]]: a dict containing the bounds for each node from
            op_graph, stored with the node as key and a dict with keys "min", "max" and "sample" as
            value.
    """

    (
        compilation_configuration,
        compilation_artifacts,
    ) = sanitize_compilation_configuration_and_artifacts(
        compilation_configuration, compilation_artifacts
    )

    inputset = get_inputset_to_use(function_parameters, inputset, compilation_configuration)

    def compilation_function():
        return _measure_op_graph_bounds_and_update_internal(
            op_graph,
            function_parameters,
            inputset,
            compilation_configuration,
            compilation_artifacts,
            prev_node_bounds_and_samples,
            warn_on_inputset_length,
        )

    result = run_compilation_function_with_error_management(
        compilation_function, compilation_configuration, compilation_artifacts
    )

    # for mypy
    assert isinstance(result, dict)
    return result


def _compile_numpy_function_into_op_graph_and_measure_bounds_internal(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
) -> OPGraph:
    """Compile a function into an OPGraph and evaluate the intermediate nodes bounds.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]]]): The inputset over which op_graph
            is evaluated. It needs to be an iterable on tuples which are of the same length than the
            number of parameters in the function, and in the same order than these same parameters
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation

    Returns:
        OPGraph: compiled function into a graph with estimated bounds in node values.
    """

    op_graph = _compile_numpy_function_into_op_graph_internal(
        function_to_compile,
        function_parameters,
        compilation_configuration,
        compilation_artifacts,
    )

    _measure_op_graph_bounds_and_update_internal(
        op_graph,
        function_parameters,
        inputset,
        compilation_configuration,
        compilation_artifacts,
    )

    # Add the final graph as an artifact
    compilation_artifacts.add_operation_graph("final", op_graph)

    return op_graph


def compile_numpy_function_into_op_graph_and_measure_bounds(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]], str],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]], str]): The inputset over which
            op_graph is evaluated. It needs to be an iterable on tuples which are of the same length
            than the number of parameters in the function, and in the same order than these same
            parameters. Alternatively, it can be "random" but that's an unstable feature and should
            not be used in production.
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation

    Returns:
        OPGraph: compiled function into a graph
    """

    (
        compilation_configuration,
        compilation_artifacts,
    ) = sanitize_compilation_configuration_and_artifacts(
        compilation_configuration, compilation_artifacts
    )

    inputset = get_inputset_to_use(function_parameters, inputset, compilation_configuration)

    def compilation_function():
        return _compile_numpy_function_into_op_graph_and_measure_bounds_internal(
            function_to_compile,
            function_parameters,
            inputset,
            compilation_configuration,
            compilation_artifacts,
        )

    result = run_compilation_function_with_error_management(
        compilation_function, compilation_configuration, compilation_artifacts
    )

    # for mypy
    assert isinstance(result, OPGraph)
    return result


# HACK
# TODO: remove this ugly hack when
# https://github.com/zama-ai/concrete-numpy-internal/issues/1001 is done
# TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/1015
def hack_offset_negative_inputs_to_lookup_tables(op_graph: OPGraph) -> None:
    """Hack the op_graph to add offsets to signed inputs to TLUs.

    Args:
        op_graph (OPGraph): the OPGraph to hack.
    """
    # Ugly hack to add an offset before entering a TLU if its variable input node has a signed
    # output.
    # This is ugly as this makes hardcoded assumptions about the way bit widths are handled in MLIR.
    # This does not update the TLU input values to allow for proper table generation.
    # Thankfully we are not supposed to touch the op_graph beyond that point
    for node in list((nx_graph := op_graph.graph).nodes):
        if isinstance(node, GenericFunction) and node.op_kind == "TLU":
            ordered_preds_and_inputs = op_graph.get_ordered_preds_and_inputs_of(node)
            variable_input_indices = [
                idx
                for idx, (pred, _) in enumerate(ordered_preds_and_inputs)
                if not isinstance(pred, Constant)
            ]
            assert_true(len(variable_input_indices) == 1)
            variable_input_idx = variable_input_indices[0]
            variable_input_node = ordered_preds_and_inputs[variable_input_idx][0]
            variable_input_value = variable_input_node.outputs[0]
            variable_input_dtype = variable_input_value.dtype
            assert_true(isinstance(variable_input_dtype, Integer))
            variable_input_dtype = cast(Integer, variable_input_dtype)
            if not variable_input_dtype.is_signed:
                continue

            # input_bit_width + 1 to be MLIR compliant
            input_bit_width = variable_input_dtype.bit_width
            mlir_compliant_int_type = Integer(input_bit_width + 1, True)

            # Manually fix the output values to be MLIR compliant
            # offset_constant is set to abs(min_value) for the variable input so that the values
            # [- 2 ** (n - 1); 2 ** (n - 1) - 1] is mapped to [0; 2 ** n - 1], changing the signed
            # TLU to an actual unsigned TLU. The get_table function creates the table from the min
            # value to the max value. As we keep the input value as a signed value, it will be from
            # - 2 ** (n - 1) to 2 ** (n - 1) - 1. Then, the get_table function stores corresponding
            # values in increasing indexes from 0 to 2 ** n - 1. As our signed values have been
            # shifted by 2 ** (n - 1), the table will be usable as-is, without needing any change in
            # the lambda function of the GenericFunction.
            offset_constant = Constant(abs(variable_input_dtype.min_value()))
            offset_constant.outputs[0].dtype = deepcopy(mlir_compliant_int_type)
            add_offset = Add(
                [deepcopy(variable_input_value), ClearScalar(deepcopy(mlir_compliant_int_type))]
            )
            add_offset.outputs[0] = deepcopy(variable_input_value)

            nx_graph.remove_edge(variable_input_node, node)
            nx_graph.add_edge(variable_input_node, add_offset, input_idx=0, output_idx=0)
            nx_graph.add_edge(offset_constant, add_offset, input_idx=1, output_idx=0)
            nx_graph.add_edge(add_offset, node, input_idx=variable_input_idx, output_idx=0)


def prepare_op_graph_for_mlir(op_graph: OPGraph):
    """Prepare OPGraph for MLIR lowering.

    This includes checking compatibility, changing bit-widths, and modifying lookup tables.

    Args:
        op_graph (OPGraph): The operation graph to prepare

    Returns:
        None
    """

    # Make sure the graph can be lowered to MLIR
    offending_nodes = check_graph_values_compatibility_with_mlir(op_graph)
    if offending_nodes is not None:
        raise RuntimeError(
            "function you are trying to compile isn't supported for MLIR lowering\n\n"
            + format_operation_graph(op_graph, highlighted_nodes=offending_nodes)
        )

    # Update bit_width for MLIR
    update_bit_width_for_mlir(op_graph)

    # HACK
    # TODO: remove this ugly hack when
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1001 is done
    # TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/1015
    hack_offset_negative_inputs_to_lookup_tables(op_graph)


def _compile_op_graph_to_fhe_circuit_internal(
    op_graph: OPGraph,
    show_mlir: bool,
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
) -> FHECircuit:
    """Compile the OPGraph to an FHECircuit.

    Args:
        op_graph (OPGraph): the OPGraph to compile.
        show_mlir (bool): determine whether we print the mlir string.
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation

    Returns:
        FHECircuit: the compiled FHECircuit
    """
    prepare_op_graph_for_mlir(op_graph)

    # Convert graph to an MLIR representation
    converter = NPMLIRConverter()
    mlir_result = converter.convert(op_graph)

    # Show MLIR representation if requested
    if show_mlir:
        print(f"MLIR which is going to be compiled: \n{mlir_result}")

    # Add MLIR representation as an artifact
    compilation_artifacts.add_final_operation_graph_mlir(mlir_result)

    if _COMPILE_FHE_INSECURE_KEY_CACHE_DIR is not None and not (
        compilation_configuration.use_insecure_key_cache
        and compilation_configuration.enable_unsafe_features
    ):
        raise RuntimeError(
            f"Unable to use insecure key cache {_COMPILE_FHE_INSECURE_KEY_CACHE_DIR} "
            "as use_insecure_key_cache or enable_unsafe_features are not set to True in"
            "compilation_configuration"
        )

    # Compile the MLIR representation
    engine = CompilerEngine()
    engine.compile_fhe(mlir_result, unsecure_key_set_cache_path=_COMPILE_FHE_INSECURE_KEY_CACHE_DIR)

    return FHECircuit(op_graph, engine)


def compile_op_graph_to_fhe_circuit(
    op_graph: OPGraph,
    show_mlir: bool,
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> FHECircuit:
    """Compile the OPGraph to an FHECircuit.

    Args:
        op_graph (OPGraph): the OPGraph to compile.
        show_mlir (bool): determine whether we print the mlir string.
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation

    Returns:
        FHECircuit: the compiled circuit and the compiled FHECircuit
    """

    (
        compilation_configuration,
        compilation_artifacts,
    ) = sanitize_compilation_configuration_and_artifacts(
        compilation_configuration, compilation_artifacts
    )

    def compilation_function():
        return _compile_op_graph_to_fhe_circuit_internal(
            op_graph, show_mlir, compilation_configuration, compilation_artifacts
        )

    result = run_compilation_function_with_error_management(
        compilation_function, compilation_configuration, compilation_artifacts
    )

    # for mypy
    assert isinstance(result, FHECircuit)
    return result


def _compile_numpy_function_internal(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
    show_mlir: bool,
) -> FHECircuit:
    """Compile an homomorphic program (internal part of the API).

    Args:
        function_to_compile (Callable): The function you want to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]]]): The inputset over which op_graph
            is evaluated. It needs to be an iterable on tuples which are of the same length than the
            number of parameters in the function, and in the same order than these same parameters
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo

    Returns:
        CompilerEngine: engine to run and debug the compiled graph
    """

    # Compile into an OPGraph
    op_graph = _compile_numpy_function_into_op_graph_and_measure_bounds_internal(
        function_to_compile,
        function_parameters,
        inputset,
        compilation_configuration,
        compilation_artifacts,
    )

    fhe_circuit = _compile_op_graph_to_fhe_circuit_internal(
        op_graph, show_mlir, compilation_configuration, compilation_artifacts
    )

    return fhe_circuit


def compile_numpy_function(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]], str],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    show_mlir: bool = False,
) -> FHECircuit:
    """Compile an homomorphic program (main API).

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]], str]): The inputset over which
            op_graph is evaluated. It needs to be an iterable on tuples which are of the same length
            than the number of parameters in the function, and in the same order than these same
            parameters. Alternatively, it can be "random" but that's an unstable feature and should
            not be used in production.
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo

    Returns:
        CompilerEngine: engine to run and debug the compiled graph
    """

    (
        compilation_configuration,
        compilation_artifacts,
    ) = sanitize_compilation_configuration_and_artifacts(
        compilation_configuration, compilation_artifacts
    )

    inputset = get_inputset_to_use(function_parameters, inputset, compilation_configuration)

    def compilation_function():
        return _compile_numpy_function_internal(
            function_to_compile,
            function_parameters,
            inputset,
            compilation_configuration,
            compilation_artifacts,
            show_mlir,
        )

    result = run_compilation_function_with_error_management(
        compilation_function, compilation_configuration, compilation_artifacts
    )

    # for mypy
    assert isinstance(result, FHECircuit)
    return result
