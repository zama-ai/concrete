"""numpy compilation function."""

import sys
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy
from zamalang import CompilerEngine

from ..common.bounds_measurement.inputset_eval import eval_op_graph_bounds_on_inputset
from ..common.common_helpers import check_op_graph_is_integer_program
from ..common.compilation import CompilationArtifacts, CompilationConfiguration
from ..common.data_types import Integer
from ..common.debugging import get_printable_graph
from ..common.fhe_circuit import FHECircuit
from ..common.mlir import V0_OPSET_CONVERSION_FUNCTIONS, MLIRConverter
from ..common.mlir.utils import (
    check_graph_values_compatibility_with_mlir,
    extend_direct_lookup_tables,
    update_bit_width_for_mlir,
)
from ..common.operator_graph import OPGraph
from ..common.optimization.topological import fuse_float_operations
from ..common.representation.intermediate import IntermediateNode
from ..common.values import BaseValue
from ..numpy.tracing import trace_numpy_function
from .np_dtypes_helpers import (
    get_base_data_type_for_numpy_or_python_constant_data,
    get_base_value_for_numpy_or_python_constant_data,
    get_constructor_for_numpy_or_python_constant_data,
)


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


def _compile_numpy_function_into_op_graph_internal(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Iterable[Tuple[Any, ...]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Iterable[Tuple[Any, ...]]): The inputset over which op_graph is evaluated. It
            needs to be an iterable on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation

    Returns:
        OPGraph: compiled function into a graph
    """

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
        # Fuse float operations to have int to int UnivariateFunction
        if not check_op_graph_is_integer_program(op_graph):
            fuse_float_operations(op_graph, compilation_artifacts)

    # TODO: To be removed once we support more than integers
    offending_non_integer_nodes: List[IntermediateNode] = []
    op_grap_is_int_prog = check_op_graph_is_integer_program(op_graph, offending_non_integer_nodes)
    if not op_grap_is_int_prog:
        raise ValueError(
            f"{function_to_compile.__name__} cannot be compiled as it has nodes with either float"
            f" inputs or outputs.\nOffending nodes : "
            f"{', '.join(str(node) for node in offending_non_integer_nodes)}"
        )

    # Find bounds with the inputset
    inputset_size, node_bounds_and_samples = eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=compilation_configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
    )

    # Check inputset size
    inputset_size_upper_limit = 1

    # this loop will determine the number of possible inputs of the function
    # if a function have a single 3-bit input, for example, `inputset_size_upper_limit` will be 8
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

    # Add the initial graph as an artifact
    compilation_artifacts.add_operation_graph("final", op_graph)

    # Make sure the graph can be lowered to MLIR
    offending_nodes = check_graph_values_compatibility_with_mlir(op_graph)
    if offending_nodes is not None:
        raise RuntimeError(
            "function you are trying to compile isn't supported for MLIR lowering\n\n"
            + get_printable_graph(op_graph, show_data_types=True, highlighted_nodes=offending_nodes)
        )

    # Update bit_width for MLIR
    update_bit_width_for_mlir(op_graph)

    # TODO: workaround extend LUT #359
    extend_direct_lookup_tables(op_graph)

    return op_graph


def compile_numpy_function_into_op_graph(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Iterable[Tuple[Any, ...]],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Iterable[Tuple[Any, ...]]): The inputset over which op_graph is evaluated. It
            needs to be an iterable on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation

    Returns:
        OPGraph: compiled function into a graph
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

    # Try to compile the function and save partial artifacts on failure
    try:
        return _compile_numpy_function_into_op_graph_internal(
            function_to_compile,
            function_parameters,
            inputset,
            compilation_configuration,
            compilation_artifacts,
        )
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


def _compile_numpy_function_internal(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Iterable[Tuple[Any, ...]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
    show_mlir: bool,
) -> FHECircuit:
    """Compile an homomorphic program (internal part of the API).

    Args:
        function_to_compile (Callable): The function you want to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Iterable[Tuple[Any, ...]]): The inputset over which op_graph is evaluated. It
            needs to be an iterable on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
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
    op_graph = _compile_numpy_function_into_op_graph_internal(
        function_to_compile,
        function_parameters,
        inputset,
        compilation_configuration,
        compilation_artifacts,
    )

    # Convert graph to an MLIR representation
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    mlir_result = converter.convert(op_graph)

    # Show MLIR representation if requested
    if show_mlir:
        print(f"MLIR which is going to be compiled: \n{mlir_result}")

    # Add MLIR representation as an artifact
    compilation_artifacts.add_final_operation_graph_mlir(mlir_result)

    # Compile the MLIR representation
    engine = CompilerEngine()
    engine.compile_fhe(mlir_result)

    return FHECircuit(op_graph, engine)


def compile_numpy_function(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    inputset: Iterable[Tuple[Any, ...]],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    show_mlir: bool = False,
) -> FHECircuit:
    """Compile an homomorphic program (main API).

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        inputset (Iterable[Tuple[Any, ...]]): The inputset over which op_graph is evaluated. It
            needs to be an iterable on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo

    Returns:
        CompilerEngine: engine to run and debug the compiled graph
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

    # Try to compile the function and save partial artifacts on failure
    try:
        return _compile_numpy_function_internal(
            function_to_compile,
            function_parameters,
            inputset,
            compilation_configuration,
            compilation_artifacts,
            show_mlir,
        )
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
