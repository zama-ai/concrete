"""numpy compilation function."""

import traceback
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy
from zamalang import CompilerEngine

from ..common.bounds_measurement.dataset_eval import eval_op_graph_bounds_on_dataset
from ..common.common_helpers import check_op_graph_is_integer_program
from ..common.compilation import CompilationArtifacts, CompilationConfiguration
from ..common.mlir import V0_OPSET_CONVERSION_FUNCTIONS, MLIRConverter
from ..common.mlir.utils import (
    is_graph_values_compatible_with_mlir,
    update_bit_width_for_mlir,
)
from ..common.operator_graph import OPGraph
from ..common.optimization.topological import fuse_float_operations
from ..common.representation import intermediate as ir
from ..common.values import BaseValue
from ..numpy.tracing import trace_numpy_function
from .np_dtypes_helpers import get_base_data_type_for_numpy_or_python_constant_data


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
    dataset: Iterator[Tuple[Any, ...]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
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
        # Fuse float operations to have int to int ArbitraryFunction
        if not check_op_graph_is_integer_program(op_graph):
            fuse_float_operations(op_graph, compilation_artifacts)

    # TODO: To be removed once we support more than integers
    offending_non_integer_nodes: List[ir.IntermediateNode] = []
    op_grap_is_int_prog = check_op_graph_is_integer_program(op_graph, offending_non_integer_nodes)
    if not op_grap_is_int_prog:
        raise ValueError(
            f"{function_to_compile.__name__} cannot be compiled as it has nodes with either float"
            f" inputs or outputs.\nOffending nodes : "
            f"{', '.join(str(node) for node in offending_non_integer_nodes)}"
        )

    # Find bounds with the dataset
    node_bounds = eval_op_graph_bounds_on_dataset(
        op_graph,
        dataset,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
    )

    # Add the bounds as an artifact
    compilation_artifacts.add_final_operation_graph_bounds(node_bounds)

    # Update the graph accordingly: after that, we have the compilable graph
    op_graph.update_values_with_bounds(
        node_bounds, get_base_data_type_for_numpy_or_python_constant_data
    )

    # Add the initial graph as an artifact
    compilation_artifacts.add_operation_graph("final", op_graph)

    # Make sure the graph can be lowered to MLIR
    if not is_graph_values_compatible_with_mlir(op_graph):
        raise TypeError("signed integers aren't supported for MLIR lowering")

    # Update bit_width for MLIR
    update_bit_width_for_mlir(op_graph)

    return op_graph


def compile_numpy_function_into_op_graph(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    dataset: Iterator[Tuple[Any, ...]],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
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
            dataset,
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
            with open(compilation_artifacts.output_directory.joinpath("traceback.txt"), "w") as f:
                f.write(traceback.format_exc())

        raise


def _compile_numpy_function_internal(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    dataset: Iterator[Tuple[Any, ...]],
    compilation_configuration: CompilationConfiguration,
    compilation_artifacts: CompilationArtifacts,
    show_mlir: bool,
) -> CompilerEngine:
    """Internal part of the API to be able to compile an homomorphic program.

    Args:
        function_to_compile (Callable): The function you want to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
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
        dataset,
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

    return engine


def compile_numpy_function(
    function_to_compile: Callable,
    function_parameters: Dict[str, BaseValue],
    dataset: Iterator[Tuple[Any, ...]],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    show_mlir: bool = False,
) -> CompilerEngine:
    """Main API to be able to compile an homomorphic program.

    Args:
        function_to_compile (Callable): The function to compile
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedScalar holding a 7bits unsigned Integer
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
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
            dataset,
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
            with open(compilation_artifacts.output_directory.joinpath("traceback.txt"), "w") as f:
                f.write(traceback.format_exc())

        raise
