"""hnumpy compilation function."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

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
from ..hnumpy.tracing import trace_numpy_function
from .np_dtypes_helpers import get_base_data_type_for_numpy_or_python_constant_data


def compile_numpy_function_into_op_graph(
    function_to_trace: Callable,
    function_parameters: Dict[str, BaseValue],
    dataset: Iterator[Tuple[Any, ...]],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> OPGraph:
    """Compile a function into an OPGraph.

    Args:
        function_to_trace (Callable): The function you want to trace
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedValue holding a 7bits unsigned Integer
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

    # Trace
    op_graph = trace_numpy_function(function_to_trace, function_parameters)

    # Apply topological optimizations if they are enabled
    if compilation_configuration.enable_topological_optimizations:
        # Fuse float operations to have int to int ArbitraryFunction
        if not check_op_graph_is_integer_program(op_graph):
            fuse_float_operations(op_graph)

    # TODO: To be removed once we support more than integers
    offending_non_integer_nodes: List[ir.IntermediateNode] = []
    op_grap_is_int_prog = check_op_graph_is_integer_program(op_graph, offending_non_integer_nodes)
    if not op_grap_is_int_prog:
        raise ValueError(
            f"{function_to_trace.__name__} cannot be compiled as it has nodes with either float "
            f"inputs or outputs.\nOffending nodes : "
            f"{', '.join(str(node) for node in offending_non_integer_nodes)}"
        )

    # Find bounds with the dataset
    node_bounds = eval_op_graph_bounds_on_dataset(op_graph, dataset)

    # Update the graph accordingly: after that, we have the compilable graph
    op_graph.update_values_with_bounds(
        node_bounds, get_base_data_type_for_numpy_or_python_constant_data
    )

    # Make sure the graph can be lowered to MLIR
    if not is_graph_values_compatible_with_mlir(op_graph):
        raise TypeError("signed integers aren't supported for MLIR lowering")

    # Update bit_width for MLIR
    update_bit_width_for_mlir(op_graph)

    # Fill compilation artifacts
    if compilation_artifacts is not None:
        compilation_artifacts.operation_graph = op_graph
        compilation_artifacts.bounds = node_bounds

    return op_graph


def compile_numpy_function(
    function_to_trace: Callable,
    function_parameters: Dict[str, BaseValue],
    dataset: Iterator[Tuple[Any, ...]],
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> CompilerEngine:
    """Main API of hnumpy, to be able to compile an homomorphic program.

    Args:
        function_to_trace (Callable): The function you want to trace
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedValue holding a 7bits unsigned Integer
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        compilation_configuration (Optional[CompilationConfiguration]): Configuration object to use
            during compilation
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation

    Returns:
        CompilerEngine: engine to run and debug the compiled graph
    """

    # Create default configuration if custom configuration is not specified
    compilation_configuration = (
        CompilationConfiguration()
        if compilation_configuration is None
        else compilation_configuration
    )

    # Compile into an OPGraph
    op_graph = compile_numpy_function_into_op_graph(
        function_to_trace,
        function_parameters,
        dataset,
        compilation_configuration,
        compilation_artifacts,
    )

    # Convert graph to an MLIR representation
    converter = MLIRConverter(V0_OPSET_CONVERSION_FUNCTIONS)
    mlir_result = converter.convert(op_graph)

    # Compile the MLIR representation
    engine = CompilerEngine()
    engine.compile_fhe(mlir_result)

    return engine
