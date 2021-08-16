"""hnumpy compilation function."""

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from ..common.bounds_measurement.dataset_eval import eval_op_graph_bounds_on_dataset
from ..common.compilation import CompilationArtifacts
from ..common.data_types import BaseValue
from ..common.mlir.utils import (
    is_graph_values_compatible_with_mlir,
    update_bit_width_for_mlir,
)
from ..common.operator_graph import OPGraph
from ..hnumpy.tracing import trace_numpy_function


def compile_numpy_function(
    function_to_trace: Callable,
    function_parameters: Dict[str, BaseValue],
    dataset: Iterator[Tuple[Any, ...]],
    compilation_artifacts: Optional[CompilationArtifacts] = None,
) -> OPGraph:
    """Main API of hnumpy, to be able to compile an homomorphic program.

    Args:
        function_to_trace (Callable): The function you want to trace
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedValue holding a 7bits unsigned Integer
        dataset (Iterator[Tuple[Any, ...]]): The dataset over which op_graph is evaluated. It
            needs to be an iterator on tuples which are of the same length than the number of
            parameters in the function, and in the same order than these same parameters
        compilation_artifacts (Optional[CompilationArtifacts]): Artifacts object to fill
            during compilation

    Returns:
        OPGraph: currently returns a compilable graph, but later, it will return an MLIR compatible
            with the compiler, and even later, it will return the result of the compilation
    """
    # Trace
    op_graph = trace_numpy_function(function_to_trace, function_parameters)

    # Find bounds with the dataset
    node_bounds = eval_op_graph_bounds_on_dataset(op_graph, dataset)

    # Update the graph accordingly: after that, we have the compilable graph
    op_graph.update_values_with_bounds(node_bounds)

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
