"""Helper functions for tracing"""
from inspect import signature
from typing import Callable, Dict, Iterable, Set, Tuple, Type

import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph

from ..data_types import BaseValue
from ..representation import intermediate as ir
from .base_tracer import BaseTracer


def make_input_tracer(
    tracer_class: Type[BaseTracer],
    input_name: str,
    input_value: BaseValue,
) -> BaseTracer:
    """Helper function to create a tracer for an input value

    Args:
        tracer_class (Type[BaseTracer]): the class of tracer to create an Input for
        input_name (str): the name of the input in the traced function
        input_value (BaseValue): the Value that is an input and needs to be wrapped in an
            BaseTracer

    Returns:
        BaseTracer: The BaseTracer for that input value
    """
    return tracer_class([], ir.Input(input_value, input_name), 0)


def prepare_function_parameters(
    function_to_trace: Callable, function_parameters: Dict[str, BaseValue]
) -> Dict[str, BaseValue]:
    """Function to filter the passed function_parameters to trace function_to_trace

    Args:
        function_to_trace (Callable): function that will be traced for which parameters are checked
        function_parameters (Dict[str, BaseValue]): parameters given to trace the function

    Raises:
        ValueError: Raised when some parameters are missing to trace function_to_trace

    Returns:
        Dict[str, BaseValue]: filtered function_parameters dictionary
    """
    function_signature = signature(function_to_trace)

    missing_args = function_signature.parameters.keys() - function_parameters.keys()

    if len(missing_args) > 0:
        raise ValueError(
            f"The function '{function_to_trace.__name__}' requires the following parameters"
            f"that were not provided: {', '.join(sorted(missing_args))}"
        )

    useless_arguments = function_parameters.keys() - function_signature.parameters.keys()
    useful_arguments = function_signature.parameters.keys() - useless_arguments

    return {k: function_parameters[k] for k in useful_arguments}


def create_graph_from_output_tracers(
    output_tracers: Iterable[BaseTracer],
) -> nx.MultiDiGraph:
    """Generate a networkx Directed Graph that will represent the computation from a traced function

    Args:
        output_tracers (Iterable[BaseTracer]): the output tracers resulting from running the
            function over the proper input tracers

    Returns:
        nx.MultiDiGraph: Directed Graph that is guaranteed to be a DAG containing the ir nodes
            representing the traced program/function
    """
    graph = nx.MultiDiGraph()

    visited_tracers: Set[BaseTracer] = set()
    current_tracers = tuple(output_tracers)

    while current_tracers:
        next_tracers: Tuple[BaseTracer, ...] = tuple()
        for tracer in current_tracers:
            current_ir_node = tracer.traced_computation
            graph.add_node(current_ir_node, content=current_ir_node)

            for input_idx, input_tracer in enumerate(tracer.inputs):
                input_ir_node = input_tracer.traced_computation
                graph.add_node(input_ir_node, content=input_ir_node)
                graph.add_edge(input_ir_node, current_ir_node, input_idx=input_idx)
                if input_tracer not in visited_tracers:
                    next_tracers += (input_tracer,)

            visited_tracers.add(tracer)

        current_tracers = next_tracers

    assert is_directed_acyclic_graph(graph)

    return graph
