"""Helper functions for tracing."""
import collections
from contextlib import contextmanager
from inspect import signature
from typing import Callable, Dict, Iterable, List, OrderedDict, Set, Type

import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph

from ..debugging.custom_assert import assert_true
from ..representation.intermediate import Input
from ..values import BaseValue
from .base_tracer import BaseTracer


def make_input_tracers(
    tracer_class: Type[BaseTracer],
    function_parameters: OrderedDict[str, BaseValue],
) -> OrderedDict[str, BaseTracer]:
    """Create tracers for a function's parameters.

    Args:
        tracer_class (Type[BaseTracer]): the class of tracer to create an Input for
        function_parameters (OrderedDict[str, BaseValue]): the dictionary with the parameters names
            and corresponding Values

    Returns:
        OrderedDict[str, BaseTracer]: the dictionary containing the Input Tracers for each parameter
    """
    return collections.OrderedDict(
        (param_name, make_input_tracer(tracer_class, param_name, input_idx, param))
        for input_idx, (param_name, param) in enumerate(function_parameters.items())
    )


def make_input_tracer(
    tracer_class: Type[BaseTracer],
    input_name: str,
    input_idx: int,
    input_value: BaseValue,
) -> BaseTracer:
    """Create a tracer for an input value.

    Args:
        tracer_class (Type[BaseTracer]): the class of tracer to create an Input for
        input_name (str): the name of the input in the traced function
        input_idx (int): the input index in the function parameters
        input_value (BaseValue): the Value that is an input and needs to be wrapped in an
            BaseTracer

    Returns:
        BaseTracer: The BaseTracer for that input value
    """
    return tracer_class([], Input(input_value, input_name, input_idx), 0)


def prepare_function_parameters(
    function_to_trace: Callable, function_parameters: Dict[str, BaseValue]
) -> OrderedDict[str, BaseValue]:
    """Filter the passed function_parameters to trace function_to_trace.

    Args:
        function_to_trace (Callable): function that will be traced for which parameters are checked
        function_parameters (Dict[str, BaseValue]): parameters given to trace the function

    Raises:
        ValueError: Raised when some parameters are missing to trace function_to_trace

    Returns:
        OrderedDict[str, BaseValue]: filtered function_parameters dictionary
    """
    function_signature = signature(function_to_trace)

    missing_args = function_signature.parameters.keys() - function_parameters.keys()

    if len(missing_args) > 0:
        raise ValueError(
            f"The function '{function_to_trace.__name__}' requires the following parameters"
            f"that were not provided: {', '.join(sorted(missing_args))}"
        )

    # This convoluted way of creating the dict is to ensure key order is maintained
    return collections.OrderedDict(
        (param_name, function_parameters[param_name])
        for param_name in function_signature.parameters.keys()
    )


def create_graph_from_output_tracers(
    output_tracers: Iterable[BaseTracer],
) -> nx.MultiDiGraph:
    """Generate a networkx Directed Graph that represents the computation from a traced function.

    Args:
        output_tracers (Iterable[BaseTracer]): the output tracers resulting from running the
            function over the proper input tracers

    Returns:
        nx.MultiDiGraph: Directed Graph that is guaranteed to be a DAG containing the ir nodes
            representing the traced program/function
    """
    graph = nx.MultiDiGraph()

    visited_tracers: Set[BaseTracer] = set()
    # use dict as ordered set
    current_tracers = {tracer: None for tracer in output_tracers}

    while current_tracers:
        # use dict as ordered set
        next_tracers: Dict[BaseTracer, None] = {}
        for tracer in current_tracers:
            if tracer in visited_tracers:
                continue
            current_ir_node = tracer.traced_computation
            graph.add_node(current_ir_node)

            for input_idx, input_tracer in enumerate(tracer.inputs):
                input_ir_node = input_tracer.traced_computation
                output_idx = input_tracer.output_idx
                graph.add_node(input_ir_node)
                graph.add_edge(
                    input_ir_node,
                    current_ir_node,
                    input_idx=input_idx,
                    output_idx=output_idx,
                )
                if input_tracer not in visited_tracers:
                    next_tracers.update({input_tracer: None})

            visited_tracers.add(tracer)

        current_tracers = next_tracers

    assert_true(is_directed_acyclic_graph(graph))

    # Check each edge is unique
    unique_edges = set(
        (pred, succ, tuple((k, v) for k, v in edge_data.items()))
        for pred, succ, edge_data in graph.edges(data=True)
    )
    number_of_edges = len(graph.edges)
    assert_true(len(unique_edges) == number_of_edges)

    return graph


@contextmanager
def tracing_context(tracer_classes: List[Type[BaseTracer]]):
    """Set tracer classes in tracing mode.

    Args:
        tracer_classes (List[Type[BaseTracer]]): The list of tracers for which we should enable
            tracing.
    """

    try:
        for tracer_class in tracer_classes:
            tracer_class.set_is_tracing(True)
        yield
    finally:
        for tracer_class in tracer_classes:
            tracer_class.set_is_tracing(False)
