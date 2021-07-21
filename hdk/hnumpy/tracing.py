"""hnumpy tracing utilities"""
from typing import Callable, Dict

import networkx as nx

from ..common.data_types import BaseValue
from ..common.tracing import (
    BaseTracer,
    create_graph_from_output_tracers,
    make_input_tracer,
    prepare_function_parameters,
)


class NPTracer(BaseTracer):
    """Tracer class for numpy operations"""


def trace_numpy_function(
    function_to_trace: Callable, function_parameters: Dict[str, BaseValue]
) -> nx.MultiDiGraph:
    """Function used to trace a numpy function

    Args:
        function_to_trace (Callable): The function you want to trace
        function_parameters (Dict[str, BaseValue]): A dictionary indicating what each input of the
            function is e.g. an EncryptedValue holding a 7bits unsigned Integer

    Returns:
        nx.MultiDiGraph: The graph containing the ir nodes representing the computation done in the
            input function
    """
    function_parameters = prepare_function_parameters(function_to_trace, function_parameters)

    input_tracers = {
        param_name: make_input_tracer(NPTracer, param)
        for param_name, param in function_parameters.items()
    }

    # We could easily create a graph of NPTracer, but we may end up with dead nodes starting from
    # the inputs that's why we create the graph starting from the outputs
    output_tracers = function_to_trace(**input_tracers)
    if isinstance(output_tracers, NPTracer):
        output_tracers = (output_tracers,)

    graph = create_graph_from_output_tracers(output_tracers)

    return graph
