"""Code to wrap and make manipulating networkx graphs easier"""

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping

import networkx as nx

from .data_types.floats import Float
from .data_types.integers import make_integer_to_hold_ints
from .representation import intermediate as ir
from .tracing import BaseTracer
from .tracing.tracing_helpers import create_graph_from_output_tracers


class OPGraph:
    """Class to make work with nx graphs easier"""

    graph: nx.MultiDiGraph
    input_nodes: Mapping[int, ir.Input]
    output_nodes: Mapping[int, ir.IntermediateNode]

    def __init__(self, output_tracers: Iterable[BaseTracer]) -> None:
        self.output_nodes = {
            output_idx: tracer.traced_computation
            for output_idx, tracer in enumerate(output_tracers)
        }
        self.graph = create_graph_from_output_tracers(output_tracers)
        self.input_nodes = {
            node.program_input_idx: node
            for node in self.graph.nodes()
            if len(self.graph.pred[node]) == 0 and isinstance(node, ir.Input)
        }

    def evaluate(self, inputs: Mapping[int, Any]) -> Dict[ir.IntermediateNode, Any]:
        """Function to evaluate a graph and get intermediate values for all nodes

        Args:
            inputs (Mapping[int, Any]): The inputs to the program

        Returns:
            Dict[ir.IntermediateNode, Any]: Dictionary with node as keys and resulting values
        """
        node_results: Dict[ir.IntermediateNode, Any] = {}

        for node in nx.topological_sort(self.graph):
            if not isinstance(node, ir.Input):
                curr_inputs = {}
                for pred_node in self.graph.pred[node]:
                    edges = self.graph.get_edge_data(pred_node, node)
                    for edge in edges.values():
                        curr_inputs[edge["input_idx"]] = node_results[pred_node]
                node_results[node] = node.evaluate(curr_inputs)
            else:
                node_results[node] = node.evaluate({0: inputs[node.program_input_idx]})

        return node_results

    def update_values_with_bounds(self, node_bounds: dict):
        """Update nodes inputs and outputs values with data types able to hold data ranges measured
            and passed in nodes_bounds

        Args:
            node_bounds (dict): Dictionary with nodes as keys, holding dicts with a 'min' and 'max'
                keys. Those bounds will be taken as the data range to be represented, per node.
        """

        node: ir.IntermediateNode

        for node in self.graph.nodes():
            current_node_bounds = node_bounds[node]
            min_bound, max_bound = current_node_bounds["min"], current_node_bounds["max"]

            if not isinstance(node, ir.Input):
                for output_value in node.outputs:
                    if isinstance(min_bound, int) and isinstance(max_bound, int):
                        output_value.data_type = make_integer_to_hold_ints(
                            (min_bound, max_bound), force_signed=False
                        )
                    else:
                        output_value.data_type = Float(64)
            else:
                # Currently variable inputs are only allowed to be integers
                assert isinstance(min_bound, int) and isinstance(max_bound, int), (
                    f"Inputs to a graph should be integers, got bounds that were not float, \n"
                    f"min: {min_bound} ({type(min_bound)}), max: {max_bound} ({type(max_bound)})"
                )
                node.inputs[0].data_type = make_integer_to_hold_ints(
                    (min_bound, max_bound), force_signed=False
                )
                node.outputs[0] = deepcopy(node.inputs[0])

            # TODO: #57 manage multiple outputs from a node, probably requires an output_idx when
            # adding an edge
            assert len(node.outputs) == 1

            successors = self.graph.succ[node]
            for succ in successors:
                edge_data = self.graph.get_edge_data(node, succ)
                for edge in edge_data.values():
                    input_idx = edge["input_idx"]
                    succ.inputs[input_idx] = deepcopy(node.outputs[0])
