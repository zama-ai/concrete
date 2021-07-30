"""Code to wrap and make manipulating networkx graphs easier"""

from typing import Any, Dict, Iterable, Mapping

import networkx as nx

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
            if len(self.graph.pred[node]) == 0
        }

        assert all(map(lambda x: isinstance(x, ir.Input), self.input_nodes.values()))

        graph_outputs = set(node for node in self.graph.nodes() if len(self.graph.succ[node]) == 0)

        assert set(self.output_nodes.values()) == graph_outputs

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
