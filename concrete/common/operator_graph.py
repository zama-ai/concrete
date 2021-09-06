"""Code to wrap and make manipulating networkx graphs easier."""

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Union

import networkx as nx

from .data_types.base import BaseDataType
from .data_types.dtypes_helpers import get_base_data_type_for_python_constant_data
from .data_types.floats import Float
from .data_types.integers import Integer, make_integer_to_hold
from .representation import intermediate as ir
from .tracing import BaseTracer
from .tracing.tracing_helpers import create_graph_from_output_tracers


class OPGraph:
    """Class to make work with nx graphs easier."""

    graph: nx.MultiDiGraph
    input_nodes: Dict[int, ir.Input]
    output_nodes: Dict[int, ir.IntermediateNode]

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        input_nodes: Dict[int, ir.Input],
        output_nodes: Dict[int, ir.IntermediateNode],
    ) -> None:
        assert len(input_nodes) > 0, "Got a graph without input nodes which is not supported"
        assert all(
            isinstance(node, ir.Input) for node in input_nodes.values()
        ), "Got input nodes that were not ir.Input, which is not supported"
        assert all(
            isinstance(node, ir.IntermediateNode) for node in output_nodes.values()
        ), "Got output nodes which were not ir.IntermediateNode, which is not supported"

        self.graph = graph
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.prune_nodes()

    def __call__(self, *args) -> Union[Any, Tuple[Any, ...]]:
        inputs = dict(enumerate(args))

        assert len(inputs) == len(
            self.input_nodes
        ), f"Expected {len(self.input_nodes)} arguments, got {len(inputs)} : {args}"

        results = self.evaluate(inputs)
        tuple_result = tuple(results[output_node] for output_node in self.get_ordered_outputs())
        return tuple_result if len(tuple_result) > 1 else tuple_result[0]

    @staticmethod
    def from_output_tracers(output_tracers: Iterable[BaseTracer]) -> "OPGraph":
        """Construct OPGraph from output tracers.

        Args:
            output_tracers (Iterable[BaseTracer]): The tracers output by the function that was
                traced.

        Returns:
            OPGraph: The resulting OPGraph.
        """
        graph = create_graph_from_output_tracers(output_tracers)
        input_nodes = {
            node.program_input_idx: node
            for node in graph.nodes()
            if len(graph.pred[node]) == 0 and isinstance(node, ir.Input)
        }
        output_nodes = {
            output_idx: tracer.traced_computation
            for output_idx, tracer in enumerate(output_tracers)
        }
        return OPGraph(graph, input_nodes, output_nodes)

    @staticmethod
    def from_graph(
        graph: nx.MultiDiGraph,
        input_nodes: Iterable[ir.Input],
        output_nodes: Iterable[ir.IntermediateNode],
    ) -> "OPGraph":
        """Construct OPGraph from an existing networkx MultiDiGraph.

        Args:
            graph (nx.MultiDiGraph): The networkx MultiDiGraph to use.
            input_nodes (Iterable[ir.Input]): The input nodes of the MultiDiGraph.
            output_nodes (Iterable[ir.IntermediateNode]): The output nodes of the MultiDiGraph.

        Returns:
            OPGraph: The resulting OPGraph.
        """
        return OPGraph(graph, dict(enumerate(input_nodes)), dict(enumerate(output_nodes)))

    def get_ordered_inputs(self) -> List[ir.Input]:
        """Get the input nodes of the graph, ordered by their index.

        Returns:
            List[ir.Input]: ordered input nodes
        """
        return [self.input_nodes[idx] for idx in range(len(self.input_nodes))]

    def get_ordered_outputs(self) -> List[ir.IntermediateNode]:
        """Get the output nodes of the graph, ordered by their index.

        Returns:
            List[ir.IntermediateNode]: ordered input nodes
        """
        return [self.output_nodes[idx] for idx in range(len(self.output_nodes))]

    def evaluate(self, inputs: Dict[int, Any]) -> Dict[ir.IntermediateNode, Any]:
        """Function to evaluate a graph and get intermediate values for all nodes.

        Args:
            inputs (Dict[int, Any]): The inputs to the program

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

    def update_values_with_bounds(
        self,
        node_bounds: dict,
        get_base_data_type_for_constant_data: Callable[
            [Any], BaseDataType
        ] = get_base_data_type_for_python_constant_data,
    ):
        """Update values with bounds.

        Update nodes inputs and outputs values with data types able to hold data ranges measured
        and passed in nodes_bounds

        Args:
            node_bounds (dict): Dictionary with nodes as keys, holding dicts with a 'min' and 'max'
                keys. Those bounds will be taken as the data range to be represented, per node.
            get_base_data_type_for_constant_data (Callable[ [Type], BaseDataType ], optional): This
                is a callback function to convert data encountered during value updates to
                BaseDataType. This allows to manage data coming from foreign frameworks without
                specialising OPGraph. Defaults to get_base_data_type_for_python_constant_data.
        """
        node: ir.IntermediateNode

        for node in self.graph.nodes():
            current_node_bounds = node_bounds[node]
            min_bound, max_bound = (
                current_node_bounds["min"],
                current_node_bounds["max"],
            )

            min_data_type = get_base_data_type_for_constant_data(min_bound)
            max_data_type = get_base_data_type_for_constant_data(max_bound)

            if not isinstance(node, ir.Input):
                for output_value in node.outputs:
                    if isinstance(min_data_type, Integer) and isinstance(max_data_type, Integer):
                        output_value.data_type = make_integer_to_hold(
                            (min_bound, max_bound), force_signed=False
                        )
                    else:
                        output_value.data_type = Float(64)
            else:
                # Currently variable inputs are only allowed to be integers
                assert isinstance(min_data_type, Integer) and isinstance(max_data_type, Integer), (
                    f"Inputs to a graph should be integers, got bounds that were float, \n"
                    f"min: {min_bound} ({type(min_bound)}), max: {max_bound} ({type(max_bound)})"
                )
                node.inputs[0].data_type = make_integer_to_hold(
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

    def prune_nodes(self):
        """Function to remove unreachable nodes from outputs."""

        current_nodes = set(self.output_nodes.values())
        useful_nodes: Set[ir.IntermediateNode] = set()
        while current_nodes:
            next_nodes: Set[ir.IntermediateNode] = set()
            useful_nodes.update(current_nodes)
            for node in current_nodes:
                next_nodes.update(self.graph.pred[node])
            current_nodes = next_nodes

        useless_nodes = set(self.graph.nodes()) - useful_nodes
        self.graph.remove_nodes_from(useless_nodes)
