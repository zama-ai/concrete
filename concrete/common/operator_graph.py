"""Code to wrap and make manipulating networkx graphs easier."""

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Type, Union

import networkx as nx

from .data_types.base import BaseDataType
from .data_types.dtypes_helpers import (
    get_base_data_type_for_python_constant_data,
    get_type_constructor_for_python_constant_data,
)
from .data_types.floats import Float
from .data_types.integers import Integer, make_integer_to_hold
from .debugging.custom_assert import custom_assert
from .representation.intermediate import Input, IntermediateNode
from .tracing import BaseTracer
from .tracing.tracing_helpers import create_graph_from_output_tracers


class OPGraph:
    """Class to make work with nx graphs easier."""

    graph: nx.MultiDiGraph
    input_nodes: Dict[int, Input]
    output_nodes: Dict[int, IntermediateNode]

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        input_nodes: Dict[int, Input],
        output_nodes: Dict[int, IntermediateNode],
    ) -> None:
        custom_assert(
            len(input_nodes) > 0, "Got a graph without input nodes which is not supported"
        )
        custom_assert(
            all(isinstance(node, Input) for node in input_nodes.values()),
            "Got input nodes that were not Input, which is not supported",
        )
        custom_assert(
            all(isinstance(node, IntermediateNode) for node in output_nodes.values()),
            "Got output nodes which were not IntermediateNode, which is not supported",
        )

        self.graph = graph
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.prune_nodes()

    def __call__(self, *args) -> Union[Any, Tuple[Any, ...]]:
        inputs = dict(enumerate(args))

        custom_assert(
            len(inputs) == len(self.input_nodes),
            f"Expected {len(self.input_nodes)} arguments, got {len(inputs)} : {args}",
        )

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
            if len(graph.pred[node]) == 0 and isinstance(node, Input)
        }
        output_nodes = {
            output_idx: tracer.traced_computation
            for output_idx, tracer in enumerate(output_tracers)
        }
        return OPGraph(graph, input_nodes, output_nodes)

    @staticmethod
    def from_graph(
        graph: nx.MultiDiGraph,
        input_nodes: Iterable[Input],
        output_nodes: Iterable[IntermediateNode],
    ) -> "OPGraph":
        """Construct OPGraph from an existing networkx MultiDiGraph.

        Args:
            graph (nx.MultiDiGraph): The networkx MultiDiGraph to use.
            input_nodes (Iterable[Input]): The input nodes of the MultiDiGraph.
            output_nodes (Iterable[IntermediateNode]): The output nodes of the MultiDiGraph.

        Returns:
            OPGraph: The resulting OPGraph.
        """
        return OPGraph(graph, dict(enumerate(input_nodes)), dict(enumerate(output_nodes)))

    def get_ordered_inputs(self) -> List[Input]:
        """Get the input nodes of the graph, ordered by their index.

        Returns:
            List[Input]: ordered input nodes
        """
        return [self.input_nodes[idx] for idx in range(len(self.input_nodes))]

    def get_ordered_outputs(self) -> List[IntermediateNode]:
        """Get the output nodes of the graph, ordered by their index.

        Returns:
            List[IntermediateNode]: ordered input nodes
        """
        return [self.output_nodes[idx] for idx in range(len(self.output_nodes))]

    def evaluate(self, inputs: Dict[int, Any]) -> Dict[IntermediateNode, Any]:
        """Evaluate a graph and get intermediate values for all nodes.

        Args:
            inputs (Dict[int, Any]): The inputs to the program

        Returns:
            Dict[IntermediateNode, Any]: Dictionary with node as keys and resulting values
        """
        node_results: Dict[IntermediateNode, Any] = {}

        for node in nx.topological_sort(self.graph):
            if not isinstance(node, Input):
                curr_inputs = {}
                for pred_node in self.graph.pred[node]:
                    edges = self.graph.get_edge_data(pred_node, node)
                    curr_inputs.update(
                        {edge["input_idx"]: node_results[pred_node] for edge in edges.values()}
                    )
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
        get_type_constructor_for_constant_data: Callable[
            ..., Type
        ] = get_type_constructor_for_python_constant_data,
    ):
        """Update values with bounds.

        Update nodes inputs and outputs values with data types able to hold data ranges measured
        and passed in nodes_bounds

        Args:
            node_bounds (dict): Dictionary with nodes as keys, holding dicts with a 'min' and 'max'
                keys. Those bounds will be taken as the data range to be represented, per node.
            get_base_data_type_for_constant_data (Callable[ [Any], BaseDataType ], optional): This
                is a callback function to convert data encountered during value updates to
                BaseDataType. This allows to manage data coming from foreign frameworks without
                specialising OPGraph. Defaults to get_base_data_type_for_python_constant_data.
            get_type_constructor_for_constant_data (Callable[ ..., Type ], optional): This is a
                callback function to determine the type constructor of the data encountered while
                updating the graph bounds. Defaults to get_type_constructor_python_constant_data.
        """
        node: IntermediateNode

        for node in self.graph.nodes():
            current_node_bounds = node_bounds[node]
            min_bound, max_bound = (
                current_node_bounds["min"],
                current_node_bounds["max"],
            )

            min_data_type = get_base_data_type_for_constant_data(min_bound)
            max_data_type = get_base_data_type_for_constant_data(max_bound)

            min_data_type_constructor = get_type_constructor_for_constant_data(min_bound)
            max_data_type_constructor = get_type_constructor_for_constant_data(max_bound)

            custom_assert(
                max_data_type_constructor == min_data_type_constructor,
                (
                    f"Got two different type constructors for min and max bound: "
                    f"{min_data_type_constructor}, {max_data_type_constructor}"
                ),
            )

            data_type_constructor = max_data_type_constructor

            if not isinstance(node, Input):
                for output_value in node.outputs:
                    if isinstance(min_data_type, Integer) and isinstance(max_data_type, Integer):
                        output_value.dtype = make_integer_to_hold(
                            (min_bound, max_bound), force_signed=False
                        )
                    else:
                        custom_assert(
                            isinstance(min_data_type, Float) and isinstance(max_data_type, Float),
                            (
                                "min_bound and max_bound have different common types, "
                                "this should never happen.\n"
                                f"min_bound: {min_data_type}, max_bound: {max_data_type}"
                            ),
                        )
                        output_value.dtype = Float(64)
                    output_value.dtype.underlying_type_constructor = data_type_constructor
            else:
                # Currently variable inputs are only allowed to be integers
                custom_assert(
                    isinstance(min_data_type, Integer) and isinstance(max_data_type, Integer),
                    (
                        f"Inputs to a graph should be integers, got bounds that were float, \n"
                        f"min: {min_bound} ({type(min_bound)}), "
                        f"max: {max_bound} ({type(max_bound)})"
                    ),
                )
                node.inputs[0].dtype = make_integer_to_hold(
                    (min_bound, max_bound), force_signed=False
                )
                node.inputs[0].dtype.underlying_type_constructor = data_type_constructor

                node.outputs[0] = deepcopy(node.inputs[0])

            # TODO: #57 manage multiple outputs from a node, probably requires an output_idx when
            # adding an edge
            custom_assert(len(node.outputs) == 1)

            successors = self.graph.succ[node]
            for succ in successors:
                edge_data = self.graph.get_edge_data(node, succ)
                for edge in edge_data.values():
                    input_idx = edge["input_idx"]
                    succ.inputs[input_idx] = deepcopy(node.outputs[0])

    def prune_nodes(self):
        """Remove unreachable nodes from outputs."""

        current_nodes = set(self.output_nodes.values())
        useful_nodes: Set[IntermediateNode] = set()
        while current_nodes:
            next_nodes: Set[IntermediateNode] = set()
            useful_nodes.update(current_nodes)
            for node in current_nodes:
                next_nodes.update(self.graph.pred[node])
            current_nodes = next_nodes

        useless_nodes = set(self.graph.nodes()) - useful_nodes
        self.graph.remove_nodes_from(useless_nodes)
