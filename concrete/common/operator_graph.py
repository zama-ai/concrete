"""Code to wrap and make manipulating networkx graphs easier."""

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import networkx as nx

from .data_types.base import BaseDataType
from .data_types.dtypes_helpers import (
    get_base_data_type_for_python_constant_data,
    get_constructor_for_python_constant_data,
)
from .data_types.floats import Float
from .data_types.integers import Integer, make_integer_to_hold
from .debugging.custom_assert import assert_true
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
        assert_true(
            all(isinstance(node, Input) for node in input_nodes.values()),
            "Got input nodes that were not Input, which is not supported",
        )
        assert_true(
            all(isinstance(node, IntermediateNode) for node in output_nodes.values()),
            "Got output nodes which were not IntermediateNode, which is not supported",
        )

        self.graph = graph
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.prune_nodes()

    def __call__(self, *args) -> Union[Any, Tuple[Any, ...]]:
        assert_true(len(self.input_nodes) > 0, "Cannot evaluate a graph with no input nodes")
        inputs = dict(enumerate(args))

        assert_true(
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

    def get_ordered_preds(self, node: IntermediateNode) -> List[IntermediateNode]:
        """Get node predecessors ordered by their indices.

        Args:
            node (IntermediateNode): The node for which we want the ordered predecessors.

        Returns:
            List[IntermediateNode]: The list of predecessors ordered by input index.
        """
        # Replication of pred is managed e.g. x + x will yield the proper pred x twice
        idx_to_pred: Dict[int, IntermediateNode] = {}
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            idx_to_pred.update((data["input_idx"], pred) for data in edge_data.values())
        return [idx_to_pred[i] for i in range(len(idx_to_pred))]

    def get_ordered_preds_and_inputs_of(
        self, node: IntermediateNode
    ) -> List[Tuple[IntermediateNode, int]]:
        """Get node preds and inputs ordered by their indices.

        Args:
            node (IntermediateNode): the node for which we want the ordered inputs

        Returns:
            List[Tuple[IntermediateNode, int]]: the ordered list of preds and inputs
        """

        idx_to_inp: Dict[int, Tuple[IntermediateNode, int]] = {}
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            idx_to_inp.update(
                (data["input_idx"], (pred, data["output_idx"])) for data in edge_data.values()
            )
        return [idx_to_inp[i] for i in range(len(idx_to_inp))]

    def evaluate(self, inputs: Dict[int, Any]) -> Dict[IntermediateNode, Any]:
        """Evaluate a graph and get intermediate values for all nodes.

        Args:
            inputs (Dict[int, Any]): The inputs to the program

        Returns:
            Dict[IntermediateNode, Any]: Dictionary with node as keys and resulting values
        """
        node_results: Dict[IntermediateNode, Any] = {}

        def get_result_of_node_at_index(node: IntermediateNode, output_idx: int) -> Any:
            """Get the output result at index output_idx for a node.

            Args:
                node (IntermediateNode): the node from which we want the output.
                output_idx (int): which output we want.

            Returns:
                Any: the output value of the evaluation of node.
            """
            result = node_results[node]
            # TODO: #81 remove no cover once we have nodes with multiple outputs
            if isinstance(result, tuple):  # pragma: no cover
                # If the node has multiple outputs (i.e. the result is a tuple), return the
                # requested output
                return result[output_idx]
            # If the result is not a tuple, then the result is the node's only output. Check that
            # the requested index is 0 (as it's the only valid value) and return the result itself.
            assert_true(
                output_idx == 0,
                f"Unable to get output at index {output_idx} for node {node}.\n"
                f"Node result: {result}",
            )
            return result

        for node in nx.topological_sort(self.graph):
            if not isinstance(node, Input):
                curr_inputs = {}
                for pred_node in self.graph.predecessors(node):
                    edges = self.graph.get_edge_data(pred_node, node)
                    curr_inputs.update(
                        {
                            edge["input_idx"]: get_result_of_node_at_index(
                                pred_node,
                                output_idx=edge["output_idx"],
                            )
                            for edge in edges.values()
                        }
                    )
                node_results[node] = node.evaluate(curr_inputs)
            else:
                node_results[node] = node.evaluate({0: inputs[node.program_input_idx]})

        return node_results

    def update_values_with_bounds_and_samples(
        self,
        node_bounds_and_samples: dict,
        get_base_data_type_for_constant_data: Callable[
            [Any], BaseDataType
        ] = get_base_data_type_for_python_constant_data,
        get_constructor_for_constant_data: Callable[
            ..., Callable
        ] = get_constructor_for_python_constant_data,
    ):
        """Update values with bounds.

        Update nodes inputs and outputs values with data types able to hold data ranges measured
        and passed in nodes_bounds

        Args:
            node_bounds_and_samples (dict): Dictionary with nodes as keys, holding dicts with a
                'min', 'max' and 'sample' keys. Those bounds will be taken as the data range to be
                represented, per node. The sample allows to determine the data constructors to
                prepare the GenericFunction nodes for table generation.
            get_base_data_type_for_constant_data (Callable[ [Any], BaseDataType ], optional): This
                is a callback function to convert data encountered during value updates to
                BaseDataType. This allows to manage data coming from foreign frameworks without
                specialising OPGraph. Defaults to get_base_data_type_for_python_constant_data.
            get_constructor_for_constant_data (Callable[ ..., Callable ], optional): This is a
                callback function to determine the type constructor of the data encountered while
                updating the graph bounds. Defaults to get_constructor_for_python_constant_data.
        """
        node: IntermediateNode

        for node in self.graph.nodes():
            current_node_bounds_and_samples = node_bounds_and_samples[node]
            min_bound, max_bound, sample = (
                current_node_bounds_and_samples["min"],
                current_node_bounds_and_samples["max"],
                current_node_bounds_and_samples["sample"],
            )

            min_data_type = get_base_data_type_for_constant_data(min_bound)
            max_data_type = get_base_data_type_for_constant_data(max_bound)

            # This is a sanity check
            min_value_constructor = get_constructor_for_constant_data(min_bound)
            max_value_constructor = get_constructor_for_constant_data(max_bound)

            assert_true(
                max_value_constructor == min_value_constructor,
                (
                    f"Got two different type constructors for min and max bound: "
                    f"{min_value_constructor}, {max_value_constructor}"
                ),
            )

            value_constructor = get_constructor_for_constant_data(sample)

            if not isinstance(node, Input):
                for output_value in node.outputs:
                    if isinstance(min_data_type, Integer) and isinstance(max_data_type, Integer):
                        output_value.dtype = make_integer_to_hold(
                            (min_bound, max_bound), force_signed=False
                        )
                    else:
                        assert_true(
                            isinstance(min_data_type, Float) and isinstance(max_data_type, Float),
                            (
                                "min_bound and max_bound have different common types, "
                                "this should never happen.\n"
                                f"min_bound: {min_data_type}, max_bound: {max_data_type}"
                            ),
                        )
                        output_value.dtype = Float(64)
                    output_value.underlying_constructor = value_constructor
            else:
                # Currently variable inputs are only allowed to be integers
                assert_true(
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
                node.inputs[0].underlying_constructor = value_constructor

                node.outputs[0] = deepcopy(node.inputs[0])

            successors = self.graph.successors(node)
            for succ in successors:
                edge_data = self.graph.get_edge_data(node, succ)
                for edge in edge_data.values():
                    input_idx, output_idx = edge["input_idx"], edge["output_idx"]
                    succ.inputs[input_idx] = deepcopy(node.outputs[output_idx])

    def prune_nodes(self):
        """Remove unreachable nodes from outputs."""

        current_nodes = {node: None for node in self.get_ordered_outputs()}
        useful_nodes: Dict[IntermediateNode, None] = {}
        while current_nodes:
            next_nodes: Dict[IntermediateNode, None] = {}
            useful_nodes.update(current_nodes)
            for node in current_nodes:
                next_nodes.update({node: None for node in self.graph.predecessors(node)})
            current_nodes = next_nodes

        useless_nodes = [node for node in self.graph.nodes() if node not in useful_nodes]
        self.graph.remove_nodes_from(useless_nodes)
