"""File holding topological optimization/simplification code."""
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..compilation.artifacts import CompilationArtifacts
from ..data_types.floats import Float
from ..data_types.integers import Integer
from ..debugging.custom_assert import custom_assert
from ..operator_graph import OPGraph
from ..representation import intermediate as ir


def fuse_float_operations(
    op_graph: OPGraph,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
):
    """Find and fuse float domains into single Integer to Integer ArbitraryFunction.

    Args:
        op_graph (OPGraph): The OPGraph to simplify
        compilation_artifacts (Optional[CompilationArtifacts]): The CompilationArtifacts of the
            current compilation, this argument is optional as it's not required to execute float
            fusing.
    """

    nx_graph = op_graph.graph
    processed_terminal_nodes: Set[ir.IntermediateNode] = set()
    number_of_fuse = 0
    while True:
        float_subgraph_search_result = find_float_subgraph_with_unique_terminal_node(
            nx_graph, processed_terminal_nodes
        )
        if float_subgraph_search_result is None:
            break

        float_subgraph_start_nodes, terminal_node, subgraph_all_nodes = float_subgraph_search_result
        processed_terminal_nodes.add(terminal_node)

        subgraph_conversion_result = convert_float_subgraph_to_fused_node(
            op_graph,
            float_subgraph_start_nodes,
            terminal_node,
            subgraph_all_nodes,
        )

        # Not a subgraph we can handle, continue
        if subgraph_conversion_result is None:
            continue

        fused_node, node_before_subgraph = subgraph_conversion_result

        nx_graph.add_node(fused_node)

        if terminal_node in op_graph.output_nodes.values():
            # Output value replace it
            # As the graph changes recreate the output_node_to_idx dict
            output_node_to_idx: Dict[ir.IntermediateNode, List[int]] = {
                out_node: [] for out_node in op_graph.output_nodes.values()
            }
            for output_idx, output_node in op_graph.output_nodes.items():
                output_node_to_idx[output_node].append(output_idx)

            for output_idx in output_node_to_idx.get(terminal_node, []):
                op_graph.output_nodes[output_idx] = fused_node

        # Disconnect after terminal node and connect fused node instead
        terminal_node_succ = list(nx_graph.successors(terminal_node))
        for succ in terminal_node_succ:
            succ_edge_data = deepcopy(nx_graph.get_edge_data(terminal_node, succ))
            for edge_key, edge_data in succ_edge_data.items():
                nx_graph.remove_edge(terminal_node, succ, key=edge_key)
                nx_graph.add_edge(fused_node, succ, key=edge_key, **edge_data)

        # Connect the node feeding the subgraph contained in fused_node
        nx_graph.add_edge(node_before_subgraph, fused_node, input_idx=0)

        op_graph.prune_nodes()
        if compilation_artifacts is not None:
            compilation_artifacts.add_operation_graph(
                f"after-float-fuse-{number_of_fuse}", op_graph
            )

        number_of_fuse += 1


def convert_float_subgraph_to_fused_node(
    op_graph: OPGraph,
    float_subgraph_start_nodes: Set[ir.IntermediateNode],
    terminal_node: ir.IntermediateNode,
    subgraph_all_nodes: Set[ir.IntermediateNode],
) -> Optional[Tuple[ir.ArbitraryFunction, ir.IntermediateNode]]:
    """Convert a float subgraph to an equivalent fused ArbitraryFunction node.

    Args:
        op_graph (OPGraph): The OPGraph the float subgraph is part of.
        float_subgraph_start_nodes (Set[ir.IntermediateNode]): The nodes starting the float subgraph
            in `op_graph`.
        terminal_node (ir.IntermediateNode): The node ending the float subgraph.
        subgraph_all_nodes (Set[ir.IntermediateNode]): All the nodes in the float subgraph.

    Returns:
        Optional[Tuple[ir.ArbitraryFunction, ir.IntermediateNode]]: None if the float subgraph
            cannot be fused, otherwise returns a tuple containing the fused node and the node whose
            output must be plugged as the input to the subgraph.
    """

    if not subgraph_has_unique_variable_input(float_subgraph_start_nodes):
        return None

    # Only one variable input node, find which node feeds its input
    non_constant_start_nodes = [
        node for node in float_subgraph_start_nodes if not isinstance(node, ir.Constant)
    ]
    custom_assert(len(non_constant_start_nodes) == 1)

    current_subgraph_variable_input = non_constant_start_nodes[0]
    new_input_value = deepcopy(current_subgraph_variable_input.outputs[0])

    nx_graph = op_graph.graph

    nodes_after_input_set = subgraph_all_nodes.intersection(
        nx_graph.succ[current_subgraph_variable_input]
    )

    float_subgraph = nx.MultiDiGraph(nx_graph.subgraph(subgraph_all_nodes))

    new_subgraph_variable_input = ir.Input(new_input_value, "float_subgraph_input", 0)
    float_subgraph.add_node(new_subgraph_variable_input)

    for node_after_input in nodes_after_input_set:
        # Connect the new input to our subgraph
        edge_data_input_to_subgraph = deepcopy(
            float_subgraph.get_edge_data(
                current_subgraph_variable_input,
                node_after_input,
            )
        )
        for edge_key, edge_data in edge_data_input_to_subgraph.items():
            float_subgraph.remove_edge(
                current_subgraph_variable_input, node_after_input, key=edge_key
            )
            float_subgraph.add_edge(
                new_subgraph_variable_input,
                node_after_input,
                key=edge_key,
                **edge_data,
            )

    float_op_subgraph = OPGraph.from_graph(
        float_subgraph,
        [new_subgraph_variable_input],
        [terminal_node],
    )

    # Create fused_node
    fused_node = ir.ArbitraryFunction(
        deepcopy(new_subgraph_variable_input.inputs[0]),
        lambda x, float_op_subgraph, terminal_node: float_op_subgraph.evaluate({0: x})[
            terminal_node
        ],
        deepcopy(terminal_node.outputs[0].data_type),
        op_kwargs={
            "float_op_subgraph": float_op_subgraph,
            "terminal_node": terminal_node,
        },
        op_name="Subgraph",
    )

    return (
        fused_node,
        current_subgraph_variable_input,
    )


def find_float_subgraph_with_unique_terminal_node(
    nx_graph: nx.MultiDiGraph,
    processed_terminal_nodes: Set[ir.IntermediateNode],
) -> Optional[Tuple[Set[ir.IntermediateNode], ir.IntermediateNode, Set[ir.IntermediateNode]]]:
    """Find a subgraph of the graph with float computations.

    The subgraph has a single terminal node with a single Integer output and has a single variable
    predecessor node with a single Integer output.

    Args:
        nx_graph (nx.MultiDiGraph): The networkx graph to search in.
        processed_terminal_nodes (Set[ir.IntermediateNode]): The set of terminal nodes for which
            subgraphs have already been searched, those will be skipped.

    Returns:
        Optional[Tuple[Set[ir.IntermediateNode], ir.IntermediateNode, Set[ir.IntermediateNode]]]:
            None if there are no float subgraphs to process in `nx_graph`. Otherwise returns a tuple
            containing the set of nodes beginning a float subgraph, the terminal node of the
            subgraph and the set of all the nodes in the subgraph.
    """

    def is_float_to_single_int_node(node: ir.IntermediateNode) -> bool:
        return (
            any(isinstance(input_.data_type, Float) for input_ in node.inputs)
            and len(node.outputs) == 1
            and isinstance(node.outputs[0].data_type, Integer)
        )

    def single_int_output_node(node: ir.IntermediateNode) -> bool:
        return len(node.outputs) == 1 and isinstance(node.outputs[0].data_type, Integer)

    float_subgraphs_terminal_nodes = (
        node
        for node in nx_graph.nodes()
        if is_float_to_single_int_node(node) and node not in processed_terminal_nodes
    )

    terminal_node: ir.IntermediateNode

    try:
        terminal_node = next(float_subgraphs_terminal_nodes)
    except StopIteration:
        return None

    # Use dict as ordered set
    current_nodes = {terminal_node: None}
    float_subgraph_start_nodes: Set[ir.IntermediateNode] = set()
    subgraph_all_nodes: Set[ir.IntermediateNode] = set()
    while current_nodes:
        next_nodes: Dict[ir.IntermediateNode, None] = {}
        for node in current_nodes:
            subgraph_all_nodes.add(node)
            predecessors = nx_graph.pred[node]
            for pred in predecessors:
                if single_int_output_node(pred):
                    # Limit of subgraph, record that and record the node as we won't visit it
                    float_subgraph_start_nodes.add(pred)
                    subgraph_all_nodes.add(pred)
                else:
                    next_nodes.update({pred: None})
        current_nodes = next_nodes

    return float_subgraph_start_nodes, terminal_node, subgraph_all_nodes


def subgraph_has_unique_variable_input(
    float_subgraph_start_nodes: Set[ir.IntermediateNode],
) -> bool:
    """Check that only one of the nodes starting the subgraph is variable.

    Args:
        float_subgraph_start_nodes (Set[ir.IntermediateNode]): The nodes starting the subgraph.

    Returns:
        bool: True if only one of the nodes is not an ir.Constant
    """
    # Only one input to the subgraph where computations are done in floats is variable, this
    # is the only case we can manage with ArbitraryFunction fusing
    return sum(not isinstance(node, ir.Constant) for node in float_subgraph_start_nodes) == 1
