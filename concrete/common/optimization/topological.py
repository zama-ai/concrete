"""File holding topological optimization/simplification code."""
from collections import defaultdict
from copy import deepcopy
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, cast

import networkx as nx
from loguru import logger

from ..compilation.artifacts import CompilationArtifacts
from ..data_types.floats import Float
from ..data_types.integers import Integer
from ..debugging import get_printable_graph
from ..debugging.custom_assert import assert_true
from ..operator_graph import OPGraph
from ..representation.intermediate import Constant, GenericFunction, Input, IntermediateNode
from ..values import TensorValue


def fuse_float_operations(
    op_graph: OPGraph,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
):
    """Find and fuse float domains into single Integer to Integer GenericFunction.

    Args:
        op_graph (OPGraph): The OPGraph to simplify
        compilation_artifacts (Optional[CompilationArtifacts]): The CompilationArtifacts of the
            current compilation, this argument is optional as it's not required to execute float
            fusing.
    """

    nx_graph = op_graph.graph
    processed_terminal_nodes: Set[IntermediateNode] = set()
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
            output_node_to_idx: Dict[IntermediateNode, List[int]] = {
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
                # fused_node is always a GenericFunction so output_idx == 0 always
                new_edge_data = deepcopy(edge_data)
                new_edge_data["output_idx"] = 0
                nx_graph.add_edge(fused_node, succ, key=edge_key, **new_edge_data)

        # Connect the node feeding the subgraph contained in fused_node
        # node_before_subgraph has a single integer output currently so output_idx == 0
        nx_graph.add_edge(node_before_subgraph, fused_node, input_idx=0, output_idx=0)

        op_graph.prune_nodes()
        if compilation_artifacts is not None:
            compilation_artifacts.add_operation_graph(
                f"after-float-fuse-{number_of_fuse}", op_graph
            )

        number_of_fuse += 1


def convert_float_subgraph_to_fused_node(
    op_graph: OPGraph,
    float_subgraph_start_nodes: Set[IntermediateNode],
    terminal_node: IntermediateNode,
    subgraph_all_nodes: Set[IntermediateNode],
) -> Optional[Tuple[GenericFunction, IntermediateNode]]:
    """Convert a float subgraph to an equivalent fused GenericFunction node.

    Args:
        op_graph (OPGraph): The OPGraph the float subgraph is part of.
        float_subgraph_start_nodes (Set[IntermediateNode]): The nodes starting the float subgraph
            in `op_graph`.
        terminal_node (IntermediateNode): The node ending the float subgraph.
        subgraph_all_nodes (Set[IntermediateNode]): All the nodes in the float subgraph.

    Returns:
        Optional[Tuple[GenericFunction, IntermediateNode]]: None if the float subgraph
            cannot be fused, otherwise returns a tuple containing the fused node and the node whose
            output must be plugged as the input to the subgraph.
    """

    node_with_issues_for_fusing: DefaultDict[IntermediateNode, List[str]] = defaultdict(list)

    subgraph_can_be_fused = subgraph_has_unique_variable_input(
        float_subgraph_start_nodes, terminal_node, node_with_issues_for_fusing
    )

    if subgraph_can_be_fused:
        # subgraph_values_allow_fusing can be called iff the subgraph has a unique variable input
        subgraph_can_be_fused = subgraph_nodes_and_values_allow_fusing(
            float_subgraph_start_nodes, subgraph_all_nodes, node_with_issues_for_fusing
        )

    # This test is separate from the previous one to only handle printing issues once
    if not subgraph_can_be_fused:
        float_subgraph = nx.MultiDiGraph(op_graph.graph.subgraph(subgraph_all_nodes))
        float_subgraph_as_op_graph = OPGraph.from_graph(float_subgraph, [], [terminal_node])

        printable_graph = get_printable_graph(
            float_subgraph_as_op_graph,
            show_data_types=True,
            highlighted_nodes=node_with_issues_for_fusing,
        )
        message = f"The following subgraph is not fusable:\n{printable_graph}"
        logger.warning(message)
        return None

    # Only one variable input node, find which node feeds its input
    variable_input_nodes = [
        node for node in float_subgraph_start_nodes if not isinstance(node, Constant)
    ]
    assert_true(len(variable_input_nodes) == 1)

    current_subgraph_variable_input = variable_input_nodes[0]
    assert_true(len(current_subgraph_variable_input.outputs) == 1)
    new_input_value = deepcopy(current_subgraph_variable_input.outputs[0])

    nx_graph = op_graph.graph

    nodes_after_input_set = subgraph_all_nodes.intersection(
        nx_graph.succ[current_subgraph_variable_input]
    )

    float_subgraph = nx.MultiDiGraph(nx_graph.subgraph(subgraph_all_nodes))

    new_subgraph_variable_input = Input(new_input_value, "float_subgraph_input", 0)
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
            # new_subgraph_variable_input is always an Input so output_idx == 0 always
            new_edge_data = deepcopy(edge_data)
            new_edge_data["output_idx"] = 0
            float_subgraph.add_edge(
                new_subgraph_variable_input,
                node_after_input,
                key=edge_key,
                **new_edge_data,
            )

    float_op_subgraph = OPGraph.from_graph(
        float_subgraph,
        [new_subgraph_variable_input],
        [terminal_node],
    )

    assert_true(len(terminal_node.outputs) == 1)

    # Create fused_node
    fused_node = GenericFunction(
        inputs=[deepcopy(new_subgraph_variable_input.inputs[0])],
        arbitrary_func=lambda x, float_op_subgraph, terminal_node: float_op_subgraph.evaluate(
            {0: x}
        )[terminal_node],
        output_value=terminal_node.outputs[0],
        op_kind="TLU",
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
    processed_terminal_nodes: Set[IntermediateNode],
) -> Optional[Tuple[Set[IntermediateNode], IntermediateNode, Set[IntermediateNode]]]:
    """Find a subgraph of the graph with float computations.

    The subgraph has a single terminal node with a single Integer output and has a single variable
    predecessor node with a single Integer output.

    Args:
        nx_graph (nx.MultiDiGraph): The networkx graph to search in.
        processed_terminal_nodes (Set[IntermediateNode]): The set of terminal nodes for which
            subgraphs have already been searched, those will be skipped.

    Returns:
        Optional[Tuple[Set[IntermediateNode], IntermediateNode, Set[IntermediateNode]]]:
            None if there are no float subgraphs to process in `nx_graph`. Otherwise returns a tuple
            containing the set of nodes beginning a float subgraph, the terminal node of the
            subgraph and the set of all the nodes in the subgraph.
    """

    def is_float_to_single_int_node(node: IntermediateNode) -> bool:
        return (
            any(isinstance(input_.dtype, Float) for input_ in node.inputs)
            and len(node.outputs) == 1
            and isinstance(node.outputs[0].dtype, Integer)
        )

    def single_int_output_node(node: IntermediateNode) -> bool:
        return len(node.outputs) == 1 and isinstance(node.outputs[0].dtype, Integer)

    float_subgraphs_terminal_nodes = (
        node
        for node in nx_graph.nodes()
        if is_float_to_single_int_node(node) and node not in processed_terminal_nodes
    )

    terminal_node: IntermediateNode

    try:
        terminal_node = next(float_subgraphs_terminal_nodes)
    except StopIteration:
        return None

    # Use dict as ordered set
    current_nodes = {terminal_node: None}
    float_subgraph_start_nodes: Set[IntermediateNode] = set()
    subgraph_all_nodes: Set[IntermediateNode] = set()
    while current_nodes:
        next_nodes: Dict[IntermediateNode, None] = {}
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


def subgraph_nodes_and_values_allow_fusing(
    float_subgraph_start_nodes: Set[IntermediateNode],
    subgraph_all_nodes: Set[IntermediateNode],
    node_with_issues_for_fusing: DefaultDict[IntermediateNode, List[str]],
) -> bool:
    """Check if a subgraph's values are compatible with fusing.

    A fused subgraph for example only works on an input tensor if the resulting GenericFunction
    can be applied per cell, hence shuffling or tensor shape changes make fusing impossible.

    Args:
        float_subgraph_start_nodes (Set[IntermediateNode]): The nodes starting the float subgraph.
        subgraph_all_nodes (Set[IntermediateNode]): All the nodes in the float subgraph.
        node_with_issues_for_fusing (DefaultDict[IntermediateNode, List[str]]): Dictionary to fill
            with potential nodes issues preventing fusing.

    Returns:
        bool: True if all inputs and outputs of the nodes in the subgraph are compatible with fusing
            i.e. outputs have the same shapes equal to the variable input.
    """

    node: IntermediateNode

    variable_input_nodes = [
        node for node in float_subgraph_start_nodes if not isinstance(node, Constant)
    ]

    assert_true(
        (num_variable_input_nodes := len(variable_input_nodes)) == 1,
        f"{subgraph_nodes_and_values_allow_fusing.__name__} "
        f"only works for subgraphs with 1 variable input node, got {num_variable_input_nodes}",
    )

    explicitely_non_fusable = [
        node
        for node in subgraph_all_nodes
        if isinstance(node, GenericFunction) and not node.op_attributes["fusable"]
    ]
    for node in explicitely_non_fusable:
        node_with_issues_for_fusing[node].append(
            "this node is explicitely marked by the package as non-fusable"
        )
    if len(explicitely_non_fusable) > 0:
        return False

    all_values_are_tensors = all(
        all(isinstance(input_, TensorValue) for input_ in node.inputs)
        and all(isinstance(output, TensorValue) for output in node.outputs)
        for node in subgraph_all_nodes
    )

    if not all_values_are_tensors:
        # This cannot be reached today as scalars are Tensors with shape == () (numpy convention)
        return False  # pragma: no cover

    variable_input_node = variable_input_nodes[0]

    # A cheap check is that the variable input node must have the biggest size, i.e. have the most
    # elements, meaning all constants will broadcast to its shape. This is because the
    # GenericFunction input and output must have the same shape so that it can be applied to each
    # of the input tensor cells.
    # There *may* be a way to manage the other case by simulating the broadcast of the smaller input
    # array and then concatenating/stacking the results. This is not currently doable as we don't
    # have a concatenate operator on the compiler side.
    # TODO: #587 https://github.com/zama-ai/concretefhe-internal/issues/587

    variable_input_node_output = cast(TensorValue, variable_input_node.outputs[0])
    variable_input_node_output_size, variable_input_node_output_shape = (
        variable_input_node_output.size,
        variable_input_node_output.shape,
    )

    constant_nodes_with_bigger_size_than_variable_input = [
        constant_input_node
        for constant_input_node in subgraph_all_nodes
        if isinstance(constant_input_node, Constant)
        and cast(TensorValue, constant_input_node.outputs[0]).size > variable_input_node_output_size
    ]

    for bigger_constant_node in constant_nodes_with_bigger_size_than_variable_input:
        bigger_constant_node_shape = cast(TensorValue, bigger_constant_node.outputs[0]).shape
        node_with_issues_for_fusing[bigger_constant_node].append(
            f"this constant node has a bigger shape {bigger_constant_node_shape} "
            f"than the subgraph's input: {variable_input_node_output_shape}"
        )

    if len(constant_nodes_with_bigger_size_than_variable_input) > 0:
        node_with_issues_for_fusing[variable_input_node].append(
            f"input node with shape {variable_input_node_output_shape}"
        )
        return False

    # Now that we know the variable input node has the biggest size we can check shapes are
    # consistent throughout the subgraph: outputs of ir nodes that are not constant must be equal.

    non_constant_nodes = (node for node in subgraph_all_nodes if not isinstance(node, Constant))

    nodes_with_different_output_shapes = {
        node: [
            (output_idx, output.shape)
            for output_idx, output in enumerate(node.outputs)
            if isinstance(output, TensorValue) and output.shape != variable_input_node
        ]
        for node in non_constant_nodes
        if any(
            isinstance(output, TensorValue) and output.shape != variable_input_node_output_shape
            for output in node.outputs
        )
    }

    for node, node_shape_infos in nodes_with_different_output_shapes.items():
        shape_issue_details = "; ".join(
            f"#{output_idx}, {output_shape}" for output_idx, output_shape in node_shape_infos
        )
        node_with_issues_for_fusing[node].append(
            f"output shapes: {shape_issue_details} are not the same as the subgraph's input: "
            f"{variable_input_node_output_shape}"
        )

    all_nodes_have_same_shape_as_input = len(nodes_with_different_output_shapes) == 0

    if not all_nodes_have_same_shape_as_input:
        node_with_issues_for_fusing[variable_input_node].append(
            f"input node with shape {variable_input_node_output_shape}"
        )

    # All non constant node outputs currently need to have the same shape
    return all_nodes_have_same_shape_as_input


def subgraph_has_unique_variable_input(
    float_subgraph_start_nodes: Set[IntermediateNode],
    terminal_node: IntermediateNode,
    node_with_issues_for_fusing: DefaultDict[IntermediateNode, List[str]],
) -> bool:
    """Check that only one of the nodes starting the subgraph is variable.

    Args:
        float_subgraph_start_nodes (Set[IntermediateNode]): The nodes starting the subgraph.
        terminal_node (IntermediateNode): The node ending the float subgraph.
        node_with_issues_for_fusing (DefaultDict[IntermediateNode, List[str]]): Dictionary to fill
            with potential nodes issues preventing fusing.

    Returns:
        bool: True if only one of the nodes is not an Constant
    """

    variable_inputs_list = [
        node for node in float_subgraph_start_nodes if not isinstance(node, Constant)
    ]
    variable_inputs_num = len(variable_inputs_list)

    # Only one input to the subgraph where computations are done in floats can be variable, this
    # is the only case we can manage with GenericFunction fusing
    has_unique_variable_input = variable_inputs_num == 1

    if not has_unique_variable_input:
        for node in variable_inputs_list:
            node_with_issues_for_fusing[node].append(
                f"one of {variable_inputs_num} variable inputs (can only have 1 for fusing)"
            )
        node_with_issues_for_fusing[terminal_node].append(
            f"cannot fuse here as the subgraph has {variable_inputs_num} variable inputs"
        )

    return has_unique_variable_input
