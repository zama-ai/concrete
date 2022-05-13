"""
Declaration of various functions and constants related to compilation.
"""

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from ..dtypes import Float, Integer
from ..representation import Graph, Node, Operation
from .artifacts import DebugArtifacts


def fuse(graph: Graph, artifacts: Optional[DebugArtifacts] = None):
    """
    Fuse appropriate subgraphs in a graph to a single Operation.Generic node.

    Args:
        graph (Graph):
            graph to search and update

        artifacts (Optional[DebugArtifacts], default = None):
            compilation artifacts to store information about the fusing process
    """

    nx_graph = graph.graph
    processed_terminal_nodes: Set[Node] = set()

    fusing_floats = True
    while True:
        subgraph_to_fuse = (
            find_float_subgraph_with_unique_terminal_node(
                nx_graph,
                processed_terminal_nodes,
            )
            if fusing_floats
            else find_tlu_subgraph_with_multiple_variable_inputs_that_has_a_single_common_ancestor(
                nx_graph,
                processed_terminal_nodes,
            )
        )

        if subgraph_to_fuse is None:
            if fusing_floats:
                fusing_floats = False
                processed_terminal_nodes.clear()
                continue
            break

        all_nodes, start_nodes, terminal_node = subgraph_to_fuse
        processed_terminal_nodes.add(terminal_node)

        subgraph_conversion_result = convert_subgraph_to_subgraph_node(
            nx_graph,
            all_nodes,
            start_nodes,
            terminal_node,
        )
        if subgraph_conversion_result is None:
            continue

        fused_node, node_before_subgraph = subgraph_conversion_result
        nx_graph.add_node(fused_node)

        if terminal_node in graph.output_nodes.values():
            output_node_to_idx: Dict[Node, List[int]] = {
                out_node: [] for out_node in graph.output_nodes.values()
            }
            for output_idx, output_node in graph.output_nodes.items():
                output_node_to_idx[output_node].append(output_idx)

            for output_idx in output_node_to_idx.get(terminal_node, []):
                graph.output_nodes[output_idx] = fused_node

        terminal_node_succ = list(nx_graph.successors(terminal_node))
        for succ in terminal_node_succ:
            succ_edge_data = deepcopy(nx_graph.get_edge_data(terminal_node, succ))
            for edge_key, edge_data in succ_edge_data.items():
                nx_graph.remove_edge(terminal_node, succ, key=edge_key)
                new_edge_data = deepcopy(edge_data)
                nx_graph.add_edge(fused_node, succ, key=edge_key, **new_edge_data)

        nx_graph.add_edge(node_before_subgraph, fused_node, input_idx=0)

        graph.prune_useless_nodes()
        if artifacts is not None:
            artifacts.add_graph("after-fusing", graph)


def find_float_subgraph_with_unique_terminal_node(
    nx_graph: nx.MultiDiGraph,
    processed_terminal_nodes: Set[Node],
) -> Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
    """
    Find a subgraph with float computations that end with an integer output.

    Args:
        nx_graph (nx.MultiDiGraph):
            graph to search

        processed_terminal_nodes (Set[Node]):
            set of terminal nodes which have already been searched for float subgraphs

    Returns:
        Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
            None if there are no such subgraphs,
            tuple containing all nodes in the subgraph, start nodes of the subgraph,
            and terminal node of the subgraph otherwise
    """

    terminal_nodes = (
        node
        for node in nx_graph.nodes()
        if (
            node not in processed_terminal_nodes
            and any(isinstance(input.dtype, Float) for input in node.inputs)
            and isinstance(node.output.dtype, Integer)
        )
    )
    try:
        terminal_node = next(terminal_nodes)
    except StopIteration:
        return None

    # networkx does not implement lowest common ancestor search for multidigraph, but we only care
    # about parent relationship here and not the meaning of edges, so we can convert our
    # multidigraph to a digraph and use the lca search algorithm (if needed), we create the
    # equivalent digraph here as it will avoid recreating it in a loop. Constant nodes could cause
    # issues in our search, so we remove them.
    equivalent_subgraph_without_constants = nx.DiGraph(nx_graph)
    constant_nodes = [
        node
        for node in equivalent_subgraph_without_constants.nodes()
        if node.operation == Operation.Constant
    ]
    equivalent_subgraph_without_constants.remove_nodes_from(constant_nodes)

    all_nodes: Dict[Node, None] = {}

    start_single_int_output_nodes_search_from = terminal_node
    while True:
        all_nodes, start_nodes = find_closest_integer_output_nodes(
            nx_graph,
            [start_single_int_output_nodes_search_from],
            all_nodes,
        )

        variable_start_nodes = [
            start_node for start_node in start_nodes if start_node.operation != Operation.Constant
        ]
        if len(variable_start_nodes) == 1:
            break

        # find a common ancestor as we need a single variable input node
        # lca == lowest common ancestor
        # lca search only works for node pairs in networkx, so we progressively find the ancestors
        # setting the lca by default to one of the nodes we are searching the lca for
        lca = variable_start_nodes.pop()
        while len(variable_start_nodes) > 0 and lca is not None:
            node_to_find_new_lca = variable_start_nodes.pop()
            if lca == node_to_find_new_lca:
                continue

            ancestors_of_lca = nx.ancestors(
                equivalent_subgraph_without_constants,
                lca,
            )
            ancestors_of_node_to_find_new_lca = nx.ancestors(
                equivalent_subgraph_without_constants,
                node_to_find_new_lca,
            )

            lca_is_ancestor_of_node_to_find_new_lca = lca in ancestors_of_node_to_find_new_lca
            node_to_find_new_lca_is_ancestor_of_lca = node_to_find_new_lca in ancestors_of_lca

            if lca_is_ancestor_of_node_to_find_new_lca or node_to_find_new_lca_is_ancestor_of_lca:
                variable_start_nodes += list(
                    pred
                    for pred in nx_graph.predecessors(
                        node_to_find_new_lca if lca_is_ancestor_of_node_to_find_new_lca else lca
                    )
                    if pred.operation != Operation.Constant
                )
                lca = lca if lca_is_ancestor_of_node_to_find_new_lca else node_to_find_new_lca
                continue

            lca = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
                equivalent_subgraph_without_constants, lca, node_to_find_new_lca, default=None
            )

        # if subgraph cannot be fused because there is no way to find a common ancestor, break
        if lca is None:
            break

        # add the nodes from the `start_nodes` to `lca`, to `all_nodes`
        all_nodes = add_nodes_from_to(nx_graph, start_nodes, {lca: None}, all_nodes)

        # if `lca` is a valid starting node for fusing break
        if isinstance(lca.output.dtype, Integer):
            # `lca` is the new start node
            start_nodes = {lca: None}
            break

        # otherwise, push a little further
        # (e.g., if there is a node just before, which has an integer output)
        start_single_int_output_nodes_search_from = lca

    return all_nodes, start_nodes, terminal_node


def find_tlu_subgraph_with_multiple_variable_inputs_that_has_a_single_common_ancestor(
    nx_graph: nx.MultiDiGraph,
    processed_terminal_nodes: Set[Node],
) -> Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
    """
    Find a subgraph with a tlu computation that has multiple variable inputs \
    where all variable inputs share a common ancestor.

    Args:
        nx_graph (nx.MultiDiGraph):
            graph to search

        processed_terminal_nodes (Set[Node]):
            set of terminal nodes which have already been searched for tlu subgraphs

    Returns:
        Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
            None if there are no such subgraphs,
            tuple containing all nodes in the subgraph, start nodes of the subgraph,
            and terminal node of the subgraph otherwise
    """

    terminal_nodes = (
        node
        for node in nx_graph.nodes()
        if (
            node not in processed_terminal_nodes
            and node.converted_to_table_lookup
            and all(isinstance(input.dtype, Integer) for input in node.inputs)
            and isinstance(node.output.dtype, Integer)
            and len(
                [
                    pred
                    for pred in nx_graph.predecessors(node)
                    if pred.operation != Operation.Constant
                ]
            )
            > 1
        )
    )
    try:
        terminal_node = next(terminal_nodes)
    except StopIteration:
        return None

    # networkx does not implement lowest common ancestor search for multidigraph, but we only care
    # about parent relationship here and not the meaning of edges, so we can convert our
    # multidigraph to a digraph and use the lca search algorithm (if needed), we create the
    # equivalent digraph here as it will avoid recreating it in a loop. Constant nodes could cause
    # issues in our search, so we remove them.
    equivalent_subgraph_without_constants = nx.DiGraph(nx_graph)
    constant_nodes = [
        node
        for node in equivalent_subgraph_without_constants.nodes()
        if node.operation == Operation.Constant
    ]
    equivalent_subgraph_without_constants.remove_nodes_from(constant_nodes)

    all_nodes: Dict[Node, None] = {}

    while True:
        variable_start_nodes = list(nx_graph.predecessors(terminal_node))

        # find a common ancestor as we need a single variable input node
        # lca == lowest common ancestor
        # lca search only works for node pairs in networkx, so we progressively find the ancestors
        # setting the lca by default to one of the nodes we are searching the lca for
        lca = variable_start_nodes.pop()
        while len(variable_start_nodes) > 0 and lca is not None:
            node_to_find_new_lca = variable_start_nodes.pop()
            if lca == node_to_find_new_lca:
                continue

            ancestors_of_lca = nx.ancestors(
                equivalent_subgraph_without_constants,
                lca,
            )
            ancestors_of_node_to_find_new_lca = nx.ancestors(
                equivalent_subgraph_without_constants,
                node_to_find_new_lca,
            )

            lca_is_ancestor_of_node_to_find_new_lca = lca in ancestors_of_node_to_find_new_lca
            node_to_find_new_lca_is_ancestor_of_lca = node_to_find_new_lca in ancestors_of_lca

            if lca_is_ancestor_of_node_to_find_new_lca or node_to_find_new_lca_is_ancestor_of_lca:
                variable_start_nodes += list(
                    pred
                    for pred in nx_graph.predecessors(
                        node_to_find_new_lca if lca_is_ancestor_of_node_to_find_new_lca else lca
                    )
                    if pred.operation != Operation.Constant
                )
                lca = lca if lca_is_ancestor_of_node_to_find_new_lca else node_to_find_new_lca
                continue

            lca = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
                equivalent_subgraph_without_constants, lca, node_to_find_new_lca, default=None
            )

        # if subgraph cannot be fused because there is no way to find a common ancestor, break
        if lca is None:
            start_nodes = {}
            break

        # add the nodes from the `start_nodes` to `lca`, to `all_nodes`
        all_nodes = add_nodes_from_to(
            nx_graph,
            list(nx_graph.predecessors(terminal_node)),
            {lca: None},
            all_nodes,
        )
        all_nodes[terminal_node] = None

        # if `lca` is a valid starting node for fusing break
        if isinstance(lca.output.dtype, Integer):
            # `lca` is the new start node
            start_nodes = {lca: None}
            break

    return all_nodes, start_nodes, terminal_node


def find_closest_integer_output_nodes(
    nx_graph: nx.MultiDiGraph,
    start_nodes: List[Node],
    all_nodes: Dict[Node, None],
) -> Tuple[Dict[Node, None], Dict[Node, None]]:
    """
    Find the closest upstream integer output nodes to a set of start nodes in a graph.

    Args:
        nx_graph (nx.MultiDiGraph):
            graph to search

        start_nodes (List[Node]):
            nodes from which to start the search

        all_nodes (Dict[Node, None]):
            set of nodes to be extended with visited nodes during the search

    Returns:
        Tuple[Dict[Node, None], Dict[Node, None]]:
            tuple containing extended `all_nodes` and integer output nodes closest to `start_nodes`
    """

    closest_integer_output_nodes: Dict[Node, None] = {}
    visited_nodes: Set[Node] = set()

    current_nodes = {start_node: None for start_node in start_nodes}
    while current_nodes:
        next_nodes: Dict[Node, None] = {}
        for node in current_nodes:
            if node not in visited_nodes:
                visited_nodes.add(node)

                all_nodes.update({node: None})
                for pred in nx_graph.predecessors(node):
                    if isinstance(pred.output.dtype, Integer):
                        closest_integer_output_nodes.update({pred: None})
                        all_nodes.update({pred: None})
                    else:
                        next_nodes.update({pred: None})
        current_nodes = next_nodes

    return all_nodes, closest_integer_output_nodes


def add_nodes_from_to(
    nx_graph: nx.MultiDiGraph,
    from_nodes: Iterable[Node],
    to_nodes: Dict[Node, None],
    all_nodes: Dict[Node, None],
) -> Dict[Node, None]:
    """
    Add nodes from `from_nodes` to `to_nodes`, to `all_nodes`.

    Args:
        nx_graph (nx.MultiDiGraph):
            graph to traverse

        from_nodes (Iterable[Node]):
            nodes from which extending `all_nodes` start

        to_nodes (Dict[Node, None]):
            nodes to which extending `all_nodes` stop

        all_nodes (Dict[Node, None]):
            nodes to be extended

    Returns:
        Dict[Node, None]:
            extended `all_nodes`
    """

    all_nodes.update(to_nodes)
    visited_nodes: Set[Node] = set()

    current_nodes = {from_node: None for from_node in from_nodes}
    while current_nodes:
        next_nodes: Dict[Node, None] = {}
        for node in current_nodes:
            if node not in visited_nodes:
                visited_nodes.add(node)

                all_nodes.update({node: None})
                if node not in to_nodes:
                    predecessors = nx_graph.predecessors(node)
                    next_nodes.update({pred: None for pred in predecessors if pred not in to_nodes})
        current_nodes = next_nodes

    return all_nodes


def convert_subgraph_to_subgraph_node(
    nx_graph: nx.MultiDiGraph,
    all_nodes: Dict[Node, None],
    start_nodes: Dict[Node, None],
    terminal_node: Node,
) -> Optional[Tuple[Node, Node]]:
    """
    Convert a subgraph to Operation.Generic node.

    Args:
        nx_graph (nx.MultiDiGraph):
            orginal networkx graph

        all_nodes (Dict[Node, None]):
            all nodes in the subgraph

        start_nodes (Dict[Node, None]):
            start nodes of the subgraph

        terminal_node (Node):
            terminal node of the subgraph

    Returns:
        Optional[Tuple[Node, Node]]:
            None if the subgraph cannot be fused,
            subgraph node and its predecessor otherwise
    """

    variable_input_nodes = [node for node in start_nodes if node.operation != Operation.Constant]
    if len(variable_input_nodes) != 1:
        return None

    variable_input_node = variable_input_nodes[0]
    if not subgraph_can_be_fused(all_nodes, variable_input_node):
        return None

    nx_subgraph = nx.MultiDiGraph(nx_graph)
    nodes_to_remove = [node for node in nx_subgraph.nodes() if node not in all_nodes]
    nx_subgraph.remove_nodes_from(nodes_to_remove)

    subgraph_variable_input_node = Node.input("input", deepcopy(variable_input_node.output))
    nx_subgraph.add_node(subgraph_variable_input_node)

    variable_input_node_successors = {
        node: None for node in all_nodes if node in nx_graph.succ[variable_input_node]
    }
    for successor in variable_input_node_successors:
        edges = deepcopy(nx_subgraph.get_edge_data(variable_input_node, successor))
        for edge_key, edge_data in edges.items():
            nx_subgraph.remove_edge(variable_input_node, successor, key=edge_key)
            new_edge_data = deepcopy(edge_data)
            nx_subgraph.add_edge(
                subgraph_variable_input_node,
                successor,
                key=edge_key,
                **new_edge_data,
            )

    subgraph = Graph(nx_subgraph, {0: subgraph_variable_input_node}, {0: terminal_node})
    subgraph_node = Node.generic(
        "subgraph",
        subgraph_variable_input_node.inputs,
        terminal_node.output,
        lambda x, subgraph, terminal_node: subgraph.evaluate(x)[terminal_node],
        kwargs={
            "subgraph": subgraph,
            "terminal_node": terminal_node,
        },
    )

    return subgraph_node, variable_input_node


def subgraph_can_be_fused(
    all_nodes: Dict[Node, None],
    variable_input_node: Node,
) -> bool:
    """
    Determine if a subgraph can be fused.

    e.g.,

    shuffling or reshaping a tensor make fusing impossible as there should be a one-to-one mapping
    between each cell of the input and each cell of the output for table lookups

    Args:
        all_nodes (Dict[Node, None]):
            all nodes in the subgraph

        variable_input_node (Node):
            variable input node to the subgraph

    Returns:
        bool:
            True if subgraph can be fused,
            False otherwise
    """

    constant_nodes_with_bigger_size_than_variable_input = [
        node
        for node in all_nodes
        if (
            node.operation == Operation.Constant
            and node.output.size > variable_input_node.output.size
        )
    ]
    if len(constant_nodes_with_bigger_size_than_variable_input) > 0:
        return False

    non_constant_nodes = (node for node in all_nodes if node.operation != Operation.Constant)
    for node in non_constant_nodes:
        if node == variable_input_node:
            continue

        if not node.is_fusable or node.output.shape != variable_input_node.output.shape:
            return False

    return True
