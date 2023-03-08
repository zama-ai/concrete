"""
Declaration of various functions and constants related to compilation.
"""

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from ..dtypes import Float, Integer
from ..representation import Graph, Node, Operation
from .artifacts import DebugArtifacts

# ruff: noqa: ERA001


def fuse(graph: Graph, artifacts: Optional[DebugArtifacts] = None):
    """
    Fuse appropriate subgraphs in a graph to a single Operation.Generic node.

    Args:
        graph (Graph):
            graph to search and update

        artifacts (Optional[DebugArtifacts], default = None):
            compilation artifacts to store information about the fusing process

    Raises:
        RuntimeError:
            if there is a subgraph which needs to be fused cannot be fused
    """

    nx_graph = graph.graph
    processed_terminal_nodes: Set[Node] = set()

    fusing_floats = True
    while True:
        subgraph_to_fuse = (
            find_float_subgraph_with_unique_terminal_node(
                graph,
                processed_terminal_nodes,
            )
            if fusing_floats
            else find_tlu_subgraph_with_multiple_variable_inputs_that_has_a_single_common_ancestor(
                graph,
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

        fused_node, node_before_subgraph = convert_subgraph_to_subgraph_node(
            graph,
            all_nodes,
            start_nodes,
            terminal_node,
        )
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
    graph: Graph,
    processed_terminal_nodes: Set[Node],
) -> Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
    """
    Find a subgraph with float computations that end with an integer output.

    Args:
        graph (Graph):
            graph to search

        processed_terminal_nodes (Set[Node]):
            set of terminal nodes which have already been searched for float subgraphs

    Returns:
        Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
            None if there are no such subgraphs,
            tuple containing all nodes in the subgraph, start nodes of the subgraph,
            and terminal node of the subgraph otherwise
    """

    nx_graph = graph.graph
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

    all_nodes: Dict[Node, None] = {}

    start_single_int_output_nodes_search_from = terminal_node
    while True:
        all_nodes, start_nodes = find_closest_integer_output_nodes(
            graph,
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
        lca = find_single_lca(graph, variable_start_nodes)

        # if subgraph cannot be fused because there is no way to find a common ancestor, break
        if lca is None:
            break

        # add the nodes from the `start_nodes` to `lca`, to `all_nodes`
        all_nodes = add_nodes_from_to(graph, start_nodes, {lca: None}, all_nodes)

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
    graph: Graph,
    processed_terminal_nodes: Set[Node],
) -> Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
    """
    Find a subgraph with a tlu computation that has multiple variable inputs \
    where all variable inputs share a common ancestor.

    Args:
        graph (Graph):
            graph to search

        processed_terminal_nodes (Set[Node]):
            set of terminal nodes which have already been searched for tlu subgraphs

    Returns:
        Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:
            None if there are no such subgraphs,
            tuple containing all nodes in the subgraph, start nodes of the subgraph,
            and terminal node of the subgraph otherwise
    """

    nx_graph = graph.graph
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

    all_nodes: Dict[Node, None] = {}

    while True:
        variable_start_nodes = list(nx_graph.predecessors(terminal_node))

        # find a common ancestor as we need a single variable input node
        # lca == lowest common ancestor
        lca = find_single_lca(graph, variable_start_nodes)

        # if subgraph cannot be fused because there is no way to find a common ancestor, break
        if lca is None:
            start_nodes = {node: None for node in variable_start_nodes}
            all_nodes = {node: None for node in variable_start_nodes + [terminal_node]}
            break

        # add the nodes from the `start_nodes` to `lca`, to `all_nodes`
        all_nodes = add_nodes_from_to(
            graph,
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


def find_single_lca(graph: Graph, nodes: List[Node]) -> Optional[Node]:
    """
    Find the single lowest common ancestor of a list of nodes.

    Args:
        graph (Graph):
            graph to search for single lca

        nodes (List[Node]):
            nodes to find the single lca of

    Returns
        Optional[Node]:
            single lca if it exists, None otherwise
    """

    nx_graph = graph.graph

    # find all ancestors of `nodes`
    # nodes themselves need to be in this set because the single lca can be within `nodes`
    all_ancestors = [set(list(nx.ancestors(nx_graph, node)) + [node]) for node in nodes]

    # find common ancestors among `nodes`
    # if the single lca exists, it's in this set
    common_ancestors = {
        node
        for node in nx_graph.nodes()
        if node.operation != Operation.Constant
        and all(node in ancestors for ancestors in all_ancestors)
    }

    # iterate over every node in the graph reversed topological order
    # this is to ensure result, if found, is the single "lowest" common ancestor
    for candidate in reversed(list(nx.topological_sort(nx_graph))):
        # check if node is a common ancestor of all `nodes`
        if candidate not in common_ancestors:
            # if not, it cannot be the single lca
            continue

        # check if node is a single common ancestor of `nodes`
        if is_single_common_ancestor(graph, candidate, nodes):
            # if so, it's the single lca of `nodes`
            # so return it
            return candidate

    # if none of the nodes in `common_ancestors` is the single lca
    # there is no single lca of this set of nodes, so return None
    return None


def is_single_common_ancestor(
    graph: Graph,
    candidate: Node,
    nodes: List[Node],
) -> bool:
    """
    Determine if a node is the single common ancestor of a list of nodes.

    Note that this function doesn't care about `lowest` property of `lca`.

    Args:
        graph (Graph):
            graph to perform the check

        candidate (Node):
            node to determine single common ancestor status

        nodes (List[Node]):
            nodes to determine single common ancestor status against

    Returns
        bool:
            True if `candidate` is a single common ancestor of `nodes`, False otherwise
    """

    nx_graph = graph.graph

    # create a subgraph with `candidate` node
    subgraph = nx.DiGraph()
    subgraph.add_node(candidate)

    # iterate over `nodes` to add them to the subgraph
    # along with every path from `candidate` to them
    for node in nodes:
        subgraph.add_node(node)
        for path in nx.all_simple_paths(nx_graph, source=candidate, target=node):
            nx.add_path(subgraph, path)

    # iterate over the nodes of the subgraph
    for node in subgraph.nodes():
        # the condition below doesn't apply to `candidate`
        # as its predecessors are not in the subgraph
        if node == candidate:
            continue

        # find number of predecessors in the subgraph and in the original graph
        # except constant nodes in the original graph as
        #   - they are not in the subgraph
        #   - they don't affect fusability status
        predecessor_count_in_subgraph = len(list(subgraph.predecessors(node)))
        predecessor_count_in_nx_graph = len(
            [pred for pred in nx_graph.predecessors(node) if pred.operation != Operation.Constant]
        )

        # see if number of predecessors are different
        if predecessor_count_in_subgraph != predecessor_count_in_nx_graph:
            # if so, `candidate` cannot be a single common ancestor
            # reasoning for is explained below
            return False

    # if every node in the subgraph has the same number of predecessors
    # as in the original graph `candidate` is in fact a single common ancestor
    return True

    # Here is why this function works.
    #
    # Legend:
    #   - /|\- = Edge
    #   - (...) = Intermediate Node
    #   - {...} = Candidate Node
    #   - [...] = Node of which single common ancestor is searched
    #   - {[...]} = Both Candidate Node and Node of which single common ancestor is searched
    #
    # Consider the folowing graph:
    #
    # (3)       (x)     (2)
    #    \     /   \   /
    #     [{*}]    (/)
    #          \   /
    #           [+]
    #
    # - Operation: (x * 3) + (x / 2)
    # - Candidate: {*}
    # - Nodes: [*] and [+]
    #
    # So we want to know if multiplication node is a single common ancestor of
    # multiplication and addition nodes. The result is no in this case for our purposes.
    #
    # Once you apply the subgraph creation above, you'll get the following graph:
    #
    # (*)
    #  |
    # (+)
    #
    # In this subgraph, addition node only have a single predecessor,
    # which means there is path leading to the addition node and that path doesn't include
    # the multiplication node, so we conclude multiplication node is not a single common ancestor
    #
    # Now, consider the folowing graph:
    #
    # (3)     {x}     (2)
    #    \   /   \   /
    #     [*]     (/)
    #        \   /
    #         [+]
    #
    # - Operation: (x * 3) + (x / 2)
    # - Candidate: {x}
    # - Nodes: [*] and [+]
    #
    # So we want to know if the input node 'x' is the single common ancestor of
    # multiplication and addition nodes. The result is yes in this case.
    #
    # Once you apply the subgraph creation above, you'll get the following graph:
    #
    #     {x}
    #    /   \
    # [*]     (/)
    #    \   /
    #     [+]
    #
    # In this subgraph, every node except the candidate node
    # will keep all of their non-constant predecessors,
    # which means all of their non-constant predecessors originated
    # from the `candidate`, so it's a single common anscestor.
    #
    # When you think about it, this implementation makes a lot of sense for our purposes
    # It basically determines if `nodes` "solely" depend on the `candidate`,
    # which is the condition for fusing.


def find_closest_integer_output_nodes(
    graph: Graph,
    start_nodes: List[Node],
    all_nodes: Dict[Node, None],
) -> Tuple[Dict[Node, None], Dict[Node, None]]:
    """
    Find the closest upstream integer output nodes to a set of start nodes in a graph.

    Args:
        graph (Graph):
            graph to search

        start_nodes (List[Node]):
            nodes from which to start the search

        all_nodes (Dict[Node, None]):
            set of nodes to be extended with visited nodes during the search

    Returns:
        Tuple[Dict[Node, None], Dict[Node, None]]:
            tuple containing extended `all_nodes` and integer output nodes closest to `start_nodes`
    """

    nx_graph = graph.graph

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
    graph: Graph,
    from_nodes: Iterable[Node],
    to_nodes: Dict[Node, None],
    all_nodes: Dict[Node, None],
) -> Dict[Node, None]:
    """
    Add nodes from `from_nodes` to `to_nodes`, to `all_nodes`.

    Args:
        graph (Graph):
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

    nx_graph = graph.graph

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
    graph: Graph,
    all_nodes: Dict[Node, None],
    start_nodes: Dict[Node, None],
    terminal_node: Node,
) -> Tuple[Node, Node]:
    """
    Convert a subgraph to Operation.Generic node.

    Args:
        graph (Graph):
            orginal graph

        all_nodes (Dict[Node, None]):
            all nodes in the subgraph

        start_nodes (Dict[Node, None]):
            start nodes of the subgraph

        terminal_node (Node):
            terminal node of the subgraph

    Raises:
        RuntimeError:
            if subgraph is not fusable

    Returns:
        Tuple[Node, Node]:
            None if the subgraph cannot be fused,
            subgraph node and its predecessor otherwise
    """

    nx_graph = graph.graph

    variable_input_nodes = [node for node in start_nodes if node.operation != Operation.Constant]
    if len(variable_input_nodes) != 1:
        base_highlighted_nodes = {
            node: ["within this subgraph", node.location] for node in all_nodes
        }
        for variable_input_node in variable_input_nodes:
            base_highlighted_nodes[variable_input_node] = [
                "this is one of the input nodes",
                variable_input_node.location,
            ]

        raise RuntimeError(
            "A subgraph within the function you are trying to compile cannot be fused "
            "because it has multiple input nodes\n\n"
            + graph.format(highlighted_nodes=base_highlighted_nodes, show_bounds=False)
        )

    variable_input_node = variable_input_nodes[0]
    check_subgraph_fusability(graph, all_nodes, variable_input_node)

    nx_subgraph = nx.MultiDiGraph(nx_graph)
    nodes_to_remove = [node for node in nx_subgraph.nodes() if node not in all_nodes]
    nx_subgraph.remove_nodes_from(nodes_to_remove)

    subgraph_variable_input_node = Node.input("input", deepcopy(variable_input_node.output))
    nx_subgraph.add_node(subgraph_variable_input_node)

    subgraph_variable_input_node.location = variable_input_node.location
    subgraph_variable_input_node.tag = variable_input_node.tag
    subgraph_variable_input_node.created_at = variable_input_node.created_at

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

    original_location = terminal_node.location
    original_tag = terminal_node.tag
    original_created_at = terminal_node.created_at

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

    subgraph_node.location = original_location
    subgraph_node.tag = original_tag
    subgraph_node.created_at = original_created_at

    return subgraph_node, variable_input_node


def check_subgraph_fusability(
    graph: Graph,
    all_nodes: Dict[Node, None],
    variable_input_node: Node,
):
    """
    Determine if a subgraph can be fused.

    e.g.,

    shuffling or reshaping a tensor make fusing impossible as there should be a one-to-one mapping
    between each cell of the input and each cell of the output for table lookups

    Args:
        graph (Graph):
            original graph

        all_nodes (Dict[Node, None]):
            all nodes in the subgraph

        variable_input_node (Node):
            variable input node to the subgraph

    Raises:
        RuntimeError:
            if subgraph is not fusable
    """

    base_highlighted_nodes = {node: ["within this subgraph", node.location] for node in all_nodes}
    base_highlighted_nodes[variable_input_node] = [
        "with this input node",
        variable_input_node.location,
    ]

    non_constant_nodes = (node for node in all_nodes if node.operation != Operation.Constant)
    for node in non_constant_nodes:
        if node == variable_input_node:
            continue

        if not node.is_fusable:
            base_highlighted_nodes[node] = ["this node is not fusable", node.location]
            raise RuntimeError(
                "A subgraph within the function you are trying to compile cannot be fused "
                "because of a node, which is marked explicitly as non-fusable\n\n"
                + graph.format(highlighted_nodes=base_highlighted_nodes, show_bounds=False)
            )

        if node.output.shape != variable_input_node.output.shape:
            base_highlighted_nodes[node] = [
                "this node has a different shape than the input node",
                node.location,
            ]
            raise RuntimeError(
                "A subgraph within the function you are trying to compile cannot be fused "
                "because of a node, which is has a different shape than the input node\n\n"
                + graph.format(highlighted_nodes=base_highlighted_nodes, show_bounds=False)
            )

    return True
