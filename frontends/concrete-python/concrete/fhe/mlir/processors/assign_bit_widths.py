"""
Declaration of `AssignBitWidths` graph processor.
"""
from __future__ import annotations

import typing
from collections.abc import Iterable

from ...dtypes import Integer
from ...representation import Graph, Node, Operation
from . import GraphProcessor


class AssignBitWidths(GraphProcessor):
    """
    Assign a precision to all nodes inputs/output.

    The precisions are compatible graph constraints and MLIR.
    There are two modes:
        - single precision: where all encrypted values have the same precision.
        - multi precision: where encrypted values can have different precisions.
    """

    def __init__(self, single_precision=False):
        self.single_precision = single_precision

    def apply(self, graph: Graph):
        nodes = graph.query_nodes()
        for node in nodes:
            assert isinstance(node.output.dtype, Integer)
            node.properties["original_bit_width"] = node.output.dtype.bit_width

        if self.single_precision:
            assign_single_precision(nodes)
        else:
            assign_multi_precision(graph, nodes)


def assign_single_precision(nodes: list[Node]):
    """Assign one single encryption precision to all nodes."""
    p = required_encrypted_bitwidth(nodes)
    for node in nodes:
        assign_precisions_1_node(node, p, p)


def assign_precisions_1_node(node: Node, output_p: int, inputs_p: int):
    """Assign input/output precision to a single node.

    Precision are adjusted to match different use, e.g. encrypted and constant case.
    """
    assert isinstance(node.output.dtype, Integer)
    if node.output.is_encrypted:
        node.output.dtype.bit_width = output_p
    else:
        node.output.dtype.bit_width = output_p + 1

    for value in node.inputs:
        assert isinstance(value.dtype, Integer)
        if value.is_encrypted:
            value.dtype.bit_width = inputs_p
        else:
            value.dtype.bit_width = inputs_p + 1


CHUNKED_COMPARISON = {"greater", "greater_equal", "less", "less_equal"}
CHUNKED_COMPARISON_MIN_BITWIDTH = 4
MAX_POOLS = {"maxpool1d", "maxpool2d", "maxpool3d"}
ROUNDING = {"round_bit_pattern"}
MULTIPLY = {"multiply", "matmul", "dot"}


def max_encrypted_bitwidth_node(node: Node):
    """Give the minimal precision to implement the node.

    This applies to both input and output precisions.
    """
    assert isinstance(node.output.dtype, Integer)
    if node.output.is_encrypted or node.operation == Operation.Constant:
        normal_p = node.output.dtype.bit_width
    else:
        normal_p = -1
    name = node.properties.get("name")

    if name in CHUNKED_COMPARISON:
        return max(normal_p, CHUNKED_COMPARISON_MIN_BITWIDTH)

    if name in MAX_POOLS:
        return normal_p + 1

    if name in MULTIPLY and all(value.is_encrypted for value in node.inputs):
        # For operations that use multiply, an additional bit
        # needs to be added to the bitwidths of the inputs.
        # For single precision circuits the max of the input / output
        # precisions will be taken in required_encrypted_bitwidth. For
        # multi-precision, the circuit partitions will handle the
        # input and output precisions separately.
        all_inp_bitwidths = []
        # Need a loop here to allow typechecking and make mypy happy
        for inp in node.inputs:
            dtype_inp = inp.dtype
            assert isinstance(dtype_inp, Integer)
            all_inp_bitwidths.append(dtype_inp.bit_width)

        normal_p = max(all_inp_bitwidths)

        # FIXME: This probably does not work well with multi-precision!
        return max(normal_p + 1, node.output.dtype.bit_width)

    return normal_p


def required_encrypted_bitwidth(nodes: Iterable[Node]) -> int:
    """Give the minimal precision to implement all the nodes.

    This function is called for both single-precision (for the whole circuit)
    and for multi-precision circuits (for circuit partitions).

    Ops for which the compiler introduces TLUs need to be handled explicitly
    in `max_encrypted_bitwidth_node`. The maximum
    of all precisions of the various operations is returned.
    """

    bitwidths = map(max_encrypted_bitwidth_node, nodes)
    return max(bitwidths, default=-1)


def required_inputs_encrypted_bitwidth(graph, node, nodes_output_p: list[tuple[Node, int]]) -> int:
    """Give the minimal precision to supports the inputs."""
    preds = graph.ordered_preds_of(node)
    get_prec = lambda node: nodes_output_p[node.properties[NODE_ID]][1]
    # by definition all inputs have the same block precision
    # see uniform_precision_per_blocks
    return get_prec(node) if len(preds) == 0 else get_prec(preds[0])


def assign_multi_precision(graph, nodes):
    """Assign a specific encryption precision to each nodes."""
    add_nodes_id(nodes)
    nodes_output_p = uniform_precision_per_blocks(graph, nodes)
    for node, _ in nodes_output_p:
        node.properties["original_bit_width"] = node.output.dtype.bit_width
    nodes_inputs_p = [
        required_inputs_encrypted_bitwidth(graph, node, nodes_output_p)
        if can_change_precision(node)
        else output_p
        for node, output_p in nodes_output_p
    ]
    for (node, output_p), inputs_p in zip(nodes_output_p, nodes_inputs_p):
        assign_precisions_1_node(node, output_p, inputs_p)
    clear_nodes_id(nodes)


TLU_WITHOUT_PRECISION_CHANGE = CHUNKED_COMPARISON | MAX_POOLS | MULTIPLY


def can_change_precision(node):
    """Detect if a node completely ties inputs/output precisions together."""
    if (
        node.properties.get("name") in ROUNDING
        and node.properties["attributes"]["overflow_protection"]
    ):
        return False  # protection can change precision

    return (
        node.converted_to_table_lookup
        and node.properties.get("name") not in TLU_WITHOUT_PRECISION_CHANGE
    )


def convert_union_to_blocks(node_union: UnionFind) -> Iterable[list[int]]:
    """Convert a `UnionFind` to blocks.

    The result is an iterable of blocks.A block being a list of node id.
    """
    blocks = {}
    for node_id in range(node_union.size):
        node_canon = node_union.find_canonical(node_id)
        if node_canon == node_id:
            assert node_canon not in blocks
            blocks[node_canon] = [node_id]
        else:
            blocks[node_canon].append(node_id)
    return blocks.values()


NODE_ID = "node_id"


def add_nodes_id(nodes):
    """Temporarily add a NODE_ID property to all nodes."""
    for node_id, node in enumerate(nodes):
        assert NODE_ID not in node.properties
        node.properties[NODE_ID] = node_id


def clear_nodes_id(nodes):
    """Remove the NODE_ID property from all nodes."""
    for node in nodes:
        del node.properties[NODE_ID]


def uniform_precision_per_blocks(graph: Graph, nodes: list[Node]) -> list[tuple[Node, int]]:
    """Find the required precision of blocks and associate it corresponding nodes."""
    size = len(nodes)
    node_union = UnionFind(size)
    for node_id, node in enumerate(nodes):
        preds = graph.ordered_preds_of(node)
        if not preds:
            continue
        # we always unify all inputs
        first_input_id = preds[0].properties[NODE_ID]
        for pred in preds[1:]:
            pred_id = pred.properties[NODE_ID]
            node_union.union(first_input_id, pred_id)
        # we unify with outputs only if no precision change can occur
        if not can_change_precision(node):
            node_union.union(first_input_id, node_id)

    blocks = convert_union_to_blocks(node_union)
    result: list[None | tuple[Node, int]]
    result = [None] * len(nodes)
    for nodes_id in blocks:
        output_p = required_encrypted_bitwidth(nodes[node_id] for node_id in nodes_id)
        for node_id in nodes_id:
            result[node_id] = (nodes[node_id], output_p)
    assert None not in result
    return typing.cast("list[tuple[Node, int]]", result)


class UnionFind:
    """
    Utility class joins the nodes in equivalent precision classes.

    Nodes are just integers id.
    """

    parent: list[int]

    def __init__(self, size: int):
        """Create a union find suitable for `size` nodes."""
        self.parent = list(range(size))

    @property
    def size(self):
        """Size in number of nodes."""
        return len(self.parent)

    def find_canonical(self, a: int) -> int:
        """Find the current canonical node for a given input node."""
        parent = self.parent[a]
        if a == parent:
            return a
        canonical = self.find_canonical(parent)
        self.parent[a] = canonical
        return canonical

    def union(self, a: int, b: int):
        """Union both nodes."""
        self.united_common_ancestor(a, b)

    def united_common_ancestor(self, a: int, b: int) -> int:
        """Deduce the common ancestor of both nodes after unification."""
        parent_a = self.parent[a]
        parent_b = self.parent[b]

        if parent_a == parent_b:
            return parent_a

        if a == parent_a and parent_b < parent_a:
            common_ancestor = parent_b
        elif b == parent_b and parent_a < parent_b:
            common_ancestor = parent_a
        else:
            common_ancestor = self.united_common_ancestor(parent_a, parent_b)

        self.parent[a] = common_ancestor
        self.parent[b] = common_ancestor
        return common_ancestor
