"""
Declaration of `ProcessRounding` graph processor.
"""

from copy import deepcopy
from typing import Optional

from ...dtypes import Integer
from ...representation import Graph, Node
from . import GraphProcessor


class ProcessRounding(GraphProcessor):
    """
    ProcessRounding graph processor, to analyze rounding and support regular operations on it.
    """

    def apply(self, graph: Graph):
        rounding_nodes = graph.query_nodes(operation_filter="round_bit_pattern")
        for node in rounding_nodes:
            self.process_predecessors(graph, node)
            self.process_successors(graph, node)

    def process_predecessors(self, graph: Graph, node: Node):
        """
        Process predecessors of the rounding.
        """

        preds = graph.ordered_preds_of(node)
        assert len(preds) == 1

        pred = preds[0]
        assert isinstance(pred.output.dtype, Integer)

        overflow_protection = node.properties["attributes"]["overflow_protection"]
        overflow_detected = (
            overflow_protection
            and pred.properties["original_bit_width"] != node.properties["original_bit_width"]
        )

        node.properties["overflow_protection"] = overflow_protection
        node.properties["overflow_detected"] = overflow_detected

        original_bit_width = pred.properties["original_bit_width"]
        lsbs_to_remove = node.properties["kwargs"]["lsbs_to_remove"]

        msbs_to_keep = original_bit_width - lsbs_to_remove

        bit_width = pred.output.dtype.bit_width
        final_lsbs_to_remove = bit_width - msbs_to_keep

        if overflow_protection and overflow_detected:
            final_lsbs_to_remove -= 1

        node.properties["final_lsbs_to_remove"] = final_lsbs_to_remove
        node.properties["resulting_bit_width"] = bit_width - final_lsbs_to_remove

        node.properties["original_input_bit_width"] = original_bit_width

    def process_successors(self, graph: Graph, node: Node):
        """
        Process successors of the rounding.
        """

        identity: Optional[Node] = None

        def initialize(identity: Optional[Node]) -> Node:
            if identity is not None:
                return identity

            identity = Node.generic(
                "identity",
                [deepcopy(node.output)],
                deepcopy(node.output),
                lambda x: x,
            )
            identity.properties["original_bit_width"] = node.properties["original_bit_width"]

            nx_graph.add_edge(node, identity, input_idx=0)
            return identity

        nx_graph = graph.graph
        for successor in list(nx_graph.successors(node)):
            if successor.converted_to_table_lookup:
                continue

            edge_data = nx_graph.get_edge_data(node, successor).values()
            for data in list(edge_data):
                input_idx = data["input_idx"]
                nx_graph.remove_edge(node, successor)

                identity = initialize(identity)
                nx_graph.add_edge(identity, successor, input_idx=input_idx)

        for i, candidate in graph.output_nodes.items():
            if candidate is node:
                identity = initialize(identity)
                graph.output_nodes[i] = identity
