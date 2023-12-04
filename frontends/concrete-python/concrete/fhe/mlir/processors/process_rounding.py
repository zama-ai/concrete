"""
Declaration of `ProcessRounding` graph processor.
"""

from copy import deepcopy
from itertools import chain
from typing import Optional

import numpy as np

from ...dtypes import Integer
from ...extensions.table import LookupTable
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

            original_lsbs_to_remove = node.properties["kwargs"]["lsbs_to_remove"]
            final_lsbs_to_remove = node.properties["final_lsbs_to_remove"]

            if original_lsbs_to_remove != 0 and final_lsbs_to_remove == 0:
                self.replace_with_tlu(graph, node)
                continue

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

    def replace_with_tlu(self, graph: Graph, node: Node):
        """
        Replace rounding node with a TLU node that simulates rounding.

        This is a special case where:
        - user wanted to remove 1-bits
        - there was an overflow
        - overflow protection was on

        Let's say user wanted to remove 1-bit from 3-bits, but there was an overflow:
        - [0, 1, 2, 3, 4, 5, 6, 7] would be mapped to [0, 2, 2, 4, 4, 6, 6, 8]
        - or in the actual implementation [(0)00, (0)01, (0)01, (0)10, (0)10, (0)11, (0)11, (1)00]
        - (first bit is the padding bit, which is overwritten on overflows)
        - so the input is 3-bits and the output needs to be 3-bits to store the result
        - which can't be achieved with rounding
        - so we just replace the rounding with a TLU
        - using the table [0, 2, 2, 4, 4, 6, 6, 8]
        """

        preds = graph.ordered_preds_of(node)
        assert len(preds) == 1

        pred = preds[0]
        assert isinstance(pred.output.dtype, Integer)

        pred_dtype = pred.output.dtype
        pred_range = chain(range(0, pred_dtype.max() + 1), range(pred_dtype.min(), 0))

        def simulate_rounding(x, lsbs_to_remove):
            assert lsbs_to_remove != 0
            unit = 1 << lsbs_to_remove
            half = 1 << lsbs_to_remove - 1
            rounded = (x + half) // unit
            return rounded * unit

        original_lsbs_to_remove = node.properties["kwargs"]["lsbs_to_remove"]
        table = [
            simulate_rounding(value, lsbs_to_remove=original_lsbs_to_remove) for value in pred_range
        ]

        replacement = Node.generic(
            "tlu",
            [deepcopy(pred.output)],
            deepcopy(node.output),
            LookupTable.apply,
            kwargs={"table": np.array(table)},
        )
        replacement.properties["original_bit_width"] = node.properties["original_bit_width"]

        nx_graph = graph.graph

        edge_data = nx_graph.get_edge_data(pred, node).values()
        for data in list(edge_data):
            input_idx = data["input_idx"]
            nx_graph.add_edge(pred, replacement, input_idx=input_idx)
        nx_graph.remove_edge(pred, node)

        for successor in list(nx_graph.successors(node)):
            edge_data = nx_graph.get_edge_data(node, successor).values()
            for data in list(edge_data):
                input_idx = data["input_idx"]
                nx_graph.add_edge(replacement, successor, input_idx=input_idx)
            nx_graph.remove_edge(node, successor)

        for i, candidate in graph.output_nodes.items():
            if candidate is node:  # pragma: no cover
                graph.output_nodes[i] = replacement

        nx_graph.remove_node(node)

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
