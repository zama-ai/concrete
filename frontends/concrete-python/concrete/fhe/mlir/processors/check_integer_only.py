"""
Declaration of `CheckIntegerOnly` graph processor.
"""

from ...dtypes import Integer
from ...representation import Graph
from . import GraphProcessor


class CheckIntegerOnly(GraphProcessor):
    """
    CheckIntegerOnly graph processor, to make sure the graph only contains integer nodes.
    """

    def apply(self, graph: Graph):
        non_integer_nodes = graph.query_nodes(
            custom_filter=(lambda node: not isinstance(node.output.dtype, Integer))
        )
        if non_integer_nodes:
            self.error(graph, {node: "only integers are supported" for node in non_integer_nodes})
