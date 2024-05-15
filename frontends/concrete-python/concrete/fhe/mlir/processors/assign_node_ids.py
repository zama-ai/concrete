"""
Declaration of `AssignNodeIds` graph processor.
"""

from typing import Dict

from ...representation import Graph, MultiGraphProcessor


class AssignNodeIds(MultiGraphProcessor):
    """
    AssignNodeIds graph processor, to assign node id (%0, %1, etc.) to node properties.
    """

    def apply_many(self, graphs: Dict[str, Graph]):
        for graph in graphs.values():
            for index, node in enumerate(graph.query_nodes(ordered=True)):
                node.properties["id"] = f"%{index}"
