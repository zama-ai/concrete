"""
Declaration of `AssignNodeIds` graph processor.
"""

from itertools import chain
from typing import Dict, List

import z3

from ...compilation.configuration import (
    BitwiseStrategy,
    ComparisonStrategy,
    MinMaxStrategy,
    MultivariateStrategy,
)
from ...dtypes import Integer
from ...representation import Graph, MultiGraphProcessor, Node, Operation


class AssignNodeIds(MultiGraphProcessor):
    """
    AssignNodeIds graph processor, to assign node id (%0, %1, etc.) to node properties.
    """

    def apply_many(self, graphs: Dict[str, Graph]):
        for graph_name, graph in graphs.items():
            for index, node in enumerate(graph.query_nodes(ordered=True)):
                node.properties["id"] = f"%{index}"
