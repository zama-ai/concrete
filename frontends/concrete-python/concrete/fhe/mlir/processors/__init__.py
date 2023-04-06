"""
Declaration of `GraphProcessor` class.
"""

from abc import ABC, abstractmethod
from typing import List, Mapping, Union

from ...representation import Graph, Node


class GraphProcessor(ABC):
    """
    GraphProcessor base class, to define the API for a graph processing pipeline.
    """

    @abstractmethod
    def apply(self, graph: Graph):
        """
        Process the graph.
        """

    @staticmethod
    def error(graph: Graph, highlights: Mapping[Node, Union[str, List[str]]]):
        """
        Fail processing with an error.

        Args:
            graph (Graph):
                graph being processed

            highlights (Mapping[Node, Union[str, List[str]]]):
                nodes to highlight along with messages
        """

        highlights_with_location = {}
        for node, messages in highlights.items():
            messages_with_location = messages if isinstance(messages, list) else [messages]
            messages_with_location.append(node.location)
            highlights_with_location[node] = messages_with_location

        message = "Function you are trying to compile cannot be compiled\n\n" + graph.format(
            highlighted_nodes=highlights_with_location
        )
        raise RuntimeError(message)
