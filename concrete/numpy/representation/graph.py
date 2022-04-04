"""
Declaration of `Graph` class.
"""

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

from ..dtypes import Float, Integer, UnsignedInteger
from .node import Node
from .operation import OPERATION_COLOR_MAPPING, Operation


class Graph:
    """
    Graph class, to represent computation graphs.
    """

    graph: nx.MultiDiGraph

    input_nodes: Dict[int, Node]
    output_nodes: Dict[int, Node]

    input_indices: Dict[Node, int]

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        input_nodes: Dict[int, Node],
        output_nodes: Dict[int, Node],
    ):
        self.graph = graph

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.input_indices = {node: index for index, node in input_nodes.items()}

        self.prune_useless_nodes()

    def __call__(
        self,
        *args: Any,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        Tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        evaluation = self.evaluate(*args)
        result = tuple(evaluation[node] for node in self.ordered_outputs())
        return result if len(result) > 1 else result[0]

    def evaluate(
        self,
        *args: Any,
    ) -> Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]]:
        r"""
        Perform the computation `Graph` represents and get resulting values for all nodes.

        Args:
            *args (List[Any]):
                inputs to the computation

        Returns:
            Dict[Node, Union[np.bool\_, np.integer, np.floating, np.ndarray]]:
                nodes and their values during computation
        """

        node_results: Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}
        for node in nx.topological_sort(self.graph):
            if node.operation == Operation.Input:
                node_results[node] = node(args[self.input_indices[node]])
                continue

            pred_results = [node_results[pred] for pred in self.ordered_preds_of(node)]
            node_results[node] = node(*pred_results)
        return node_results

    def draw(
        self,
        show: bool = False,
        horizontal: bool = False,
        save_to: Optional[Union[Path, str]] = None,
    ) -> Path:
        """
        Draw the `Graph` and optionally save/show the drawing.

        note that this function requires the python `pygraphviz` package
        which itself requires the installation of `graphviz` packages
        see https://pygraphviz.github.io/documentation/stable/install.html

        Args:
            show (bool, default = False):
                whether to show the drawing using matplotlib or not

            horizontal (bool, default = False):
                whether to draw horizontally or not

            save_to (Optional[Path], default = None):
                path to save the drawing
                a temporary file will be used if it's None

        Returns:
            Path:
                path to the saved drawing
        """

        def get_color(node, output_nodes):
            if node in output_nodes:
                return OPERATION_COLOR_MAPPING["output"]
            return OPERATION_COLOR_MAPPING[node.operation]

        graph = self.graph
        output_nodes = set(self.output_nodes.values())

        attributes = {
            node: {
                "label": node.label(),
                "color": get_color(node, output_nodes),
                "penwidth": 2,  # double thickness for circles
                "peripheries": 2 if node in output_nodes else 1,  # two circles for output nodes
            }
            for node in graph.nodes
        }
        nx.set_node_attributes(graph, attributes)

        for edge in graph.edges(keys=True):
            idx = graph.edges[edge]["input_idx"]
            graph.edges[edge]["label"] = f" {idx} "

        try:
            agraph = nx.nx_agraph.to_agraph(graph)
        except ImportError as error:  # pragma: no cover
            if "pygraphviz" in str(error):
                raise ImportError(
                    "Graph.draw requires pygraphviz. Install graphviz distribution to your OS "
                    "following https://pygraphviz.github.io/documentation/stable/install.html "
                    "and reinstall concrete-numpy with extras: `pip install --force-reinstall "
                    "concrete-numpy[full]`"
                ) from error

            raise

        agraph.graph_attr["rankdir"] = "LR" if horizontal else "TB"
        agraph.layout("dot")

        if save_to is None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                # we need to change the permissions of the temporary file
                # so that it can be read by all users

                # (https://stackoverflow.com/a/44130605)

                # get the old umask and replace it with 0o666
                old_umask = os.umask(0o666)

                # restore the old umask back
                os.umask(old_umask)

                # combine the old umask with the wanted permissions
                permissions = 0o666 & ~old_umask

                # set new permissions
                os.chmod(tmp.name, permissions)

                save_to_str = str(tmp.name)
        else:
            save_to_str = str(save_to)

        agraph.draw(save_to_str)

        if show:  # pragma: no cover
            plt.close("all")
            plt.figure()

            img = Image.open(save_to_str)
            plt.imshow(img)
            img.close()

            plt.axis("off")
            plt.show()

        return Path(save_to_str)

    def format(
        self,
        maximum_constant_length: int = 25,
        highlighted_nodes: Optional[Dict[Node, List[str]]] = None,
    ) -> str:
        """
        Get the textual representation of the `Graph`.

        Args:
            maximum_constant_length (int, default = 25):
                maximum length of formatted constants

            highlighted_nodes (Optional[Dict[Node, List[str]]], default = None):
                nodes to be highlighted and their corresponding messages

        Returns:
            str:
                textual representation of the `Graph`
        """

        # node -> identifier
        # e.g., id_map[node1] = 2
        # means line for node1 is in this form %2 = node1.format(...)
        id_map: Dict[Node, int] = {}

        # lines that will be merged at the end
        lines: List[str] = []

        # type information to add to each line
        # (for alingment, this is done after lines are determined)
        type_informations: List[str] = []

        # default highlighted nodes is empty
        highlighted_nodes = highlighted_nodes if highlighted_nodes is not None else {}

        # highlight information for lines, this is required because highlights are added to lines
        # after their type information is added, and we only have line numbers, not nodes
        highlighted_lines: Dict[int, List[str]] = {}

        # subgraphs to format after the main graph is formatted
        subgraphs: Dict[str, Graph] = {}

        # format nodes
        for node in nx.topological_sort(self.graph):
            # assign a unique id to outputs of node
            id_map[node] = len(id_map)

            # remember highlights of the node
            if node in highlighted_nodes:
                highlighted_lines[len(lines)] = highlighted_nodes[node]

            # extract predecessors and their ids
            predecessors = []
            for predecessor in self.ordered_preds_of(node):
                predecessors.append(f"%{id_map[predecessor]}")

            # start the build the line for the node
            line = ""

            # add output information to the line
            line += f"%{id_map[node]}"

            # add node information to the line
            line += " = "
            line += node.format(predecessors, maximum_constant_length)

            # append line to list of lines
            lines.append(line)

            # if exists, save the subgraph
            if node.operation == Operation.Generic and "subgraph" in node.properties["kwargs"]:
                subgraphs[line] = node.properties["kwargs"]["subgraph"]

            # remember type information of the node
            type_informations.append(str(node.output))

        # align = signs
        #
        # e.g.,
        #
        #  %1 = ...
        #  %2 = ...
        #  ...
        #  %8 = ...
        #  %9 = ...
        # %10 = ...
        # %11 = ...
        # ...
        longest_length_before_equals_sign = max(len(line.split("=")[0]) for line in lines)
        for i, line in enumerate(lines):
            length_before_equals_sign = len(line.split("=")[0])
            lines[i] = (
                " " * (longest_length_before_equals_sign - length_before_equals_sign)
            ) + line

        # add type information
        longest_line_length = max(len(line) for line in lines)
        for i, line in enumerate(lines):
            lines[i] += " " * (longest_line_length - len(line))
            lines[i] += f"        # {type_informations[i]}"

        # add highlights (this is done in reverse to keep indices consistent)
        for i in reversed(range(len(lines))):
            if i in highlighted_lines:
                for j, message in enumerate(highlighted_lines[i]):
                    highlight = "^" if j == 0 else " "
                    lines.insert(i + 1 + j, f"{highlight * len(lines[i])} {message}")

        # add return information
        # (if there is a single return, it's in the form `return %id`
        # (otherwise, it's in the form `return (%id1, %id2, ..., %idN)`
        returns: List[str] = []
        for node in self.output_nodes.values():
            returns.append(f"%{id_map[node]}")
        lines.append("return " + (returns[0] if len(returns) == 1 else f"({', '.join(returns)})"))

        # format subgraphs after the actual graph
        result = "\n".join(lines)
        if len(subgraphs) > 0:
            result += "\n\n"
            result += "Subgraphs:"
            for line, subgraph in subgraphs.items():
                subgraph_lines = subgraph.format(maximum_constant_length).split("\n")
                result += "\n\n"
                result += f"    {line}:\n\n"
                result += "\n".join(f"        {line}" for line in subgraph_lines)

        return result

    def measure_bounds(
        self,
        inputset: Union[Iterable[Any], Iterable[Tuple[Any, ...]]],
    ) -> Dict[Node, Dict[str, Union[np.integer, np.floating]]]:
        """
        Evaluate the `Graph` using an inputset and measure bounds.

        inputset is either an iterable of anything
        for a single parameter

        or

        an iterable of tuples of anything (of rank number of parameters)
        for multiple parameters

        e.g.,

        .. code-block:: python

            inputset = [1, 3, 5, 2, 4]
            def f(x):
                ...

            inputset = [(1, 2), (2, 4), (3, 1), (2, 2)]
            def g(x, y):
                ...

        Args:
            inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]]]):
                inputset to use

        Returns:
            Dict[Node, Dict[str, Union[np.integer, np.floating]]]:
                bounds of each node in the `Graph`
        """

        bounds = {}

        inputset_iterator = iter(inputset)

        sample = next(inputset_iterator)
        if not isinstance(sample, tuple):
            sample = (sample,)

        evaluation = self.evaluate(*sample)
        for node, value in evaluation.items():
            bounds[node] = {
                "min": value.min(),
                "max": value.max(),
            }

        for sample in inputset_iterator:
            if not isinstance(sample, tuple):
                sample = (sample,)

            evaluation = self.evaluate(*sample)
            for node, value in evaluation.items():
                bounds[node] = {
                    "min": np.minimum(bounds[node]["min"], value.min()),
                    "max": np.maximum(bounds[node]["max"], value.max()),
                }

        return bounds

    def update_with_bounds(self, bounds: Dict[Node, Dict[str, Union[np.integer, np.floating]]]):
        """
        Update `Value`s within the `Graph` according to measured bounds.

        Args:
            bounds (Dict[Node, Dict[str, Union[np.integer, np.floating]]]):
                bounds of each node in the `Graph`
        """

        for node in self.graph.nodes():
            if node in bounds:
                min_bound = bounds[node]["min"]
                max_bound = bounds[node]["max"]

                new_value = deepcopy(node.output)

                if isinstance(min_bound, np.integer):
                    new_value.dtype = Integer.that_can_represent(np.array([min_bound, max_bound]))
                else:
                    new_value.dtype = {
                        np.bool_: UnsignedInteger(1),
                        np.float64: Float(64),
                        np.float32: Float(32),
                        np.float16: Float(16),
                    }[type(min_bound)]

                node.output = new_value

                if node.operation == Operation.Input:
                    node.inputs[0] = new_value

                for successor in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, successor)
                    for edge in edge_data.values():
                        input_idx = edge["input_idx"]
                        successor.inputs[input_idx] = node.output

    def ordered_inputs(self) -> List[Node]:
        """
        Get the input nodes of the `Graph`, ordered by their indices.

        Returns:
            List[Node]:
                ordered input nodes
        """

        return [self.input_nodes[idx] for idx in range(len(self.input_nodes))]

    def ordered_outputs(self) -> List[Node]:
        """
        Get the output nodes of the `Graph`, ordered by their indices.

        Returns:
            List[Node]:
                ordered output nodes
        """

        return [self.output_nodes[idx] for idx in range(len(self.output_nodes))]

    def ordered_preds_of(self, node: Node) -> List[Node]:
        """
        Get predecessors of `node`, ordered by their indices.

        Args:
            node (Node):
                node whose predecessors are requested

        Returns:
            List[Node]:
                ordered predecessors of `node`.
        """

        idx_to_pred: Dict[int, Node] = {}
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            idx_to_pred.update((data["input_idx"], pred) for data in edge_data.values())
        return [idx_to_pred[i] for i in range(len(idx_to_pred))]

    def prune_useless_nodes(self):
        """
        Remove unreachable nodes from the graph.
        """

        useful_nodes: Dict[Node, None] = {}

        current_nodes = {node: None for node in self.ordered_outputs()}
        while current_nodes:
            useful_nodes.update(current_nodes)

            next_nodes: Dict[Node, None] = {}
            for node in current_nodes:
                next_nodes.update({node: None for node in self.graph.predecessors(node)})

            current_nodes = next_nodes

        useless_nodes = [node for node in self.graph.nodes() if node not in useful_nodes]
        self.graph.remove_nodes_from(useless_nodes)

    def maximum_integer_bit_width(self) -> int:
        """
        Get maximum integer bit-width within the graph.

        Returns:
            int:
                maximum integer bit-width within the graph (-1 is there are no integer nodes)
        """

        result = -1
        for node in self.graph.nodes():
            if isinstance(node.output.dtype, Integer):
                result = max(result, node.output.dtype.bit_width)
        return result
