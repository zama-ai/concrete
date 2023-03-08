"""
Declaration of `Graph` class.
"""

import math
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy.special

from ..dtypes import Float, Integer, UnsignedInteger
from .node import Node
from .operation import Operation

P_ERROR_PER_ERROR_SIZE_CACHE: Dict[float, Dict[int, float]] = {}


class Graph:
    """
    Graph class, to represent computation graphs.
    """

    graph: nx.MultiDiGraph

    input_nodes: Dict[int, Node]
    output_nodes: Dict[int, Node]

    input_indices: Dict[Node, int]

    is_direct: bool

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        input_nodes: Dict[int, Node],
        output_nodes: Dict[int, Node],
        is_direct: bool = False,
    ):
        self.graph = graph

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.input_indices = {node: index for index, node in input_nodes.items()}

        self.is_direct = is_direct

        self.prune_useless_nodes()

    def __call__(
        self,
        *args: Any,
        p_error: Optional[float] = None,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        Tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        evaluation = self.evaluate(*args, p_error=p_error)
        result = tuple(evaluation[node] for node in self.ordered_outputs())
        return result if len(result) > 1 else result[0]

    def evaluate(
        self,
        *args: Any,
        p_error: Optional[float] = None,
    ) -> Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]]:
        r"""
        Perform the computation `Graph` represents and get resulting values for all nodes.

        Args:
            *args (List[Any]):
                inputs to the computation

            p_error (Optional[float]):
                probability of error for table lookups

        Returns:
            Dict[Node, Union[np.bool\_, np.integer, np.floating, np.ndarray]]:
                nodes and their values during computation
        """

        # pylint: disable=no-member,too-many-nested-blocks,too-many-branches,too-many-statements

        if p_error is None:
            p_error = 0.0

        assert isinstance(p_error, float)

        node_results: Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}
        for node in nx.topological_sort(self.graph):
            if node.operation == Operation.Input:
                node_results[node] = node(args[self.input_indices[node]])
                continue

            pred_results = [node_results[pred] for pred in self.ordered_preds_of(node)]

            if p_error > 0.0 and node.converted_to_table_lookup:
                variable_input_indices = [
                    idx
                    for idx, pred in enumerate(self.ordered_preds_of(node))
                    if not pred.operation == Operation.Constant
                ]

                for index in variable_input_indices:
                    pred_node = self.ordered_preds_of(node)[index]
                    if pred_node.operation != Operation.Input:
                        dtype = node.inputs[index].dtype
                        if isinstance(dtype, Integer):
                            # see https://github.com/zama-ai/concrete-numpy/blob/main/docs/_static/p_error_simulation.pdf  # noqa: E501  # pylint: disable=line-too-long
                            # to learn more about the distribution of error

                            if p_error not in P_ERROR_PER_ERROR_SIZE_CACHE:
                                std_score = math.sqrt(2) * scipy.special.erfcinv(p_error)
                                p_error_per_error_size = {}

                                error_size = 1
                                last_p = 1 - p_error
                                while last_p != 1.0 or error_size == 1:
                                    new_std_score = (2 * error_size + 1) * std_score
                                    new_p = scipy.special.erf(new_std_score / math.sqrt(2))

                                    p_error_per_error_size[error_size] = new_p - last_p

                                    last_p = new_p
                                    error_size += 1

                                # ordering of `p_error_per_error_size` is relied on
                                # during the introduction of the error below
                                # thus we explicitly sort it to make sure it's ordered
                                p_error_per_error_size = dict(
                                    sorted(p_error_per_error_size.items())
                                )

                                P_ERROR_PER_ERROR_SIZE_CACHE[p_error] = p_error_per_error_size
                            else:  # pragma: no cover
                                p_error_per_error_size = P_ERROR_PER_ERROR_SIZE_CACHE[p_error]

                            error = np.random.rand(*pred_results[index].shape)

                            accumulated_p_error = 0.0
                            for error_size, p_error_for_size in p_error_per_error_size.items():
                                accumulated_p_error += p_error_for_size
                                error = np.where(error < accumulated_p_error, error_size, error)

                            error = np.where(error < 1, 0, error).astype(np.int64)

                            error_sign = np.random.rand(*pred_results[index].shape)
                            error_sign = np.where(error_sign < 0.5, 1, -1).astype(np.int64)

                            new_result = pred_results[index] + (error * error_sign)

                            if new_result.shape == ():  # pragma: no cover
                                if new_result < dtype.min():
                                    new_result = dtype.max() - (dtype.min() - new_result) + 1
                                elif new_result > dtype.max():
                                    new_result = dtype.min() - (new_result - dtype.max()) - 1

                            else:
                                underflow_indices = np.where(new_result < dtype.min())
                                new_result[underflow_indices] = (
                                    dtype.max() - (dtype.min() - new_result[underflow_indices]) + 1
                                )

                                overflow_indices = np.where(new_result > dtype.max())
                                new_result[overflow_indices] = (
                                    dtype.min() + (new_result[overflow_indices] - dtype.max()) - 1
                                )

                            pred_results[index] = new_result

            try:
                node_results[node] = node(*pred_results)
            except Exception as error:
                raise RuntimeError(
                    "Evaluation of the graph failed\n\n"
                    + self.format(
                        highlighted_nodes={node: ["evaluation of this node failed"]},
                        show_bounds=False,
                    )
                ) from error

        return node_results

    def format(
        self,
        maximum_constant_length: int = 25,
        highlighted_nodes: Optional[Dict[Node, List[str]]] = None,
        show_types: bool = True,
        show_bounds: bool = True,
        show_tags: bool = True,
        show_locations: bool = False,
    ) -> str:
        """
        Get the textual representation of the `Graph`.

        Args:
            maximum_constant_length (int, default = 25):
                maximum length of formatted constants

            highlighted_nodes (Optional[Dict[Node, List[str]]], default = None):
                nodes to be highlighted and their corresponding messages

            show_types (bool, default = True):
                whether to show types of nodes

            show_bounds (bool, default = True):
                whether to show bounds of nodes

            show_tags (bool, default = True):
                whether to show tags of nodes

            show_locations (bool, default = False):
                whether to show line information of nodes

        Returns:
            str:
                textual representation of the `Graph`
        """

        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        # ruff: noqa: ERA001

        if self.is_direct:
            show_bounds = False

        # node -> identifier
        # e.g., id_map[node1] = 2
        # means line for node1 is in this form %2 = node1.format(...)
        id_map: Dict[Node, int] = {}

        # lines that will be merged at the end
        lines: List[str] = []

        # metadata to add to each line
        # (for alignment, this is done after lines are determined)
        line_metadata: List[Dict[str, str]] = []

        # default highlighted nodes is empty
        highlighted_nodes = highlighted_nodes if highlighted_nodes is not None else {}

        # highlight information for lines, this is required because highlights are added to lines
        # after their type information is added, and we only have line numbers, not nodes
        highlighted_lines: Dict[int, List[str]] = {}

        # subgraphs to format after the main graph is formatted
        subgraphs: Dict[str, Graph] = {}

        # format nodes
        for node in nx.lexicographical_topological_sort(self.graph):
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

            # get formatted bounds
            bounds = ""
            if node.bounds is not None:
                bounds += "∈ ["

                lower, upper = node.bounds
                assert type(lower) == type(upper)  # pylint: disable=unidiomatic-typecheck

                if isinstance(lower, (float, np.float32, np.float64)):
                    bounds += f"{round(lower, 6)}, {round(upper, 6)}"
                else:
                    bounds += f"{int(lower)}, {int(upper)}"

                bounds += "]"

            # remember metadata of the node
            line_metadata.append(
                {
                    "type": f"# {node.output}",
                    "bounds": bounds,
                    "tag": (f"@ {node.tag}" if node.tag != "" else ""),
                    "location": node.location,
                },
            )

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

        # determine which metadata to show
        shown_metadata_keys = []
        if show_types:
            shown_metadata_keys.append("type")
        if show_bounds:
            shown_metadata_keys.append("bounds")
        if show_tags:
            shown_metadata_keys.append("tag")
        if show_locations:
            shown_metadata_keys.append("location")

        # show requested metadata
        indent = 8
        for metadata_key in shown_metadata_keys:
            longest_line_length = max(len(line) for line in lines)
            lines = [
                line + (" " * ((longest_line_length - len(line)) + indent)) + metadata[metadata_key]
                for line, metadata in zip(lines, line_metadata)
            ]

        # strip whitespaces
        lines = [line.rstrip() for line in lines]

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
                subgraph_lines = subgraph.format(
                    maximum_constant_length=maximum_constant_length,
                    highlighted_nodes={},
                    show_types=show_types,
                    show_bounds=False,  # doesn't make sense as we don't measure bounds in subgraphs
                    show_tags=show_tags,
                    show_locations=show_locations,
                ).split("\n")

                result += "\n\n"
                result += f"    {line}:\n\n"
                result += "\n".join(f"        {line}" for line in subgraph_lines)

        return result

        # pylint: enable=too-many-branches,too-many-locals,too-many-statements

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

        index = 0
        try:
            evaluation = self.evaluate(*sample)
            for node, value in evaluation.items():
                bounds[node] = {
                    "min": value.min(),
                    "max": value.max(),
                }

            for sample in inputset_iterator:
                index += 1
                if not isinstance(sample, tuple):
                    sample = (sample,)

                evaluation = self.evaluate(*sample)
                for node, value in evaluation.items():
                    bounds[node] = {
                        "min": np.minimum(bounds[node]["min"], value.min()),
                        "max": np.maximum(bounds[node]["max"], value.max()),
                    }

        except Exception as error:
            message = f"Bound measurement using inputset[{index}] failed"
            raise RuntimeError(message) from error

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

                node.bounds = (min_bound, max_bound)

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

    def query_nodes(
        self,
        tag_filter: Optional[Union[str, List[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, List[str], re.Pattern]] = None,
    ) -> List[Node]:
        """
        Query nodes within the graph.

        Filters work like so:
            str -> nodes without exact match is skipped
            List[str] -> nodes without exact match with one of the strings in the list is skipped
            re.Pattern -> nodes without pattern match is skipped

        Args:
            tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for tags

            operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for operations

        Returns:
            List[Node]:
                filtered nodes
        """

        def match_text_filter(text_filter, text):
            if text_filter is None:
                return True

            if isinstance(text_filter, str):
                return text == text_filter

            if isinstance(text_filter, re.Pattern):
                return text_filter.match(text)

            return any(text == alternative for alternative in text_filter)

        def get_operation_name(node):
            result: str

            if node.operation == Operation.Input:
                result = "input"
            elif node.operation == Operation.Constant:
                result = "constant"
            else:
                result = node.properties["name"]

            return result

        return [
            node
            for node in self.graph.nodes()
            if (
                match_text_filter(tag_filter, node.tag)
                and match_text_filter(operation_filter, get_operation_name(node))
            )
        ]

    def maximum_integer_bit_width(
        self,
        tag_filter: Optional[Union[str, List[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, List[str], re.Pattern]] = None,
    ) -> int:
        """
        Get maximum integer bit-width within the graph.

        Only nodes after filtering will be used to calculate the result.

        Args:
            tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for tags

            operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for operations

        Returns:
            int:
                maximum integer bit-width within the graph
                if there are no integer nodes matching the query, result is -1
        """

        filtered_bit_widths = (
            node.output.dtype.bit_width
            for node in self.query_nodes(tag_filter, operation_filter)
            if isinstance(node.output.dtype, Integer)
        )
        return max(filtered_bit_widths, default=-1)

    def integer_range(
        self,
        tag_filter: Optional[Union[str, List[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, List[str], re.Pattern]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Get integer range of the graph.

        Only nodes after filtering will be used to calculate the result.

        Args:
            tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for tags

            operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for operations

        Returns:
            Optional[Tuple[int, int]]:
                minimum and maximum integer value observed during inputset evaluation
                if there are no integer nodes matching the query, result is None
        """

        result: Optional[Tuple[int, int]] = None

        if not self.is_direct:
            filtered_bounds = (
                node.bounds
                for node in self.query_nodes(tag_filter, operation_filter)
                if isinstance(node.output.dtype, Integer) and node.bounds is not None
            )
            for min_bound, max_bound in filtered_bounds:
                assert isinstance(min_bound, np.integer) and isinstance(max_bound, np.integer)

                if result is None:
                    result = (int(min_bound), int(max_bound))
                else:
                    old_min_bound, old_max_bound = result  # pylint: disable=unpacking-non-sequence
                    result = (
                        min(old_min_bound, int(min_bound)),
                        max(old_max_bound, int(max_bound)),
                    )

        return result
