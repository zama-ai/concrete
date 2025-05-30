"""
Declaration of `Graph` class.
"""

import math
import os
import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import networkx as nx
import numpy as np
import scipy.special
import z3

from ..dtypes import Float, Integer, UnsignedInteger
from .node import Node
from .operation import Operation

P_ERROR_PER_ERROR_SIZE_CACHE: dict[float, dict[int, float]] = {}


class Graph:
    """
    Graph class, to represent computation graphs.
    """

    graph: nx.MultiDiGraph

    input_nodes: dict[int, Node]
    output_nodes: dict[int, Node]

    input_indices: dict[Node, int]

    is_direct: bool

    bit_width_constraints: Optional[z3.Optimize]
    bit_width_assignments: Optional[z3.Model]

    name: str

    location: str

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        input_nodes: dict[int, Node],
        output_nodes: dict[int, Node],
        name: str,
        is_direct: bool = False,
        location: str = "",
    ):
        self.graph = graph

        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.input_indices = {node: index for index, node in input_nodes.items()}

        self.is_direct = is_direct

        self.bit_width_assignments = None
        self.bit_width_constraints = None

        self.name = name
        self.location = location

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
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        evaluation = self.evaluate(*args, p_error=p_error)
        result = tuple(evaluation[node] for node in self.ordered_outputs())
        return result if len(result) > 1 else result[0]

    def evaluate(
        self,
        *args: Any,
        p_error: Optional[float] = None,
    ) -> dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]]:
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

        node_results: dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}
        for node in nx.topological_sort(self.graph):
            if node.operation == Operation.Input:
                node_results[node] = node(args[self.input_indices[node]])
                continue

            pred_results = [deepcopy(node_results[pred]) for pred in self.ordered_preds_of(node)]

            if p_error > 0.0 and node.converted_to_table_lookup:  # pragma: no cover
                variable_input_indices = [
                    idx
                    for idx, pred in enumerate(self.ordered_preds_of(node))
                    if pred.operation != Operation.Constant
                ]

                for index in variable_input_indices:
                    pred_node = self.ordered_preds_of(node)[index]
                    if pred_node.operation != Operation.Input:
                        dtype = node.inputs[index].dtype
                        if isinstance(dtype, Integer):
                            # see https://github.com/zama-ai/concrete/blob/main/docs/_static/p_error_simulation.pdf  # noqa: E501  # pylint: disable=line-too-long
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

    def draw(
        self,
        *,
        horizontal: bool = False,
        save_to: Optional[Union[Path, str]] = None,
        show: bool = False,
    ) -> Path:  # pragma: no cover
        """
        Draw the graph.

        That this function requires the python `pygraphviz` package
        which itself requires the installation of `graphviz` packages

        (see https://pygraphviz.github.io/documentation/stable/install.html)

        Args:
            horizontal (bool, default = False):
                whether to draw horizontally

            save_to (Optional[Path], default = None):
                path to save the drawing
                a temporary file will be used if it's None

            show (bool, default = False):
                whether to show the drawing using matplotlib

        Returns:
            Path:
                path to the drawing
        """

        def get_color(node: Node):
            if node.operation == Operation.Input:
                return "blue"

            if node.conversion_have_table_lookup:
                return "red"

            return "black"

        def get_label(index: int, node: Node):
            result = f"{node.label()}\n"

            details = "" if node.output.shape == () else "tensor["
            details += f"{node.output.dtype}"
            if node.output.shape != ():
                details += f", {', '.join(str(element) for element in node.output.shape)}]"

            result += f"[%{index}: {details}]"

            return result

        graph = self.graph
        output_nodes = set(self.output_nodes.values())

        attributes = {
            node: {
                "label": get_label(i, node),
                "color": get_color(node),
                "penwidth": 2,  # double thickness for circles
                "peripheries": 2 if node in output_nodes else 1,  # two circles for output nodes
            }
            for i, node in enumerate(nx.lexicographical_topological_sort(graph))
        }
        nx.set_node_attributes(graph, attributes)

        for edge in graph.edges(keys=True):
            idx = graph.edges[edge]["input_idx"]
            graph.edges[edge]["label"] = f" {idx} "

        try:
            agraph = nx.nx_agraph.to_agraph(graph)
        except ImportError as error:  # pragma: no cover
            if "pygraphviz" in str(error):
                message = (
                    "Graph.draw requires `full` version of Concrete. "
                    "See https://docs.zama.ai/concrete/getting-started/installing."
                )
                raise ImportError(message) from error

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
            try:
                # pylint: disable=import-outside-toplevel
                import matplotlib.pyplot as plt
                from PIL import Image

                # pylint: enable=import-outside-toplevel
            except ImportError as error:
                if "matplotlib" in str(error) or "pillow" in str(error) or "PIL" in str(error):
                    message = (
                        "Graph.draw with show=True requires `full` version of Concrete. "
                        "See https://docs.zama.ai/concrete/getting-started/installing."
                    )
                    raise ImportError(message) from error

                raise

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
        highlighted_nodes: Optional[dict[Node, list[str]]] = None,
        highlighted_result: Optional[list[str]] = None,
        show_types: bool = True,
        show_bounds: bool = True,
        show_tags: bool = True,
        show_locations: bool = False,
        show_assigned_bit_widths: bool = False,
    ) -> str:
        """
        Get the textual representation of the `Graph`.

        Args:
            maximum_constant_length (int, default = 25):
                maximum length of formatted constants

            highlighted_nodes (Optional[Dict[Node, List[str]]], default = None):
                nodes to be highlighted and their corresponding messages

            highlighted_result (Optional[List[str]], default = None):
                messages corresponding to highlighted return line

            show_types (bool, default = True):
                whether to show types of nodes

            show_bounds (bool, default = True):
                whether to show bounds of nodes

            show_tags (bool, default = True):
                whether to show tags of nodes

            show_locations (bool, default = False):
                whether to show line information of nodes

            show_assigned_bit_widths (bool, default = False)
                whether to show assigned bit width of nodes instead of their original bit width

        Returns:
            str:
                textual representation of the `Graph`
        """

        # ruff: noqa: ERA001

        if self.is_direct:
            show_bounds = False

        # node -> identifier
        # e.g., id_map[node1] = 2
        # means line for node1 is in this form %2 = node1.format(...)
        id_map: dict[Node, int] = {}

        # lines that will be merged at the end
        lines: list[str] = []

        # metadata to add to each line
        # (for alignment, this is done after lines are determined)
        line_metadata: list[dict[str, str]] = []

        # default highlighted nodes is empty
        highlighted_nodes = highlighted_nodes if highlighted_nodes is not None else {}

        # highlight information for lines, this is required because highlights are added to lines
        # after their type information is added, and we only have line numbers, not nodes
        highlighted_lines: dict[int, list[str]] = {}

        # subgraphs to format after the main graph is formatted
        subgraphs: dict[str, Graph] = {}

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

                # pylint: disable=unidiomatic-typecheck
                assert type(lower) == type(upper)  # noqa: E721
                # pylint: enable=unidiomatic-typecheck

                if isinstance(lower, (float, np.float32, np.float64)):
                    bounds += f"{round(lower, 6)}, {round(upper, 6)}"
                else:
                    bounds += f"{int(lower)}, {int(upper)}"

                bounds += "]"

            output_value = node.output
            if (
                not show_assigned_bit_widths
                and isinstance(output_value.dtype, Integer)
                and ("original_bit_width" in node.properties or "bit_width_hint" in node.properties)
            ):
                output_value = deepcopy(output_value)
                output_value.dtype.bit_width = max(
                    node.properties.get("original_bit_width", -1),
                    node.properties.get("bit_width_hint", -1),
                )

            # remember metadata of the node
            line_metadata.append(
                {
                    "type": f"# {output_value}",
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

        # add return information
        # (if there is a single return, it's in the form `return %id`
        # (otherwise, it's in the form `return (%id1, %id2, ..., %idN)`
        returns: list[str] = []
        for node in self.ordered_outputs():
            returns.append(f"%{id_map[node]}")
        lines.append(f"return {', '.join(returns)}")
        if highlighted_result:  # pragma: no cover
            highlighted_lines[len(lines) - 1] = highlighted_result

        # strip whitespaces
        lines = [line.rstrip() for line in lines]

        # add highlights (this is done in reverse to keep indices consistent)
        for i in reversed(range(len(lines))):
            if i in highlighted_lines:
                for j, message in enumerate(highlighted_lines[i]):
                    highlight = "^" if j == 0 else " "
                    lines.insert(i + 1 + j, f"{highlight * len(lines[i])} {message}")

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

    def format_bit_width_constraints(self) -> str:
        """
        Get the textual representation of bit width constraints of the graph.

        Returns:
            str:
                textual representation of bit width constraints of the graph
        """

        result = ""
        for i, node in enumerate(nx.lexicographical_topological_sort(self.graph)):
            if len(node.bit_width_constraints) > 0:
                result += f"%{i}:\n"
                for constraint in node.bit_width_constraints:
                    result += f"    {constraint.arg(0)} {constraint.decl()} {constraint.arg(1)}\n"
        return result[:-1]

    def format_bit_width_assignments(self) -> str:
        """
        Get the textual representation of bit width assignments of the graph.

        Returns:
            str:
                textual representation of bit width assignments of the graph
        """

        lines = []
        for variable in self.bit_width_assignments.decls():  # type: ignore
            if variable.name().startswith(f"{self.name}.") or variable.name() == "input_output":
                width = self.bit_width_assignments.get_interp(variable)  # type: ignore
                lines.append(f"{variable} = {width}")

        def sorter(line: str) -> int:
            if line.startswith(f"{self.name}.max"):  # pragma: no cover
                # we won't have 4 million nodes...
                return 2**32
            if line.startswith("input_output"):  # pragma: no cover
                # this is the composable constraint
                return 2**32

            equals_position = line.find("=")
            index = line[len(self.name) + 2 : equals_position - 1]
            return int(index)

        result = ""

        longest_length_before_equals_sign = max(len(line.split("=")[0]) for line in lines)
        for line in sorted(lines, key=sorter):
            length_before_equals_sign = len(line.split("=")[0])
            result += (
                (" " * (longest_length_before_equals_sign - length_before_equals_sign))
                + line
                + "\n"
            )

        return result[:-1]

    def measure_bounds(
        self,
        inputset: Union[Iterable[Any], Iterable[tuple[Any, ...]]],
    ) -> dict[Node, dict[str, Union[np.integer, np.floating]]]:
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

    def update_with_bounds(self, bounds: dict[Node, dict[str, Union[np.integer, np.floating]]]):
        """
        Update `ValueDescription`s within the `Graph` according to measured bounds.

        Args:
            bounds (Dict[Node, Dict[str, Union[np.integer, np.floating]]]):
                bounds of each node in the `Graph`
        """

        for node in self.query_nodes(ordered=True):
            if node in bounds:
                min_bound = bounds[node]["min"]
                max_bound = bounds[node]["max"]

                node.bounds = (min_bound, max_bound)  # type: ignore

                new_value = deepcopy(node.output)

                if isinstance(min_bound, (np.integer, int)):
                    assert isinstance(new_value.dtype, Integer)
                    new_value.dtype.update_to_represent(np.array([min_bound, max_bound]))

                    if node.operation == Operation.Generic and node.properties["name"] in {
                        "amin",
                        "amax",
                        "min",
                        "max",
                    }:
                        assert isinstance(node.inputs[0].dtype, Integer)
                        new_value.dtype.is_signed = node.inputs[0].dtype.is_signed
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

    def ordered_inputs(self) -> list[Node]:
        """
        Get the input nodes of the `Graph`, ordered by their indices.

        Returns:
            List[Node]:
                ordered input nodes
        """

        return [self.input_nodes[idx] for idx in range(len(self.input_nodes))]

    def ordered_outputs(self) -> list[Node]:
        """
        Get the output nodes of the `Graph`, ordered by their indices.

        Returns:
            List[Node]:
                ordered output nodes
        """

        return [self.output_nodes[idx] for idx in range(len(self.output_nodes))]

    def ordered_preds_of(self, node: Node) -> list[Node]:
        """
        Get predecessors of `node`, ordered by their indices.

        Args:
            node (Node):
                node whose predecessors are requested

        Returns:
            List[Node]:
                ordered predecessors of `node`.
        """

        idx_to_pred: dict[int, Node] = {}
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            idx_to_pred.update((data["input_idx"], pred) for data in edge_data.values())
        return [idx_to_pred[i] for i in range(len(idx_to_pred))]

    def prune_useless_nodes(self):
        """
        Remove unreachable nodes from the graph.
        """
        outputs = self.ordered_outputs()
        used = nx.ancestors(self.graph, outputs[0])
        for output in outputs[1:]:
            used.update(nx.ancestors(self.graph, output))
        used.update(outputs)
        if len(used) == len(self.graph):
            return
        # unused in original order: facilitates graph format based tests
        # copy to avoid modifying the graph while iterating on it
        unused = [node for node in self.graph if node not in used]
        self.graph.remove_nodes_from(unused)

    def query_nodes(
        self,
        tag_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        is_encrypted_filter: Optional[bool] = None,
        custom_filter: Optional[Callable[[Node], bool]] = None,
        ordered: bool = False,
    ) -> list[Node]:
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

            is_encrypted_filter (Optional[bool], default = None)
                filter for encryption status

            custom_filter (Optional[Callable[[Node], bool]], default = None):
                flexible filter

            ordered (bool)
                whether to apply topological sorting before filtering nodes

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

        def match_boolean_filter(boolean_filter, boolean):
            if boolean_filter is None:
                return True

            return boolean == boolean_filter

        def get_operation_name(node):
            result: str

            if node.operation == Operation.Input:
                result = "input"
            elif node.operation == Operation.Constant:
                result = "constant"
            else:
                result = node.properties["name"]

            return result

        nodes = nx.lexicographical_topological_sort(self.graph) if ordered else self.graph.nodes()
        return [
            node
            for node in nodes
            if (
                match_text_filter(tag_filter, node.tag)
                and match_text_filter(operation_filter, get_operation_name(node))
                and match_boolean_filter(is_encrypted_filter, node.output.is_encrypted)
                and (custom_filter is None or custom_filter(node))
            )
        ]

    def maximum_integer_bit_width(
        self,
        tag_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        is_encrypted_filter: Optional[bool] = None,
        custom_filter: Optional[Callable[[Node], bool]] = None,
        assigned_bit_width: bool = False,
    ) -> int:
        """
        Get maximum integer bit-width within the graph.

        Only nodes after filtering will be used to calculate the result.

        Args:
            tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for tags

            operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for operations

            is_encrypted_filter (Optional[bool], default = None):
                filter for encryption status

            custom_filter (Optional[Callable[[Node], bool]], default = None):
                flexible filter

            assigned_bit_width (Optional[bool], default = None):
                whether to query on assigned bit-widths

        Returns:
            int:
                maximum integer bit-width within the graph
                if there are no integer nodes matching the query, result is -1
        """

        query = self.query_nodes(tag_filter, operation_filter, is_encrypted_filter, custom_filter)
        filtered_bit_widths = (
            (
                node.output.dtype.bit_width
                if assigned_bit_width
                else max(
                    node.properties.get("original_bit_width", node.output.dtype.bit_width),
                    node.properties.get("bit_width_hint", -1),
                )
            )
            for node in query
            if isinstance(node.output.dtype, Integer)
        )
        return max(filtered_bit_widths, default=-1)

    def integer_range(
        self,
        tag_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        is_encrypted_filter: Optional[bool] = None,
        custom_filter: Optional[Callable[[Node], bool]] = None,
    ) -> Optional[tuple[int, int]]:
        """
        Get integer range of the graph.

        Only nodes after filtering will be used to calculate the result.

        Args:
            tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for tags

            operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):
                filter for operations

            is_encrypted_filter (Optional[bool], default = None)
                filter for encryption status

            custom_filter (Optional[Callable[[Node], bool]], default = None):
                flexible filter

        Returns:
            Optional[Tuple[int, int]]:
                minimum and maximum integer value observed during inputset evaluation
                if there are no integer nodes matching the query, result is None
        """

        if self.is_direct:
            return None

        result: Optional[tuple[int, int]] = None

        query = self.query_nodes(tag_filter, operation_filter, is_encrypted_filter, custom_filter)
        filtered_bounds = (
            node.bounds
            for node in query
            if isinstance(node.output.dtype, Integer) and node.bounds is not None
        )
        for min_bound, max_bound in filtered_bounds:
            assert isinstance(min_bound, np.integer) and isinstance(max_bound, np.integer)

            if result is None:
                result = (int(min_bound), int(max_bound))
            else:
                old_min_bound, old_max_bound = result
                result = (
                    min(old_min_bound, int(min_bound)),
                    max(old_max_bound, int(max_bound)),
                )

        return result

    @property
    def inputs_count(self) -> int:
        """
        Returns the number of inputs of the graph.
        """
        return len(self.input_nodes)

    @property
    def outputs_count(self) -> int:
        """
        Returns the number of outputs of the graph.
        """
        return len(self.output_nodes)


class GraphProcessor(ABC):
    """
    GraphProcessor base class, to define the API for a graph processing pipeline.

    Process a single graph.
    """

    @abstractmethod
    def apply(self, graph: Graph):
        """
        Process the graph.
        """

    @staticmethod
    def error(graph: Graph, highlights: Mapping[Node, Union[str, list[str]]]):
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


class MultiGraphProcessor(GraphProcessor):
    """
    MultiGraphProcessor base class, to define the API for a multiple graph processing pipeline.

    Processes multiple graphs at once.
    """

    @abstractmethod
    def apply_many(self, graphs: dict[str, Graph]):
        """
        Process a dictionary of graphs.
        """

    def apply(self, graph: Graph):
        """
        Process a single graph.
        """
        return self.apply_many({graph.name: graph})  # pragma: no cover
