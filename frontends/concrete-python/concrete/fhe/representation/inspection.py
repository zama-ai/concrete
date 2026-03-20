"""
Declaration of `NodeSnapshot` and `InspectionResult` classes for interactive value inspection.
"""

import re
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import numpy as np

from .node import Node
from .operation import Operation


class NodeSnapshot:
    """
    Snapshot of one node's evaluation during inspection.
    """

    node: Node
    value: Union[np.bool_, np.integer, np.floating, np.ndarray]
    index: int
    overflow: bool
    overflow_min: Optional[int]
    overflow_max: Optional[int]

    def __init__(
        self,
        node: Node,
        value: Union[np.bool_, np.integer, np.floating, np.ndarray],
        index: int,
    ):
        self.node = node
        self.value = value
        self.index = index

        # Overflow detection: check if value exceeds the node's output dtype range
        self.overflow = False
        self.overflow_min = None
        self.overflow_max = None

        from ..dtypes import Integer

        if isinstance(node.output.dtype, Integer):
            dtype_min = node.output.dtype.min()
            dtype_max = node.output.dtype.max()

            if isinstance(value, np.ndarray):
                val_min = int(value.min())
                val_max = int(value.max())
            else:
                val_min = int(value)
                val_max = int(value)

            if val_min < dtype_min or val_max > dtype_max:
                self.overflow = True
                self.overflow_min = val_min
                self.overflow_max = val_max

    @property
    def location(self) -> str:
        """Get the source location of the node."""
        return self.node.location

    @property
    def tag(self) -> str:
        """Get the tag of the node."""
        return self.node.tag

    @property
    def operation_name(self) -> str:
        """Get the operation name of the node."""
        if self.node.operation == Operation.Input:
            return "input"
        if self.node.operation == Operation.Constant:
            return "constant"
        return self.node.properties["name"]

    @property
    def is_encrypted(self) -> bool:
        """Get whether the node output is encrypted."""
        return self.node.output.is_encrypted

    @property
    def exceeds_bounds(self) -> bool:
        """Check if the value is outside the node's measured bounds (softer than overflow)."""
        if self.node.bounds is None:
            return False

        lower, upper = self.node.bounds

        if isinstance(self.value, np.ndarray):
            val_min = self.value.min()
            val_max = self.value.max()
        else:
            val_min = self.value
            val_max = self.value

        return val_min < lower or val_max > upper

    def __repr__(self) -> str:
        return (
            f"NodeSnapshot(index={self.index}, op={self.operation_name}, "
            f"value={_format_value_short(self.value)}, overflow={self.overflow})"
        )


class InspectionResult:
    """
    Collection of node snapshots from a graph inspection, with query and display methods.
    """

    _snapshots: list[NodeSnapshot]
    _graph: Any  # Graph (avoid circular import in type hint)
    _stopped: bool
    _stopped_at_node: Optional[Node]

    def __init__(
        self,
        snapshots: list[NodeSnapshot],
        graph: Any,
        stopped: bool = False,
        stopped_at_node: Optional[Node] = None,
    ):
        self._snapshots = snapshots
        self._graph = graph
        self._stopped = stopped
        self._stopped_at_node = stopped_at_node

    def __iter__(self):
        return iter(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, index: int) -> NodeSnapshot:
        return self._snapshots[index]

    def __repr__(self) -> str:
        overflow_count = sum(1 for s in self._snapshots if s.overflow)
        stopped_str = f", stopped=True" if self._stopped else ""
        return (
            f"InspectionResult(nodes={len(self._snapshots)}, "
            f"overflows={overflow_count}{stopped_str})"
        )

    def filter(
        self,
        tag_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        is_encrypted_filter: Optional[bool] = None,
        location_filter: Optional[Union[str, re.Pattern]] = None,
        custom_filter: Optional[Callable[[NodeSnapshot], bool]] = None,
        overflow_only: bool = False,
    ) -> list[NodeSnapshot]:
        """
        Filter snapshots by various criteria.

        Args:
            tag_filter: filter by node tag
            operation_filter: filter by operation name
            is_encrypted_filter: filter by encryption status
            location_filter: filter by source location
            custom_filter: arbitrary predicate on NodeSnapshot
            overflow_only: if True, only return snapshots with overflow

        Returns:
            list of matching NodeSnapshot objects
        """

        def match_text(text_filter, text):
            if text_filter is None:
                return True
            if isinstance(text_filter, str):
                return text == text_filter
            if isinstance(text_filter, re.Pattern):
                return text_filter.match(text) is not None
            return any(text == alt for alt in text_filter)

        results = []
        for snap in self._snapshots:
            if not match_text(tag_filter, snap.tag):
                continue
            if not match_text(operation_filter, snap.operation_name):
                continue
            if is_encrypted_filter is not None and snap.is_encrypted != is_encrypted_filter:
                continue
            if location_filter is not None and not match_text(location_filter, snap.location):
                continue
            if custom_filter is not None and not custom_filter(snap):
                continue
            if overflow_only and not snap.overflow:
                continue
            results.append(snap)

        return results

    @property
    def overflows(self) -> list[NodeSnapshot]:
        """Get all snapshots that have overflow."""
        return self.filter(overflow_only=True)

    @property
    def has_overflow(self) -> bool:
        """Check if any node has overflow."""
        return any(s.overflow for s in self._snapshots)

    @property
    def output(
        self,
    ) -> Union[
        np.bool_,
        np.integer,
        np.floating,
        np.ndarray,
        tuple[Union[np.bool_, np.integer, np.floating, np.ndarray], ...],
    ]:
        """
        Get the final output value(s) of the inspected graph.

        Raises:
            RuntimeError: if evaluation was stopped early via stop_at
        """
        if self._stopped:
            raise RuntimeError(
                "Cannot retrieve output: inspection was stopped early "
                f"(stopped before node at {self._stopped_at_node.location if self._stopped_at_node else 'unknown'})"
            )

        node_to_snapshot = {s.node: s for s in self._snapshots}
        outputs = []
        for node in self._graph.ordered_outputs():
            if node not in node_to_snapshot:
                raise RuntimeError(
                    "Cannot retrieve output: output node was not evaluated"
                )
            outputs.append(node_to_snapshot[node].value)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def summary(
        self,
        show_values: bool = True,
        show_overflow: bool = True,
        maximum_constant_length: int = 25,
    ) -> str:
        """
        Build a formatted summary table of the inspection.

        Args:
            show_values: whether to show computed values
            show_overflow: whether to show overflow status
            maximum_constant_length: max length for constant formatting

        Returns:
            formatted string summary
        """
        if len(self._snapshots) == 0:
            return "(empty graph)"

        import networkx as nx

        # Build id_map and node_to_snapshot
        id_map: dict[Node, int] = {}
        node_to_snapshot: dict[Node, NodeSnapshot] = {s.node: s for s in self._snapshots}

        for node in nx.lexicographical_topological_sort(self._graph.graph):
            id_map[node] = len(id_map)

        lines: list[str] = []
        extra_columns: list[dict[str, str]] = []

        for node in nx.lexicographical_topological_sort(self._graph.graph):
            predecessors = []
            for pred in self._graph.ordered_preds_of(node):
                predecessors.append(f"%{id_map[pred]}")

            line = f"%{id_map[node]} = {node.format(predecessors, maximum_constant_length)}"
            lines.append(line)

            snap = node_to_snapshot.get(node)
            cols: dict[str, str] = {}

            if show_values:
                if snap is not None:
                    cols["value"] = f"=> {_format_value_short(snap.value)}"
                else:
                    cols["value"] = "=> (not evaluated)"

            if show_overflow:
                if snap is not None and snap.overflow:
                    cols["overflow"] = f"OVERFLOW [{snap.overflow_min}, {snap.overflow_max}]"
                else:
                    cols["overflow"] = ""

            extra_columns.append(cols)

        # Align = signs
        longest_before_eq = max(len(line.split("=")[0]) for line in lines)
        for i, line in enumerate(lines):
            before_eq_len = len(line.split("=")[0])
            lines[i] = " " * (longest_before_eq - before_eq_len) + line

        # Add extra columns
        shown_keys = []
        if show_values:
            shown_keys.append("value")
        if show_overflow:
            shown_keys.append("overflow")

        indent = 4
        for key in shown_keys:
            longest = max(len(line) for line in lines)
            lines = [
                line + " " * (longest - len(line) + indent) + cols.get(key, "")
                for line, cols in zip(lines, extra_columns)
            ]

        # Add return line
        returns = []
        for node in self._graph.ordered_outputs():
            returns.append(f"%{id_map[node]}")
        lines.append(f"return {', '.join(returns)}")

        if self._stopped:
            lines.append(
                f"(inspection stopped before reaching "
                f"{self._stopped_at_node.location if self._stopped_at_node else 'unknown node'})"
            )

        return "\n".join(line.rstrip() for line in lines)


def _format_value_short(value: Union[np.bool_, np.integer, np.floating, np.ndarray]) -> str:
    """Format a value concisely for display."""
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return repr(value)
        return f"array(shape={value.shape}, min={value.min()}, max={value.max()})"
    return repr(value)
