"""ConcreteDebugSession: graph walker with pause/resume for DAP stepping."""

from copy import deepcopy
from enum import Enum, auto
from typing import Any, Optional

import networkx as nx
import numpy as np

from .breakpoints import BreakpointManager

# Module-level references — populated lazily from concrete.fhe or overridden by tests.
_NodeSnapshot = None
_OperationInput = None


def _ensure_imports():
    """Populate _NodeSnapshot and _OperationInput from concrete.fhe (if available)."""
    global _NodeSnapshot, _OperationInput
    if _NodeSnapshot is None:
        try:
            from concrete.fhe.representation.inspection import NodeSnapshot
            _NodeSnapshot = NodeSnapshot
        except ImportError:
            pass
    if _OperationInput is None:
        try:
            from concrete.fhe.representation.operation import Operation
            _OperationInput = Operation.Input
        except ImportError:
            pass


def _is_input_node(node) -> bool:
    """Check whether *node* is an input node, compatible with both real and mock types."""
    op = node.operation
    if _OperationInput is not None and op == _OperationInput:
        return True
    # Fallback: duck-type check for mock/string operations
    op_val = getattr(op, "value", op)
    return op_val == "input"


class StopReason(Enum):
    STEP = auto()
    BREAKPOINT = auto()
    ENTRY = auto()
    FINISHED = auto()
    EXCEPTION = auto()


class ConcreteDebugSession:
    """Walks a Concrete FHE graph node-by-node with pause/resume state."""

    def __init__(self, graph, args: tuple, breakpoints: BreakpointManager):
        _ensure_imports()
        self.graph = graph
        self.args = args
        self.breakpoints = breakpoints

        self.topo_order: list = list(nx.topological_sort(graph.graph))
        self.current_index: int = 0
        self.node_results: dict = {}
        self.snapshots: list = []
        self.finished: bool = False
        self._error: Optional[Exception] = None

        # Build breakpoint index
        breakpoints.build_index(self.topo_order)

    @property
    def current_node(self):
        """The node about to be evaluated (or just evaluated after a step)."""
        if self.current_index > 0 and self.current_index <= len(self.topo_order):
            return self.topo_order[self.current_index - 1]
        return None

    @property
    def current_snapshot(self):
        """The most recent snapshot after stepping."""
        if self.snapshots:
            return self.snapshots[-1]
        return None

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def evaluate_inputs_and_stop_on_entry(self) -> StopReason:
        """Evaluate all Input nodes and stop before the first non-Input node."""
        while self.current_index < len(self.topo_order):
            node = self.topo_order[self.current_index]
            if not _is_input_node(node):
                return StopReason.ENTRY
            self._evaluate_node(node)
            self.current_index += 1

        self.finished = True
        return StopReason.FINISHED

    def step_one(self) -> StopReason:
        """Evaluate the next node and stop (with same-line grouping)."""
        if self.finished or self.current_index >= len(self.topo_order):
            self.finished = True
            return StopReason.FINISHED

        node = self.topo_order[self.current_index]
        try:
            self._evaluate_node(node)
        except Exception as e:
            self._error = e
            self.finished = True
            return StopReason.EXCEPTION
        current_location = node.location
        self.current_index += 1

        # Keep stepping while the next node has the same source location
        while self.current_index < len(self.topo_order):
            next_node = self.topo_order[self.current_index]
            if _is_input_node(next_node):
                break
            if next_node.location != current_location:
                break
            try:
                self._evaluate_node(next_node)
            except Exception as e:
                self._error = e
                self.finished = True
                return StopReason.EXCEPTION
            self.current_index += 1

        if self.current_index >= len(self.topo_order):
            self.finished = True
            return StopReason.FINISHED

        return StopReason.STEP

    def continue_to_breakpoint(self) -> StopReason:
        """Evaluate nodes until a breakpoint is hit or the graph finishes."""
        if self.finished:
            return StopReason.FINISHED

        # If currently stopped at a breakpoint node, step past it first
        first_step = True
        while self.current_index < len(self.topo_order):
            node = self.topo_order[self.current_index]

            # Check breakpoint (skip on first step to avoid re-stopping at same spot)
            if not first_step and self.breakpoints.is_breakpoint(node, self.current_index):
                return StopReason.BREAKPOINT
            first_step = False

            try:
                self._evaluate_node(node)
            except Exception as e:
                self._error = e
                self.finished = True
                return StopReason.EXCEPTION
            self.current_index += 1

        self.finished = True
        return StopReason.FINISHED

    def step_out(self) -> StopReason:
        """Step out — in single-graph mode, runs to completion."""
        while self.current_index < len(self.topo_order):
            node = self.topo_order[self.current_index]
            try:
                self._evaluate_node(node)
            except Exception as e:
                self._error = e
                self.finished = True
                return StopReason.EXCEPTION
            self.current_index += 1

        self.finished = True
        return StopReason.FINISHED

    def _evaluate_node(self, node):
        """Evaluate a single node and create a snapshot."""
        if _is_input_node(node):
            self.node_results[node] = node(self.args[self.graph.input_indices[node]])
        else:
            pred_results = [
                deepcopy(self.node_results[pred])
                for pred in self.graph.ordered_preds_of(node)
            ]
            self.node_results[node] = node(*pred_results)

        snapshot_cls = _NodeSnapshot
        if snapshot_cls is None:
            # Fallback: plain object (for testing without concrete)
            snapshot_cls = _FallbackSnapshot
        snapshot = snapshot_cls(node, deepcopy(self.node_results[node]), len(self.snapshots))
        self.snapshots.append(snapshot)
        return snapshot

    def get_stack_frames(self) -> list[dict]:
        """Synthesize DAP stack frames from the tag hierarchy of the current node."""
        snapshot = self.current_snapshot
        if snapshot is None:
            return []

        tag = snapshot.tag
        location = snapshot.location
        frames = []

        # Parse file:line from location
        source_info = _parse_location(location)

        if tag:
            parts = tag.split(".")
            # Innermost frame first
            for i in range(len(parts), 0, -1):
                frame_tag = ".".join(parts[:i])
                frame_name = parts[i - 1]
                frame = {
                    "id": i - 1,
                    "name": f"{frame_name} [{frame_tag}]",
                    "line": source_info.get("line", 0),
                    "column": 0,
                }
                if source_info.get("path"):
                    frame["source"] = {"path": source_info["path"]}
                frames.append(frame)
        else:
            # No tag — single frame with operation name
            frame = {
                "id": 0,
                "name": snapshot.operation_name,
                "line": source_info.get("line", 0),
                "column": 0,
            }
            if source_info.get("path"):
                frame["source"] = {"path": source_info["path"]}
            frames.append(frame)

        return frames


class _FallbackSnapshot:
    """Minimal snapshot for when concrete.fhe is not available (testing)."""

    def __init__(self, node, value, index):
        self.node = node
        self.value = value
        self.index = index
        self.overflow = False

    @property
    def location(self):
        return self.node.location

    @property
    def tag(self):
        return self.node.tag

    @property
    def operation_name(self):
        op = self.node.operation
        op_val = getattr(op, "value", op)
        if op_val == "input":
            return "input"
        if op_val == "constant":
            return "constant"
        return self.node.properties.get("name", "unknown")

    @property
    def is_encrypted(self):
        return getattr(self.node.output, "is_encrypted", False)


def _parse_location(location: str) -> dict:
    """Parse 'file.py:42' into {path, line}."""
    if not location or ":" not in location:
        return {}
    parts = location.rsplit(":", 1)
    try:
        return {"path": parts[0], "line": int(parts[1])}
    except (ValueError, IndexError):
        return {}
