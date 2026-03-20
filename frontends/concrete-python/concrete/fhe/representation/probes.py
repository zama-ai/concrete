"""
Declaration of `ProbeSnapshot` and `ProbeResult` classes for simulation-mode debug probes.
"""

import re
from typing import Any, Callable, Optional, Union

import numpy as np

from .node import Node
from .operation import Operation


class ProbeSnapshot:
    """
    Snapshot of one probed node's value captured during simulation execution.
    """

    node: Node
    probe_id: int
    value: int
    tag: str
    nmsb: int

    def __init__(
        self,
        node: Node,
        probe_id: int,
        value: int,
        tag: str = "",
        nmsb: int = 0,
    ):
        self.node = node
        self.probe_id = probe_id
        self.value = value
        self.tag = tag
        self.nmsb = nmsb

    @property
    def location(self) -> str:
        """Get the source location of the node."""
        return self.node.location

    @property
    def node_tag(self) -> str:
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
    def overflow(self) -> bool:
        """Check if the probed value overflows the node's output dtype range."""
        from ..dtypes import Integer

        if not isinstance(self.node.output.dtype, Integer):
            return False

        dtype_min = self.node.output.dtype.min()
        dtype_max = self.node.output.dtype.max()
        return self.value < dtype_min or self.value > dtype_max

    def __repr__(self) -> str:
        return (
            f"ProbeSnapshot(probe_id={self.probe_id}, op={self.operation_name}, "
            f"value={self.value}, tag={self.tag!r})"
        )


class ProbeResult:
    """
    Collection of probe snapshots from a simulation run, with query and display methods.
    """

    _output: Any
    _snapshots: list[ProbeSnapshot]
    _graph: Any

    def __init__(
        self,
        output: Any,
        snapshots: list[ProbeSnapshot],
        graph: Any,
    ):
        self._output = output
        self._snapshots = snapshots
        self._graph = graph

    @property
    def output(self) -> Any:
        """Get the circuit output from the probed run."""
        return self._output

    def __iter__(self):
        return iter(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, index: int) -> ProbeSnapshot:
        return self._snapshots[index]

    def __repr__(self) -> str:
        overflow_count = sum(1 for s in self._snapshots if s.overflow)
        return (
            f"ProbeResult(output={self._output}, probes={len(self._snapshots)}, "
            f"overflows={overflow_count})"
        )

    def filter(
        self,
        tag_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        operation_filter: Optional[Union[str, list[str], re.Pattern]] = None,
        is_encrypted_filter: Optional[bool] = None,
        location_filter: Optional[Union[str, re.Pattern]] = None,
        custom_filter: Optional[Callable[[ProbeSnapshot], bool]] = None,
        overflow_only: bool = False,
    ) -> list[ProbeSnapshot]:
        """
        Filter probe snapshots by various criteria.

        Args:
            tag_filter: filter by node tag
            operation_filter: filter by operation name
            is_encrypted_filter: filter by encryption status
            location_filter: filter by source location
            custom_filter: arbitrary predicate on ProbeSnapshot
            overflow_only: if True, only return snapshots with overflow

        Returns:
            list of matching ProbeSnapshot objects
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
            if not match_text(tag_filter, snap.node_tag):
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
    def overflows(self) -> list[ProbeSnapshot]:
        """Get all snapshots that have overflow."""
        return self.filter(overflow_only=True)

    @property
    def has_overflow(self) -> bool:
        """Check if any probed node has overflow."""
        return any(s.overflow for s in self._snapshots)

    def summary(self) -> str:
        """
        Build a formatted summary table of the probe results.

        Returns:
            formatted string summary
        """
        if len(self._snapshots) == 0:
            return "(no probes captured)"

        lines = []
        lines.append(f"Output: {self._output}")
        lines.append(f"Probes: {len(self._snapshots)}")
        lines.append("")

        # Header
        lines.append(f"{'ID':>4}  {'Operation':<20}  {'Tag':<20}  {'Value':>12}  {'Overflow'}")
        lines.append("-" * 80)

        for snap in self._snapshots:
            overflow_str = "OVERFLOW" if snap.overflow else ""
            lines.append(
                f"{snap.probe_id:>4}  {snap.operation_name:<20}  "
                f"{snap.node_tag:<20}  {snap.value:>12}  {overflow_str}"
            )

        return "\n".join(line.rstrip() for line in lines)

    def compare_with(self, inspection: "InspectionResult") -> str:
        """
        Side-by-side comparison of simulation probes vs cleartext inspection.

        Args:
            inspection: InspectionResult from circuit.inspect()

        Returns:
            formatted comparison string
        """
        from .inspection import InspectionResult

        lines = []
        lines.append(f"{'Node':<30}  {'Inspect (cleartext)':>20}  {'Probe (simulation)':>20}  {'Match'}")
        lines.append("-" * 100)

        # Build lookup from node to probe value
        node_to_probe = {}
        for snap in self._snapshots:
            node_to_probe[snap.node] = snap.value

        for inspect_snap in inspection:
            node = inspect_snap.node
            inspect_val = inspect_snap.value
            probe_val = node_to_probe.get(node, None)

            if probe_val is not None:
                if isinstance(inspect_val, np.ndarray):
                    match = "array"
                else:
                    match = "OK" if int(inspect_val) == probe_val else "MISMATCH"
            else:
                match = "(no probe)"

            probe_str = str(probe_val) if probe_val is not None else "-"
            inspect_str = str(inspect_val)
            if len(inspect_str) > 20:
                inspect_str = inspect_str[:17] + "..."
            if len(probe_str) > 20:
                probe_str = probe_str[:17] + "..."

            op_name = inspect_snap.operation_name
            lines.append(f"{op_name:<30}  {inspect_str:>20}  {probe_str:>20}  {match}")

        return "\n".join(line.rstrip() for line in lines)
