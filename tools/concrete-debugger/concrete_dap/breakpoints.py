"""Breakpoint manager: maps file:line to DAG node sets."""

import os
from typing import Optional


class BreakpointManager:
    """Maps (filename, lineno) pairs to sets of node indices in topological order."""

    def __init__(self):
        self._location_index: dict[tuple[str, int], list[int]] = {}
        self._active: set[tuple[str, int]] = set()

    def build_index(self, topo_order: list) -> None:
        """Build a reverse index from (file, line) to node indices in topo_order."""
        self._location_index.clear()
        for idx, node in enumerate(topo_order):
            loc = getattr(node, "location", "")
            if not loc or ":" not in loc:
                continue
            parts = loc.rsplit(":", 1)
            try:
                filepath = _normalize_path(parts[0])
                lineno = int(parts[1])
            except (ValueError, IndexError):
                continue
            key = (filepath, lineno)
            if key not in self._location_index:
                self._location_index[key] = []
            self._location_index[key].append(idx)

    def set_breakpoints(self, source_path: str, lines: list[int]) -> list[dict]:
        """Set breakpoints for a source file, returning DAP Breakpoint objects."""
        norm_path = _normalize_path(source_path)

        # Remove old breakpoints for this file
        self._active = {
            (f, l) for f, l in self._active if f != norm_path
        }

        results = []
        for line in lines:
            key = (norm_path, line)
            verified = key in self._location_index
            if verified:
                self._active.add(key)
            results.append({
                "verified": verified,
                "line": line,
                "source": {"path": source_path},
            })
        return results

    def is_breakpoint(self, node, topo_index: int) -> bool:
        """Check if a node at given topo index is at a breakpoint location."""
        loc = getattr(node, "location", "")
        if not loc or ":" not in loc:
            return False
        parts = loc.rsplit(":", 1)
        try:
            filepath = _normalize_path(parts[0])
            lineno = int(parts[1])
        except (ValueError, IndexError):
            return False
        key = (filepath, lineno)
        if key not in self._active:
            return False
        # Only stop at the first node at this location (in topo order)
        first_idx = self._location_index.get(key, [None])[0]
        return topo_index == first_idx

    def get_available_lines(self, source_path: str) -> list[int]:
        """Get all lines in a source file that have DAG nodes."""
        norm_path = _normalize_path(source_path)
        return sorted({l for f, l in self._location_index if f == norm_path})


def _normalize_path(path: str) -> str:
    """Normalize a file path for consistent comparison."""
    return os.path.normcase(os.path.normpath(path))
