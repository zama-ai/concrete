"""Tests for BreakpointManager."""

import pytest

from concrete_dap.breakpoints import BreakpointManager


class MockNode:
    def __init__(self, location=""):
        self.location = location


class TestBreakpointManager:
    def test_build_index(self):
        bp = BreakpointManager()
        nodes = [
            MockNode("script.py:10"),
            MockNode("script.py:11"),
            MockNode("script.py:10"),  # same line as first
        ]
        bp.build_index(nodes)
        lines = bp.get_available_lines("script.py")
        assert sorted(lines) == [10, 11]

    def test_set_breakpoints_verified(self):
        bp = BreakpointManager()
        nodes = [MockNode("app.py:5"), MockNode("app.py:10")]
        bp.build_index(nodes)

        results = bp.set_breakpoints("app.py", [5, 7, 10])
        assert results[0]["verified"] is True
        assert results[0]["line"] == 5
        assert results[1]["verified"] is False  # line 7 has no nodes
        assert results[2]["verified"] is True

    def test_is_breakpoint(self):
        bp = BreakpointManager()
        n1 = MockNode("app.py:5")
        n2 = MockNode("app.py:5")  # same line, second node
        n3 = MockNode("app.py:10")
        nodes = [n1, n2, n3]
        bp.build_index(nodes)
        bp.set_breakpoints("app.py", [5])

        # Only the first node at line 5 should trigger
        assert bp.is_breakpoint(n1, 0) is True
        assert bp.is_breakpoint(n2, 1) is False  # not first at this line
        assert bp.is_breakpoint(n3, 2) is False  # not a breakpoint line

    def test_clear_old_breakpoints(self):
        bp = BreakpointManager()
        nodes = [MockNode("a.py:1"), MockNode("a.py:2")]
        bp.build_index(nodes)
        bp.set_breakpoints("a.py", [1, 2])

        # Now set only line 2 — line 1 should be cleared
        bp.set_breakpoints("a.py", [2])
        assert bp.is_breakpoint(nodes[0], 0) is False
        assert bp.is_breakpoint(nodes[1], 1) is True

    def test_node_without_location(self):
        bp = BreakpointManager()
        nodes = [MockNode(""), MockNode("valid.py:1")]
        bp.build_index(nodes)
        lines = bp.get_available_lines("valid.py")
        assert lines == [1]

    def test_is_breakpoint_no_location(self):
        bp = BreakpointManager()
        node = MockNode("")
        assert bp.is_breakpoint(node, 0) is False

    def test_path_normalization(self):
        """Paths should be normalized for comparison."""
        bp = BreakpointManager()
        nodes = [MockNode("/home/user/./scripts/../scripts/app.py:5")]
        bp.build_index(nodes)

        # Setting breakpoints with a different but equivalent path
        results = bp.set_breakpoints("/home/user/scripts/app.py", [5])
        assert results[0]["verified"] is True
