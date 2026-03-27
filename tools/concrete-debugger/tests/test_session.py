"""Tests for ConcreteDebugSession — graph walker with stepping."""

import networkx as nx
import numpy as np
import pytest

from concrete_dap.breakpoints import BreakpointManager
from concrete_dap.session import ConcreteDebugSession, StopReason, _FallbackSnapshot


# ── Helpers to build a mock graph ──


class MockValueDescription:
    def __init__(self, shape=(), is_encrypted=True, dtype=None):
        self.shape = shape
        self.is_encrypted = is_encrypted
        self.dtype = dtype or MockDtype()


class MockDtype:
    def __init__(self, bit_width=8):
        self.bit_width = bit_width

    def min(self):
        return -(2 ** (self.bit_width - 1))

    def max(self):
        return 2 ** (self.bit_width - 1) - 1


class MockNode:
    """Minimal Node-like object for testing."""

    def __init__(self, name, operation, evaluator, inputs=None, location="test.py:1",
                 tag="", output=None):
        self.name = name
        self.operation = operation  # string: "input", "generic", "constant"
        self.evaluator = evaluator
        self.inputs = inputs or []
        self.location = location
        self.tag = tag
        self.output = output or MockValueDescription()
        self.bounds = None
        self.properties = {"name": name}
        self.created_at = 0.0

    def __call__(self, *args):
        return self.evaluator(*args)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def label(self):
        return self.name


class MockGraph:
    """Minimal Graph-like object wrapping a networkx digraph."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.input_nodes = {}
        self.output_nodes = {}
        self.input_indices = {}

    def ordered_preds_of(self, node):
        idx_to_pred = {}
        for pred in self.graph.predecessors(node):
            edge_data = self.graph.get_edge_data(pred, node)
            for data in edge_data.values():
                idx_to_pred[data["input_idx"]] = pred
        return [idx_to_pred[i] for i in range(len(idx_to_pred))]


def _make_add_graph():
    """Build: input(x) -> input(y) -> add(x, y)."""
    g = MockGraph()

    inp_x = MockNode("x", "input", lambda x: np.int64(x),
                      location="script.py:5", tag="")
    inp_y = MockNode("y", "input", lambda y: np.int64(y),
                      location="script.py:5", tag="")
    add = MockNode("add", "generic",
                   lambda a, b: np.int64(a + b),
                   inputs=[MockValueDescription(), MockValueDescription()],
                   location="script.py:6", tag="")

    g.graph.add_node(inp_x)
    g.graph.add_node(inp_y)
    g.graph.add_node(add)
    g.graph.add_edge(inp_x, add, input_idx=0)
    g.graph.add_edge(inp_y, add, input_idx=1)

    g.input_nodes = {0: inp_x, 1: inp_y}
    g.output_nodes = {0: add}
    g.input_indices = {inp_x: 0, inp_y: 1}

    return g, inp_x, inp_y, add


def _make_chain_graph():
    """Build: input(x) -> double(x) -> triple(x)."""
    g = MockGraph()

    inp = MockNode("x", "input", lambda x: np.int64(x),
                   location="chain.py:1", tag="layer1")
    double = MockNode("double", "generic",
                      lambda a: np.int64(a * 2),
                      inputs=[MockValueDescription()],
                      location="chain.py:2", tag="layer1.double")
    triple = MockNode("triple", "generic",
                      lambda a: np.int64(a * 3),
                      inputs=[MockValueDescription()],
                      location="chain.py:3", tag="layer1.triple")

    g.graph.add_node(inp)
    g.graph.add_node(double)
    g.graph.add_node(triple)
    g.graph.add_edge(inp, double, input_idx=0)
    g.graph.add_edge(double, triple, input_idx=0)

    g.input_nodes = {0: inp}
    g.output_nodes = {0: triple}
    g.input_indices = {inp: 0}

    return g, inp, double, triple


def _make_same_line_graph():
    """Build: input(x) -> mul(x, x) -> add(mul, x). Both ops on same line."""
    g = MockGraph()

    inp = MockNode("x", "input", lambda x: np.int64(x),
                   location="sameline.py:1")
    mul = MockNode("mul", "generic",
                   lambda a, b: np.int64(a * b),
                   inputs=[MockValueDescription(), MockValueDescription()],
                   location="sameline.py:5")
    add = MockNode("add", "generic",
                   lambda a, b: np.int64(a + b),
                   inputs=[MockValueDescription(), MockValueDescription()],
                   location="sameline.py:5")

    g.graph.add_node(inp)
    g.graph.add_node(mul)
    g.graph.add_node(add)
    g.graph.add_edge(inp, mul, input_idx=0)
    g.graph.add_edge(inp, mul, input_idx=1)
    g.graph.add_edge(mul, add, input_idx=0)
    g.graph.add_edge(inp, add, input_idx=1)

    g.input_nodes = {0: inp}
    g.output_nodes = {0: add}
    g.input_indices = {inp: 0}

    return g, inp, mul, add


# ── Tests ──


class TestStopOnEntry:
    def test_evaluates_inputs_and_stops(self):
        graph, inp_x, inp_y, add = _make_add_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (3, 5), bp)

        reason = session.evaluate_inputs_and_stop_on_entry()
        assert reason == StopReason.ENTRY
        assert len(session.snapshots) == 2
        assert not session.finished
        assert session.current_index == 2

    def test_empty_graph(self):
        g = MockGraph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(g, (), bp)
        reason = session.evaluate_inputs_and_stop_on_entry()
        assert reason == StopReason.FINISHED
        assert session.finished

    def test_input_only_graph(self):
        g = MockGraph()
        inp = MockNode("x", "input", lambda x: np.int64(x), location="test.py:1")
        g.graph.add_node(inp)
        g.input_nodes = {0: inp}
        g.input_indices = {inp: 0}

        bp = BreakpointManager()
        session = ConcreteDebugSession(g, (42,), bp)
        reason = session.evaluate_inputs_and_stop_on_entry()
        assert reason == StopReason.FINISHED
        assert len(session.snapshots) == 1
        assert int(session.snapshots[0].value) == 42


class TestStepOne:
    def test_step_through_add(self):
        graph, *_ = _make_add_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (3, 5), bp)

        session.evaluate_inputs_and_stop_on_entry()
        reason = session.step_one()
        assert reason == StopReason.FINISHED
        assert len(session.snapshots) == 3
        assert int(session.snapshots[-1].value) == 8

    def test_step_through_chain(self):
        graph, *_ = _make_chain_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (4,), bp)

        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_one()
        assert reason == StopReason.STEP
        assert int(session.snapshots[-1].value) == 8

        reason = session.step_one()
        assert reason == StopReason.FINISHED
        assert int(session.snapshots[-1].value) == 24

    def test_same_line_grouping(self):
        graph, *_ = _make_same_line_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (3,), bp)

        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_one()
        assert reason == StopReason.FINISHED
        assert len(session.snapshots) == 3  # input + mul + add
        assert int(session.snapshots[-1].value) == 12  # 3*3 + 3

    def test_step_on_finished(self):
        graph, *_ = _make_add_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (3, 5), bp)
        session.evaluate_inputs_and_stop_on_entry()
        session.step_one()
        assert session.finished
        reason = session.step_one()
        assert reason == StopReason.FINISHED


class TestContinueToBreakpoint:
    def test_continue_no_breakpoints(self):
        graph, *_ = _make_chain_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (4,), bp)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.continue_to_breakpoint()
        assert reason == StopReason.FINISHED
        assert session.finished
        assert int(session.snapshots[-1].value) == 24

    def test_continue_with_breakpoint(self):
        graph, *_ = _make_chain_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (4,), bp)
        # Set breakpoint at chain.py:3 (triple node)
        bp.set_breakpoints("chain.py", [3])

        session.evaluate_inputs_and_stop_on_entry()
        reason = session.continue_to_breakpoint()
        assert reason == StopReason.BREAKPOINT
        assert len(session.snapshots) == 2  # input + double
        assert int(session.snapshots[-1].value) == 8

    def test_continue_on_finished(self):
        graph, *_ = _make_add_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (1, 2), bp)
        session.evaluate_inputs_and_stop_on_entry()
        session.step_one()
        assert session.finished
        reason = session.continue_to_breakpoint()
        assert reason == StopReason.FINISHED


class TestStepOut:
    def test_step_out_runs_to_end(self):
        graph, *_ = _make_chain_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (2,), bp)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_out()
        assert reason == StopReason.FINISHED
        assert session.finished
        assert int(session.snapshots[-1].value) == 12


class TestStackFrames:
    def test_tag_hierarchy(self):
        graph, *_ = _make_chain_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (4,), bp)
        session.evaluate_inputs_and_stop_on_entry()
        session.step_one()

        frames = session.get_stack_frames()
        assert len(frames) == 2
        assert "double" in frames[0]["name"]
        assert "layer1" in frames[1]["name"]

    def test_no_tag(self):
        graph, *_ = _make_add_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (1, 2), bp)
        session.evaluate_inputs_and_stop_on_entry()
        session.step_one()

        frames = session.get_stack_frames()
        assert len(frames) == 1
        assert frames[0]["name"] == "add"

    def test_no_snapshot(self):
        g = MockGraph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(g, (), bp)
        frames = session.get_stack_frames()
        assert frames == []


class TestEvaluationError:
    def test_step_with_error(self):
        g = MockGraph()
        inp = MockNode("x", "input", lambda x: np.int64(x), location="err.py:1")

        def bad_eval(a):
            raise ValueError("kaboom")

        bad = MockNode("bad", "generic", bad_eval,
                       inputs=[MockValueDescription()], location="err.py:2")
        g.graph.add_node(inp)
        g.graph.add_node(bad)
        g.graph.add_edge(inp, bad, input_idx=0)
        g.input_nodes = {0: inp}
        g.input_indices = {inp: 0}

        bp = BreakpointManager()
        session = ConcreteDebugSession(g, (1,), bp)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_one()
        assert reason == StopReason.EXCEPTION
        assert session.error is not None
        assert "kaboom" in str(session.error)


def _make_overflow_graph():
    """Build: input(x) -> mul(x, 100). With bit_width=8, mul overflows for x > 1."""
    g = MockGraph()

    inp = MockNode("x", "input", lambda x: np.int64(x),
                   location="overflow.py:1")
    mul = MockNode("mul", "generic",
                   lambda a: np.int64(a * 100),
                   inputs=[MockValueDescription()],
                   location="overflow.py:2",
                   output=MockValueDescription(dtype=MockDtype(bit_width=8)))

    g.graph.add_node(inp)
    g.graph.add_node(mul)
    g.graph.add_edge(inp, mul, input_idx=0)

    g.input_nodes = {0: inp}
    g.output_nodes = {0: mul}
    g.input_indices = {inp: 0}

    return g, inp, mul


class TestOverflowStop:
    def test_step_one_stops_on_overflow(self):
        graph, inp, mul = _make_overflow_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (2,), bp, stop_on_overflow=True)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_one()
        assert reason == StopReason.OVERFLOW
        assert session.snapshots[-1].overflow is True

    def test_step_one_no_stop_when_disabled(self):
        graph, inp, mul = _make_overflow_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (2,), bp, stop_on_overflow=False)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_one()
        assert reason == StopReason.FINISHED  # doesn't stop, runs to end
        assert session.snapshots[-1].overflow is True

    def test_continue_stops_on_overflow(self):
        graph, inp, mul = _make_overflow_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (2,), bp, stop_on_overflow=True)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.continue_to_breakpoint()
        assert reason == StopReason.OVERFLOW

    def test_step_out_stops_on_overflow(self):
        graph, inp, mul = _make_overflow_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (2,), bp, stop_on_overflow=True)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_out()
        assert reason == StopReason.OVERFLOW

    def test_no_overflow_no_stop(self):
        graph, inp, mul = _make_overflow_graph()
        bp = BreakpointManager()
        session = ConcreteDebugSession(graph, (1,), bp, stop_on_overflow=True)
        session.evaluate_inputs_and_stop_on_entry()

        reason = session.step_one()
        # 1*100=100, within [-128, 127]
        assert reason == StopReason.FINISHED
        assert session.snapshots[-1].overflow is False


class TestFallbackSnapshotOverflow:
    def test_overflow_detected(self):
        node = MockNode("mul", "generic", lambda a: a,
                        output=MockValueDescription(dtype=MockDtype(bit_width=8)))
        snap = _FallbackSnapshot(node, np.int64(200), 0)
        assert snap.overflow is True

    def test_no_overflow(self):
        node = MockNode("mul", "generic", lambda a: a,
                        output=MockValueDescription(dtype=MockDtype(bit_width=8)))
        snap = _FallbackSnapshot(node, np.int64(50), 0)
        assert snap.overflow is False

    def test_negative_overflow(self):
        node = MockNode("mul", "generic", lambda a: a,
                        output=MockValueDescription(dtype=MockDtype(bit_width=8)))
        snap = _FallbackSnapshot(node, np.int64(-200), 0)
        assert snap.overflow is True

    def test_array_overflow(self):
        node = MockNode("mul", "generic", lambda a: a,
                        output=MockValueDescription(dtype=MockDtype(bit_width=8)))
        snap = _FallbackSnapshot(node, np.array([50, 200]), 0)
        assert snap.overflow is True

    def test_array_no_overflow(self):
        node = MockNode("mul", "generic", lambda a: a,
                        output=MockValueDescription(dtype=MockDtype(bit_width=8)))
        snap = _FallbackSnapshot(node, np.array([50, 100]), 0)
        assert snap.overflow is False
