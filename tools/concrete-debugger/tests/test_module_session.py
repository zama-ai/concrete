"""Tests for ModuleDebugSession — multi-function stepping."""

import networkx as nx
import numpy as np
import pytest

from concrete_dap.breakpoints import BreakpointManager
from concrete_dap.session import ConcreteDebugSession, ModuleDebugSession, StopReason


# ── Helpers ──


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
    def __init__(self, name, operation, evaluator, inputs=None, location="test.py:1",
                 tag="", output=None):
        self.name = name
        self.operation = operation
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


class MockGraph:
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


def _make_simple_graph(name_prefix, location_file):
    """Build: input(x) -> double(x)."""
    g = MockGraph()

    inp = MockNode(f"{name_prefix}_x", "input", lambda x: np.int64(x),
                   location=f"{location_file}:1")
    double = MockNode(f"{name_prefix}_double", "generic",
                      lambda a: np.int64(a * 2),
                      inputs=[MockValueDescription()],
                      location=f"{location_file}:2",
                      tag=name_prefix)

    g.graph.add_node(inp)
    g.graph.add_node(double)
    g.graph.add_edge(inp, double, input_idx=0)

    g.input_nodes = {0: inp}
    g.output_nodes = {0: double}
    g.input_indices = {inp: 0}

    return g


def _make_chain_graph(name_prefix, location_file):
    """Build: input(x) -> double(x) -> add1(x). Two non-input nodes for breakpoint tests."""
    g = MockGraph()

    inp = MockNode(f"{name_prefix}_x", "input", lambda x: np.int64(x),
                   location=f"{location_file}:1")
    double = MockNode(f"{name_prefix}_double", "generic",
                      lambda a: np.int64(a * 2),
                      inputs=[MockValueDescription()],
                      location=f"{location_file}:2",
                      tag=name_prefix)
    add1 = MockNode(f"{name_prefix}_add1", "generic",
                    lambda a: np.int64(a + 1),
                    inputs=[MockValueDescription()],
                    location=f"{location_file}:3",
                    tag=name_prefix)

    g.graph.add_node(inp)
    g.graph.add_node(double)
    g.graph.add_node(add1)
    g.graph.add_edge(inp, double, input_idx=0)
    g.graph.add_edge(double, add1, input_idx=0)

    g.input_nodes = {0: inp}
    g.output_nodes = {0: add1}
    g.input_indices = {inp: 0}

    return g


def _make_module(bp=None):
    """Create a two-function module session."""
    if bp is None:
        bp = BreakpointManager()
    g1 = _make_simple_graph("f1", "f1.py")
    g2 = _make_simple_graph("f2", "f2.py")
    s1 = ConcreteDebugSession(g1, (5,), bp)
    s2 = ConcreteDebugSession(g2, (10,), bp)
    module = ModuleDebugSession([("func1", s1), ("func2", s2)], bp)
    return module, bp


# ── Tests ──


class TestModuleStepOne:
    def test_step_advances_to_next_function(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        assert module.current_function_name == "func1"

        reason = module.step_one()
        # func1 double evaluated → func1 finishes → advance to func2 entry
        assert reason == StopReason.ENTRY
        assert module.current_function_name == "func2"

    def test_step_through_all_functions(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        reason = module.step_one()
        assert reason == StopReason.ENTRY  # advanced to func2

        reason = module.step_one()
        assert reason == StopReason.FINISHED  # func2 done, no more functions

    def test_snapshots_per_function(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        module.step_one()  # finish func1, enter func2

        # snapshots should be func2's snapshots (input was evaluated on advance)
        assert len(module.snapshots) == 1  # func2's input
        assert module.current_function_name == "func2"


class TestModuleContinue:
    def test_continue_runs_all(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        reason = module.continue_to_breakpoint()
        assert reason == StopReason.FINISHED
        assert module.finished

    def test_continue_stops_at_breakpoint_in_second_function(self):
        bp = BreakpointManager()
        module, _ = _make_module(bp)

        # Set breakpoint in func2
        bp.set_breakpoints("f2.py", [2])

        module.evaluate_inputs_and_stop_on_entry()
        reason = module.continue_to_breakpoint()

        assert reason == StopReason.BREAKPOINT
        assert module.current_function_name == "func2"
        assert not module.finished


class TestModuleStepOut:
    def test_step_out_finishes_function_and_enters_next(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        reason = module.step_out()
        assert reason == StopReason.ENTRY
        assert module.current_function_name == "func2"

    def test_step_out_last_function(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        module.step_out()  # finish func1 → enter func2
        reason = module.step_out()  # finish func2 → no more functions
        assert reason == StopReason.FINISHED
        assert module.finished


class TestModuleStepIntoNextFunction:
    def test_step_into_when_function_finished(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        # Finish func1 via its inner session
        module._active.step_out()
        assert module._active.finished

        reason = module.step_into_next_function()
        assert reason == StopReason.ENTRY
        assert module.current_function_name == "func2"

    def test_step_into_when_not_finished_acts_as_step(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        # func1 is not finished, step_into falls back to step_one
        reason = module.step_into_next_function()
        assert reason == StopReason.ENTRY  # func1 finishes → advance


class TestModuleStackFrames:
    def test_includes_module_frame(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        module.step_one()  # advance to func2

        frames = module.get_stack_frames()
        assert len(frames) >= 2
        bottom = frames[-1]
        assert "Module" in bottom["name"]
        assert "func2" in bottom["name"]
        assert "2/2" in bottom["name"]

    def test_module_frame_shows_progress(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()

        frames = module.get_stack_frames()
        bottom = frames[-1]
        assert "1/2" in bottom["name"]


class TestModuleProperties:
    def test_finished_false_while_running(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        assert not module.finished

    def test_finished_true_when_all_done(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        module.continue_to_breakpoint()
        assert module.finished

    def test_all_snapshots_across_functions(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        module.continue_to_breakpoint()

        # Each function: 1 input + 1 double = 2 snapshots
        all_snaps = module.all_snapshots
        assert len(all_snaps) == 4

    def test_current_function_name(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        assert module.current_function_name == "func1"

        module.step_one()
        assert module.current_function_name == "func2"

    def test_error_from_active_session(self):
        module, _ = _make_module()
        module.evaluate_inputs_and_stop_on_entry()
        assert module.error is None


class TestModuleBreakpointsAcrossFunctions:
    def test_breakpoint_in_first_function(self):
        bp = BreakpointManager()
        # Use chain graphs so breakpoint isn't on the first non-input node
        g1 = _make_chain_graph("f1", "f1.py")
        g2 = _make_chain_graph("f2", "f2.py")
        s1 = ConcreteDebugSession(g1, (5,), bp)
        s2 = ConcreteDebugSession(g2, (10,), bp)
        module = ModuleDebugSession([("func1", s1), ("func2", s2)], bp)

        bp.set_breakpoints("f1.py", [3])  # breakpoint on second non-input node
        module.evaluate_inputs_and_stop_on_entry()
        reason = module.continue_to_breakpoint()

        assert reason == StopReason.BREAKPOINT
        assert module.current_function_name == "func1"

    def test_breakpoint_in_second_function(self):
        bp = BreakpointManager()
        g1 = _make_chain_graph("f1", "f1.py")
        g2 = _make_chain_graph("f2", "f2.py")
        s1 = ConcreteDebugSession(g1, (5,), bp)
        s2 = ConcreteDebugSession(g2, (10,), bp)
        module = ModuleDebugSession([("func1", s1), ("func2", s2)], bp)

        bp.set_breakpoints("f2.py", [3])
        module.evaluate_inputs_and_stop_on_entry()
        reason = module.continue_to_breakpoint()

        assert reason == StopReason.BREAKPOINT
        assert module.current_function_name == "func2"

    def test_breakpoints_in_both_functions(self):
        bp = BreakpointManager()
        g1 = _make_chain_graph("f1", "f1.py")
        g2 = _make_chain_graph("f2", "f2.py")
        s1 = ConcreteDebugSession(g1, (5,), bp)
        s2 = ConcreteDebugSession(g2, (10,), bp)
        module = ModuleDebugSession([("func1", s1), ("func2", s2)], bp)

        bp.set_breakpoints("f1.py", [3])
        bp.set_breakpoints("f2.py", [3])
        module.evaluate_inputs_and_stop_on_entry()

        reason = module.continue_to_breakpoint()
        assert reason == StopReason.BREAKPOINT
        assert module.current_function_name == "func1"

        reason = module.continue_to_breakpoint()
        assert reason == StopReason.BREAKPOINT
        assert module.current_function_name == "func2"
