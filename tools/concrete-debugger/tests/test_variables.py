"""Tests for VariableStore."""

import numpy as np
import pytest

from concrete_dap.variables import VariableStore


class MockDtype:
    def __init__(self, bit_width=8):
        self.bit_width = bit_width


class MockValueDescription:
    def __init__(self, is_encrypted=True, dtype=None):
        self.is_encrypted = is_encrypted
        self.dtype = dtype or MockDtype()


class MockNode:
    def __init__(self, location="test.py:1", tag="", operation="generic",
                 is_encrypted=True, bounds=None, bit_width=8):
        self.location = location
        self.tag = tag
        self.operation = operation
        self.output = MockValueDescription(is_encrypted, MockDtype(bit_width))
        self.bounds = bounds
        self.properties = {"name": "add"}


class MockSnapshot:
    def __init__(self, value, index=0, node=None, overflow=False):
        self.value = value
        self.index = index
        self.node = node or MockNode()
        self.overflow = overflow
        self.overflow_min = None
        self.overflow_max = None

    @property
    def location(self):
        return self.node.location

    @property
    def tag(self):
        return self.node.tag

    @property
    def operation_name(self):
        return self.node.properties.get("name", "unknown")

    @property
    def is_encrypted(self):
        return self.node.output.is_encrypted


class TestVariableStore:
    def test_scopes(self):
        store = VariableStore()
        snap = MockSnapshot(np.int64(42))
        scopes = store.scopes_for_stop(snap, [snap])
        assert len(scopes) == 2
        assert scopes[0]["name"] == "Current Node"
        assert scopes[1]["name"] == "All Evaluated"
        assert scopes[0]["variablesReference"] > 0
        assert scopes[1]["variablesReference"] > 0

    def test_current_node_variables(self):
        store = VariableStore()
        node = MockNode(location="test.py:10", tag="layer1", bounds=(0, 100))
        snap = MockSnapshot(np.int64(42), node=node)
        scopes = store.scopes_for_stop(snap, [snap])
        variables = store.get_variables(scopes[0]["variablesReference"])

        var_dict = {v["name"]: v["value"] for v in variables}
        assert var_dict["value"] == "np.int64(42)"
        assert var_dict["operation"] == "add"
        assert var_dict["encrypted"] == "True"
        assert var_dict["bit_width"] == "8"
        assert var_dict["overflow"] == "False"
        assert var_dict["tag"] == "layer1"
        assert var_dict["location"] == "test.py:10"
        assert var_dict["bounds"] == "[0, 100]"

    def test_no_tag(self):
        store = VariableStore()
        snap = MockSnapshot(np.int64(1), node=MockNode(tag=""))
        scopes = store.scopes_for_stop(snap, [snap])
        variables = store.get_variables(scopes[0]["variablesReference"])
        var_dict = {v["name"]: v["value"] for v in variables}
        assert var_dict["tag"] == "(none)"

    def test_array_value_expandable(self):
        store = VariableStore()
        arr = np.arange(100)
        snap = MockSnapshot(arr)
        scopes = store.scopes_for_stop(snap, [snap])
        variables = store.get_variables(scopes[0]["variablesReference"])

        value_var = next(v for v in variables if v["name"] == "value")
        assert value_var["variablesReference"] > 0  # expandable
        assert "shape" in value_var["value"]

        # Expand the array
        arr_vars = store.get_variables(value_var["variablesReference"])
        assert len(arr_vars) == 100
        assert arr_vars[0]["name"] == "[0]"
        assert arr_vars[0]["value"] == "0"

    def test_small_array_inline(self):
        store = VariableStore()
        arr = np.array([1, 2, 3])
        snap = MockSnapshot(arr)
        scopes = store.scopes_for_stop(snap, [snap])
        variables = store.get_variables(scopes[0]["variablesReference"])

        value_var = next(v for v in variables if v["name"] == "value")
        assert value_var["variablesReference"] == 0  # not expandable

    def test_all_evaluated_scope(self):
        store = VariableStore()
        snaps = [
            MockSnapshot(np.int64(1), index=0),
            MockSnapshot(np.int64(2), index=1),
            MockSnapshot(np.int64(3), index=2),
        ]
        scopes = store.scopes_for_stop(snaps[-1], snaps)
        variables = store.get_variables(scopes[1]["variablesReference"])

        assert len(variables) == 3
        assert "[0]" in variables[0]["name"]
        assert "[2]" in variables[2]["name"]
        # Each should be expandable
        assert variables[0]["variablesReference"] > 0

    def test_reset(self):
        store = VariableStore()
        snap = MockSnapshot(np.int64(1))
        scopes = store.scopes_for_stop(snap, [snap])
        ref = scopes[0]["variablesReference"]
        assert len(store.get_variables(ref)) > 0

        store.reset()
        assert store.get_variables(ref) == []

    def test_large_array_capped(self):
        store = VariableStore()
        arr = np.arange(500)
        snap = MockSnapshot(arr)
        scopes = store.scopes_for_stop(snap, [snap])
        variables = store.get_variables(scopes[0]["variablesReference"])
        value_var = next(v for v in variables if v["name"] == "value")

        arr_vars = store.get_variables(value_var["variablesReference"])
        assert len(arr_vars) == 201  # 200 elements + "..." entry
        assert arr_vars[-1]["name"] == "..."

    def test_unknown_ref(self):
        store = VariableStore()
        assert store.get_variables(999) == []

    def test_overflow_shown(self):
        store = VariableStore()
        snap = MockSnapshot(np.int64(42), overflow=True)
        scopes = store.scopes_for_stop(snap, [snap])
        variables = store.get_variables(scopes[0]["variablesReference"])
        var_dict = {v["name"]: v["value"] for v in variables}
        assert var_dict["overflow"] == "True"


class TestModuleContextScope:
    def test_module_scope_variables(self):
        store = VariableStore()
        scopes = store.scopes_for_module_stop("encrypt_layer", 0, 3, [])
        assert len(scopes) == 1
        assert scopes[0]["name"] == "Module Context"

        variables = store.get_variables(scopes[0]["variablesReference"])
        var_dict = {v["name"]: v["value"] for v in variables}
        assert var_dict["function"] == "encrypt_layer"
        assert var_dict["progress"] == "1/3"

    def test_module_scope_with_snapshots(self):
        store = VariableStore()
        snaps = [
            MockSnapshot(np.int64(1), index=0),
            MockSnapshot(np.int64(2), index=1),
        ]
        scopes = store.scopes_for_module_stop("compute", 1, 2, snaps)
        variables = store.get_variables(scopes[0]["variablesReference"])

        var_dict = {v["name"]: v for v in variables}
        assert var_dict["progress"]["value"] == "2/2"
        assert "all_functions_snapshots" in var_dict
        assert var_dict["all_functions_snapshots"]["variablesReference"] > 0

        # Expand all_functions_snapshots
        all_vars = store.get_variables(
            var_dict["all_functions_snapshots"]["variablesReference"]
        )
        assert len(all_vars) == 2

    def test_module_scope_no_snapshots_no_expand(self):
        store = VariableStore()
        scopes = store.scopes_for_module_stop("func", 0, 1, [])
        variables = store.get_variables(scopes[0]["variablesReference"])

        names = [v["name"] for v in variables]
        assert "function" in names
        assert "progress" in names
        assert "all_functions_snapshots" not in names
