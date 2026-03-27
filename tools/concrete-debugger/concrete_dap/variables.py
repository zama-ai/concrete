"""Convert NodeSnapshots into DAP variable trees."""

from typing import Any, Optional

import numpy as np


class VariableStore:
    """Manages DAP variable references for lazy expansion of complex values."""

    def __init__(self):
        self._next_ref = 1
        self._refs: dict[int, Any] = {}

    def reset(self):
        """Clear all variable references."""
        self._next_ref = 1
        self._refs.clear()

    def _alloc_ref(self, obj: Any) -> int:
        """Allocate a variables reference for an expandable object."""
        ref = self._next_ref
        self._next_ref += 1
        self._refs[ref] = obj
        return ref

    def scopes_for_stop(self, current_snapshot, all_snapshots: list) -> list[dict]:
        """Build DAP Scope objects for a stopped state."""
        scopes = []

        current_ref = self._alloc_ref(("current_node", current_snapshot))
        scopes.append({
            "name": "Current Node",
            "variablesReference": current_ref,
            "expensive": False,
        })

        all_ref = self._alloc_ref(("all_evaluated", all_snapshots))
        scopes.append({
            "name": "All Evaluated",
            "variablesReference": all_ref,
            "expensive": False,
        })

        return scopes

    def get_variables(self, variables_ref: int) -> list[dict]:
        """Resolve a variablesReference into DAP Variable objects."""
        obj = self._refs.get(variables_ref)
        if obj is None:
            return []

        if isinstance(obj, tuple) and len(obj) == 2:
            tag, data = obj

            if tag == "current_node":
                return self._snapshot_variables(data)

            if tag == "all_evaluated":
                return self._snapshot_list_variables(data)

            if tag == "ndarray":
                return self._ndarray_variables(data)

            if tag == "module_context":
                return self._module_context_variables(data)

        return []

    def _snapshot_variables(self, snapshot) -> list[dict]:
        """Build variables for a single NodeSnapshot."""
        variables = []
        value = snapshot.value

        # Value — possibly expandable if ndarray
        if isinstance(value, np.ndarray) and value.size > 8:
            ref = self._alloc_ref(("ndarray", value))
            variables.append({
                "name": "value",
                "value": f"array(shape={value.shape}, min={value.min()}, max={value.max()})",
                "variablesReference": ref,
            })
        else:
            variables.append({
                "name": "value",
                "value": _format_value(value),
                "variablesReference": 0,
            })

        variables.append({
            "name": "operation",
            "value": snapshot.operation_name,
            "variablesReference": 0,
        })
        variables.append({
            "name": "encrypted",
            "value": str(snapshot.is_encrypted),
            "variablesReference": 0,
        })

        # Bit width
        bit_width = _get_bit_width(snapshot.node)
        if bit_width is not None:
            variables.append({
                "name": "bit_width",
                "value": str(bit_width),
                "variablesReference": 0,
            })

        variables.append({
            "name": "overflow",
            "value": str(snapshot.overflow),
            "variablesReference": 0,
        })

        variables.append({
            "name": "tag",
            "value": snapshot.tag or "(none)",
            "variablesReference": 0,
        })
        variables.append({
            "name": "location",
            "value": snapshot.location or "(unknown)",
            "variablesReference": 0,
        })

        # Bounds
        if snapshot.node.bounds is not None:
            lower, upper = snapshot.node.bounds
            variables.append({
                "name": "bounds",
                "value": f"[{lower}, {upper}]",
                "variablesReference": 0,
            })

        return variables

    def _snapshot_list_variables(self, snapshots: list) -> list[dict]:
        """Build variables for a list of snapshots (expandable)."""
        variables = []
        for snap in snapshots:
            ref = self._alloc_ref(("current_node", snap))
            variables.append({
                "name": f"[{snap.index}] {snap.operation_name}",
                "value": _format_value(snap.value),
                "variablesReference": ref,
            })
        return variables

    def scopes_for_module_stop(self, function_name: str, current_idx: int,
                               total_functions: int,
                               all_snapshots: list) -> list[dict]:
        """Build a Module Context scope for module debug sessions."""
        ref = self._alloc_ref(("module_context", {
            "function_name": function_name,
            "current_idx": current_idx,
            "total_functions": total_functions,
            "all_snapshots": all_snapshots,
        }))
        return [{
            "name": "Module Context",
            "variablesReference": ref,
            "expensive": False,
        }]

    def _module_context_variables(self, ctx: dict) -> list[dict]:
        """Build variables for the Module Context scope."""
        variables = [
            {
                "name": "function",
                "value": ctx["function_name"],
                "variablesReference": 0,
            },
            {
                "name": "progress",
                "value": f"{ctx['current_idx'] + 1}/{ctx['total_functions']}",
                "variablesReference": 0,
            },
        ]

        if ctx["all_snapshots"]:
            ref = self._alloc_ref(("all_evaluated", ctx["all_snapshots"]))
            variables.append({
                "name": "all_functions_snapshots",
                "value": f"{len(ctx['all_snapshots'])} snapshots",
                "variablesReference": ref,
            })

        return variables

    def _ndarray_variables(self, arr: np.ndarray) -> list[dict]:
        """Expand an ndarray into indexed elements."""
        variables = []
        flat = arr.flatten()
        for i, val in enumerate(flat[:200]):  # cap at 200 elements
            variables.append({
                "name": f"[{i}]",
                "value": str(val),
                "variablesReference": 0,
            })
        if flat.size > 200:
            variables.append({
                "name": "...",
                "value": f"({flat.size - 200} more elements)",
                "variablesReference": 0,
            })
        return variables


def _format_value(value) -> str:
    """Format a numpy value for display."""
    if isinstance(value, np.ndarray):
        if value.size <= 8:
            return repr(value)
        return f"array(shape={value.shape}, min={value.min()}, max={value.max()})"
    return repr(value)


def _get_bit_width(node) -> Optional[int]:
    """Extract bit width from a node's output dtype, if available."""
    try:
        return node.output.dtype.bit_width
    except AttributeError:
        return None
