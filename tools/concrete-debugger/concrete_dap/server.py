"""DAP server: stdin/stdout message loop and dispatch."""

import runpy
import sys
import traceback
from typing import Any, Optional

from .breakpoints import BreakpointManager
from .protocol import make_event, make_response, read_message, write_message
from .session import ConcreteDebugSession, StopReason
from .variables import VariableStore

THREAD_ID = 1


class DAPServer:
    """Debug Adapter Protocol server for Concrete FHE circuits."""

    def __init__(self, input_stream=None, output_stream=None):
        self._input = input_stream or sys.stdin.buffer
        self._output = output_stream or sys.stdout.buffer
        self._seq = 1
        self._session: Optional[ConcreteDebugSession] = None
        self._breakpoints = BreakpointManager()
        self._variables = VariableStore()
        self._launch_config: dict = {}
        self._running = True
        self._initialized = False

        self._handlers = {
            "initialize": self._handle_initialize,
            "launch": self._handle_launch,
            "disconnect": self._handle_disconnect,
            "setBreakpoints": self._handle_set_breakpoints,
            "setFunctionBreakpoints": self._handle_set_function_breakpoints,
            "setExceptionBreakpoints": self._handle_set_exception_breakpoints,
            "configurationDone": self._handle_configuration_done,
            "threads": self._handle_threads,
            "stackTrace": self._handle_stack_trace,
            "scopes": self._handle_scopes,
            "variables": self._handle_variables,
            "continue": self._handle_continue,
            "next": self._handle_next,
            "stepIn": self._handle_step_in,
            "stepOut": self._handle_step_out,
            "evaluate": self._handle_evaluate,
            "pause": self._handle_pause,
            "source": self._handle_source,
        }

    def run(self) -> None:
        """Main message loop."""
        while self._running:
            msg = read_message(self._input)
            if msg is None:
                break
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        """Route a DAP message to its handler."""
        msg_type = msg.get("type", "")
        if msg_type != "request":
            return

        command = msg.get("command", "")
        handler = self._handlers.get(command)

        if handler is None:
            self._send(make_response(msg, success=False,
                                     message=f"Unknown command: {command}"))
            return

        try:
            handler(msg)
        except Exception as e:
            self._send(make_response(msg, success=False,
                                     message=f"Error handling {command}: {e}"))
            traceback.print_exc(file=sys.stderr)

    def _send(self, msg: dict) -> None:
        """Send a DAP message with auto-incrementing sequence number."""
        msg["seq"] = self._seq
        self._seq += 1
        write_message(msg, self._output)

    # ── Lifecycle ──

    def _handle_initialize(self, request: dict) -> None:
        capabilities = {
            "supportsConfigurationDoneRequest": True,
            "supportsEvaluateForHovers": True,
            "supportsSingleThreadExecutionRequests": True,
        }
        self._send(make_response(request, body=capabilities))
        self._send(make_event("initialized"))
        self._initialized = True

    def _handle_launch(self, request: dict) -> None:
        args = request.get("arguments", {})
        self._launch_config = args

        program = args.get("program", "")
        function_name = args.get("function", "")
        input_args = tuple(args.get("args", []))
        stop_on_entry = args.get("stopOnEntry", True)

        if not program:
            self._send(make_response(request, success=False,
                                     message="'program' is required in launch config"))
            return

        if not function_name:
            self._send(make_response(request, success=False,
                                     message="'function' is required in launch config"))
            return

        # Execute the user script to find the circuit
        try:
            namespace = runpy.run_path(program, run_name="__main__")
        except Exception as e:
            self._send(make_response(request, success=False,
                                     message=f"Failed to execute {program}: {e}"))
            return

        # Find the circuit object
        circuit_obj = namespace.get(function_name)
        if circuit_obj is None:
            self._send(make_response(request, success=False,
                                     message=f"'{function_name}' not found in {program}"))
            return

        graph = _extract_graph(circuit_obj)
        if graph is None:
            self._send(make_response(request, success=False,
                                     message=f"'{function_name}' is not a Circuit or Compiler object"))
            return

        # Create debug session
        self._session = ConcreteDebugSession(graph, input_args, self._breakpoints)
        self._send(make_response(request))

        if stop_on_entry:
            reason = self._session.evaluate_inputs_and_stop_on_entry()
            self._send_stopped_event(reason)

    def _handle_disconnect(self, request: dict) -> None:
        self._send(make_response(request))
        self._running = False

    def _handle_configuration_done(self, request: dict) -> None:
        self._send(make_response(request))
        # If session exists and not stopped on entry, start running
        if self._session and not self._launch_config.get("stopOnEntry", True):
            reason = self._session.evaluate_inputs_and_stop_on_entry()
            if reason == StopReason.ENTRY:
                reason = self._session.continue_to_breakpoint()
            self._send_stopped_event(reason)

    # ── Breakpoints ──

    def _handle_set_breakpoints(self, request: dict) -> None:
        args = request.get("arguments", {})
        source = args.get("source", {})
        source_path = source.get("path", "")
        bp_lines = [bp.get("line", 0) for bp in args.get("breakpoints", [])]

        results = self._breakpoints.set_breakpoints(source_path, bp_lines)
        self._send(make_response(request, body={"breakpoints": results}))

    def _handle_set_function_breakpoints(self, request: dict) -> None:
        self._send(make_response(request, body={"breakpoints": []}))

    def _handle_set_exception_breakpoints(self, request: dict) -> None:
        self._send(make_response(request))

    # ── Threads ──

    def _handle_threads(self, request: dict) -> None:
        threads = [{"id": THREAD_ID, "name": "FHE Circuit Evaluation"}]
        self._send(make_response(request, body={"threads": threads}))

    # ── Stack / Scopes / Variables ──

    def _handle_stack_trace(self, request: dict) -> None:
        if self._session is None:
            self._send(make_response(request, body={"stackFrames": [], "totalFrames": 0}))
            return

        frames = self._session.get_stack_frames()
        self._send(make_response(request, body={
            "stackFrames": frames,
            "totalFrames": len(frames),
        }))

    def _handle_scopes(self, request: dict) -> None:
        if self._session is None or self._session.current_snapshot is None:
            self._send(make_response(request, body={"scopes": []}))
            return

        self._variables.reset()
        scopes = self._variables.scopes_for_stop(
            self._session.current_snapshot,
            list(self._session.snapshots),
        )
        self._send(make_response(request, body={"scopes": scopes}))

    def _handle_variables(self, request: dict) -> None:
        args = request.get("arguments", {})
        ref = args.get("variablesReference", 0)
        variables = self._variables.get_variables(ref)
        self._send(make_response(request, body={"variables": variables}))

    # ── Stepping ──

    def _handle_continue(self, request: dict) -> None:
        self._send(make_response(request, body={"allThreadsContinued": True}))
        if self._session:
            reason = self._session.continue_to_breakpoint()
            self._send_stopped_event(reason)

    def _handle_next(self, request: dict) -> None:
        self._send(make_response(request))
        if self._session:
            reason = self._session.step_one()
            self._send_stopped_event(reason)

    def _handle_step_in(self, request: dict) -> None:
        # Same as step over in single-graph mode
        self._send(make_response(request))
        if self._session:
            reason = self._session.step_one()
            self._send_stopped_event(reason)

    def _handle_step_out(self, request: dict) -> None:
        self._send(make_response(request))
        if self._session:
            reason = self._session.step_out()
            self._send_stopped_event(reason)

    def _handle_pause(self, request: dict) -> None:
        # Graph evaluation is synchronous, pause is a no-op
        self._send(make_response(request))

    # ── Evaluate ──

    def _handle_evaluate(self, request: dict) -> None:
        args = request.get("arguments", {})
        expression = args.get("expression", "")

        if self._session is None:
            self._send(make_response(request, body={"result": "(no active session)", "variablesReference": 0}))
            return

        result = self._evaluate_expression(expression)
        self._send(make_response(request, body={"result": result, "variablesReference": 0}))

    def _handle_source(self, request: dict) -> None:
        self._send(make_response(request, body={"content": ""}))

    # ── Helpers ──

    def _send_stopped_event(self, reason: StopReason) -> None:
        """Send a DAP 'stopped' event based on the stop reason."""
        if reason == StopReason.FINISHED:
            self._send(make_event("terminated"))
            return

        reason_map = {
            StopReason.STEP: "step",
            StopReason.BREAKPOINT: "breakpoint",
            StopReason.ENTRY: "entry",
            StopReason.EXCEPTION: "exception",
        }

        body: dict = {
            "reason": reason_map.get(reason, "step"),
            "threadId": THREAD_ID,
            "allThreadsStopped": True,
        }

        if reason == StopReason.EXCEPTION and self._session and self._session.error:
            body["text"] = str(self._session.error)
            body["description"] = "Node evaluation failed"

        self._send(make_event("stopped", body))

    def _evaluate_expression(self, expression: str) -> str:
        """Evaluate a debug console expression against the session state."""
        session = self._session
        if session is None:
            return "(no session)"

        # Support querying by snapshot index: e.g. "snap[3]"
        if expression.startswith("snap[") and expression.endswith("]"):
            try:
                idx = int(expression[5:-1])
                if 0 <= idx < len(session.snapshots):
                    snap = session.snapshots[idx]
                    return repr(snap)
                return f"(index {idx} out of range, {len(session.snapshots)} snapshots)"
            except ValueError:
                pass

        # Support "nodes" to get count
        if expression == "nodes":
            return f"{len(session.topo_order)} nodes total, {len(session.snapshots)} evaluated"

        # Support "value" for current
        if expression == "value" and session.current_snapshot:
            return repr(session.current_snapshot.value)

        # Support "overflow" check
        if expression == "overflow":
            overflows = [s for s in session.snapshots if s.overflow]
            if overflows:
                return f"{len(overflows)} overflow(s): " + ", ".join(
                    f"[{s.index}] {s.operation_name}" for s in overflows
                )
            return "No overflows detected"

        return f"(unknown expression: {expression})"


def _extract_graph(obj):
    """Extract a Graph from a Circuit, Compiler/Compilable, or object with .graph."""
    # Circuit object
    if hasattr(obj, "graph"):
        graph = obj.graph
        if hasattr(graph, "graph") and hasattr(graph, "input_indices"):
            return graph

    # Compilable that has been traced
    if hasattr(obj, "_graph"):
        return obj._graph

    return None
