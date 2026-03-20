"""DAP server: stdin/stdout message loop and dispatch."""

import os
import runpy
import sys
import traceback
from typing import Any, Optional

from .breakpoints import BreakpointManager
from .protocol import make_event, make_response, read_message, write_message
from .session import ConcreteDebugSession, ModuleDebugSession, StopReason
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

    def _send_output(self, text: str, category: str = "console") -> None:
        """Send a DAP output event to the Debug Console."""
        self._send(make_event("output", {
            "category": category,
            "output": text + "\n",
        }))

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
        functions_config = args.get("functions")
        input_args = tuple(args.get("args", []))
        stop_on_entry = args.get("stopOnEntry", True)
        stop_on_overflow = args.get("stopOnOverflow", False)

        if not program:
            self._send(make_response(request, success=False,
                                     message="'program' is required in launch config"))
            return

        if not function_name:
            self._send(make_response(request, success=False,
                                     message="'function' is required in launch config"))
            return

        self._send_output("Loading script...")

        # Execute the user script to find the circuit.
        # Change to the script's directory so that .artifacts/ and other
        # relative paths land next to the script, not in the (potentially
        # read-only) working directory inherited from VS Code.
        script_dir = os.path.dirname(os.path.abspath(program))
        self._breakpoints.set_script_dir(script_dir)
        prev_cwd = os.getcwd()
        try:
            os.chdir(script_dir)
        except OSError:
            pass  # best-effort; if it fails we'll still try to run

        try:
            namespace = runpy.run_path(program, run_name="__main__")
        except Exception as e:
            self._send_output(f"Error: {e}", "stderr")
            self._send(make_response(request, success=False,
                                     message=f"Failed to execute {program}: {e}"))
            return
        finally:
            os.chdir(prev_cwd)

        # Find the circuit object
        circuit_obj = namespace.get(function_name)
        if circuit_obj is None:
            self._send(make_response(request, success=False,
                                     message=f"'{function_name}' not found in {program}"))
            return

        # Check for module with functions config
        if functions_config:
            module_graphs = _extract_module_graphs(circuit_obj)
            if module_graphs is None:
                self._send(make_response(request, success=False,
                                         message=f"'{function_name}' is not a module (no .graphs attribute)"))
                return
            self._handle_launch_module(request, module_graphs, functions_config,
                                       stop_on_entry, stop_on_overflow)
            return

        graph = _extract_graph(circuit_obj)
        if graph is None:
            self._send(make_response(request, success=False,
                                     message=f"'{function_name}' is not a Circuit or Compiler object"))
            return

        self._send_output(f"Found circuit with {len(graph.graph.nodes)} nodes")
        self._send_output(f"Evaluating inputs: {input_args}")

        # Create debug session
        self._session = ConcreteDebugSession(graph, input_args, self._breakpoints,
                                             stop_on_overflow=stop_on_overflow)
        self._emit_graph_summary(graph)
        self._send(make_response(request))

    def _handle_disconnect(self, request: dict) -> None:
        self._send(make_response(request))
        self._running = False

    def _handle_configuration_done(self, request: dict) -> None:
        self._send(make_response(request))
        if not self._session:
            return
        # Now that VS Code is fully configured, start evaluation.
        reason = self._session.evaluate_inputs_and_stop_on_entry()
        if self._launch_config.get("stopOnEntry", True):
            # Pause before the first non-input node
            self._send_stopped_event(reason)
        else:
            # Run until breakpoint or end
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

        if isinstance(self._session, ModuleDebugSession):
            scopes.extend(self._variables.scopes_for_module_stop(
                self._session.current_function_name,
                self._session._current_idx,
                len(self._session._sessions),
                self._session.all_snapshots,
            ))

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
        self._send(make_response(request))
        if self._session:
            if isinstance(self._session, ModuleDebugSession):
                reason = self._session.step_into_next_function()
            else:
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

    # ── Module Launch ──

    def _handle_launch_module(self, request: dict, module_graphs: dict,
                               functions_config: list, stop_on_entry: bool,
                               stop_on_overflow: bool) -> None:
        """Launch a module debug session with multiple functions."""
        named_sessions = []
        for func_conf in functions_config:
            name = func_conf.get("name", "")
            func_args = tuple(func_conf.get("args", []))

            if name not in module_graphs:
                avail = list(module_graphs.keys())
                self._send(make_response(
                    request, success=False,
                    message=f"Function '{name}' not found in module. Available: {avail}"))
                return

            graph = module_graphs[name]
            session = ConcreteDebugSession(graph, func_args, self._breakpoints,
                                           stop_on_overflow=stop_on_overflow)
            named_sessions.append((name, session))
            self._send_output(f"Function '{name}': {len(graph.graph.nodes)} nodes")
            self._emit_graph_summary(graph)

        self._session = ModuleDebugSession(named_sessions, self._breakpoints)
        self._send(make_response(request))

    def _emit_graph_summary(self, graph) -> None:
        """Emit a compact circuit summary to the Debug Console."""
        total = len(graph.graph.nodes)
        input_count = len(getattr(graph, 'input_nodes', {}))
        output_count = len(getattr(graph, 'output_nodes', {}))

        ops = set()
        input_set = set(getattr(graph, 'input_nodes', {}).values())
        for n in graph.graph.nodes:
            if n not in input_set:
                name = getattr(n, 'properties', {}).get("name", "")
                if name:
                    ops.add(name)

        has_tags = any(getattr(n, 'tag', '') for n in graph.graph.nodes)

        lines = [
            "=== Circuit Summary ===",
            f"  Nodes: {total} ({input_count} inputs, {output_count} outputs)",
            f"  Operations: {', '.join(sorted(ops)) if ops else '(none)'}",
            f"  Tags: {'yes' if has_tags else 'no'}",
        ]
        self._send_output("\n".join(lines))

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
            StopReason.OVERFLOW: "data breakpoint",
        }

        body: dict = {
            "reason": reason_map.get(reason, "step"),
            "threadId": THREAD_ID,
            "allThreadsStopped": True,
        }

        if reason == StopReason.EXCEPTION and self._session and self._session.error:
            body["text"] = str(self._session.error)
            body["description"] = "Node evaluation failed"
            snap = self._session.current_snapshot
            if snap:
                self._send_output(
                    f"Exception at node [{snap.index}] {snap.operation_name}: "
                    f"{self._session.error}", "stderr")

        if reason == StopReason.OVERFLOW and self._session:
            snap = self._session.current_snapshot
            if snap:
                body["text"] = f"Overflow at [{snap.index}] {snap.operation_name}"
                body["description"] = "Value exceeds bit width"
                self._send_output(
                    f"Overflow: node [{snap.index}] {snap.operation_name}, "
                    f"value={repr(snap.value)}", "important")

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

        # Module-specific expressions
        if isinstance(self._session, ModuleDebugSession):
            if expression == "functions":
                names = [name for name, _ in self._session._sessions]
                return f"Functions: {', '.join(names)}"
            if expression == "function":
                return f"Current function: {self._session.current_function_name}"

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


def _extract_module_graphs(obj) -> dict | None:
    """Extract a dict of graphs from an FheModule (obj.graphs)."""
    graphs = getattr(obj, 'graphs', None)
    if isinstance(graphs, dict):
        # Verify at least one entry looks like a Graph
        for g in graphs.values():
            if hasattr(g, 'graph') and hasattr(g, 'input_indices'):
                return graphs
    return None
