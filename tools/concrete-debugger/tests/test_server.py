"""Integration tests for the DAP server — send raw DAP messages, verify responses."""

import io
import json
import os
import tempfile
from typing import Optional

import numpy as np
import pytest

from concrete_dap.protocol import read_message
from concrete_dap.server import DAPServer


def _encode_dap(msg: dict) -> bytes:
    body = json.dumps(msg).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body


def _make_request(seq: int, command: str, arguments: Optional[dict] = None) -> dict:
    req = {"seq": seq, "type": "request", "command": command}
    if arguments is not None:
        req["arguments"] = arguments
    return req


class DAPTestClient:
    """Helper to drive a DAPServer with in-memory streams."""

    def __init__(self):
        self._input = io.BytesIO()
        self._output = io.BytesIO()
        self._seq = 1

    def send(self, command: str, arguments: Optional[dict] = None):
        req = _make_request(self._seq, command, arguments)
        self._seq += 1
        self._input.write(_encode_dap(req))

    def run_server(self):
        self._input.seek(0)
        server = DAPServer(self._input, self._output)
        server.run()
        self._output.seek(0)

    def read_all_messages(self) -> list[dict]:
        messages = []
        while True:
            msg = read_message(self._output)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def find_response(self, messages: list[dict], command: str) -> Optional[dict]:
        for msg in messages:
            if msg.get("type") == "response" and msg.get("command") == command:
                return msg
        return None

    def find_event(self, messages: list[dict], event: str) -> Optional[dict]:
        for msg in messages:
            if msg.get("type") == "event" and msg.get("event") == event:
                return msg
        return None

    def find_events(self, messages: list[dict], event: str) -> list[dict]:
        return [m for m in messages if m.get("type") == "event" and m.get("event") == event]


class TestInitializeDisconnect:
    def test_initialize(self):
        client = DAPTestClient()
        client.send("initialize", {"adapterID": "concrete"})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        init_resp = client.find_response(msgs, "initialize")
        assert init_resp is not None
        assert init_resp["success"] is True
        assert init_resp["body"]["supportsConfigurationDoneRequest"] is True

        init_evt = client.find_event(msgs, "initialized")
        assert init_evt is not None

    def test_disconnect(self):
        client = DAPTestClient()
        client.send("initialize", {"adapterID": "concrete"})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        disc_resp = client.find_response(msgs, "disconnect")
        assert disc_resp is not None
        assert disc_resp["success"] is True


class TestThreads:
    def test_threads(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("threads")
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "threads")
        assert resp is not None
        assert len(resp["body"]["threads"]) == 1
        assert resp["body"]["threads"][0]["name"] == "FHE Circuit Evaluation"


class TestLaunchErrors:
    def test_missing_program(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("launch", {"function": "f"})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "launch")
        assert resp["success"] is False
        assert "program" in resp["message"]

    def test_missing_function(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("launch", {"program": "test.py"})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "launch")
        assert resp["success"] is False
        assert "function" in resp["message"]

    def test_nonexistent_program(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("launch", {"program": "/nonexistent.py", "function": "f"})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "launch")
        assert resp["success"] is False


class TestUnknownCommand:
    def test_unknown(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("foobar")
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "foobar")
        assert resp is not None
        assert resp["success"] is False


class TestSetBreakpointsWithoutSession:
    def test_set_breakpoints_no_session(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("setBreakpoints", {
            "source": {"path": "test.py"},
            "breakpoints": [{"line": 5}],
        })
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "setBreakpoints")
        assert resp is not None
        assert resp["success"] is True
        # No nodes, so breakpoints won't be verified
        assert resp["body"]["breakpoints"][0]["verified"] is False


class TestStackTraceWithoutSession:
    def test_empty_stack(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("stackTrace", {"threadId": 1})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "stackTrace")
        assert resp["body"]["stackFrames"] == []
        assert resp["body"]["totalFrames"] == 0


class TestScopesWithoutSession:
    def test_empty_scopes(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("scopes", {"frameId": 0})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "scopes")
        assert resp["body"]["scopes"] == []


class TestVariablesWithoutSession:
    def test_empty_variables(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("variables", {"variablesReference": 1})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "variables")
        assert resp["body"]["variables"] == []


class TestEvaluateWithoutSession:
    def test_no_session(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("evaluate", {"expression": "value"})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "evaluate")
        assert "no active session" in resp["body"]["result"]


class TestSetFunctionBreakpoints:
    def test_returns_empty(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("setFunctionBreakpoints", {"breakpoints": []})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "setFunctionBreakpoints")
        assert resp["success"] is True


class TestSetExceptionBreakpoints:
    def test_returns_ok(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("setExceptionBreakpoints", {"filters": []})
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "setExceptionBreakpoints")
        assert resp["success"] is True


class TestConfigurationDoneWithoutSession:
    def test_ok(self):
        client = DAPTestClient()
        client.send("initialize")
        client.send("configurationDone")
        client.send("disconnect")
        client.run_server()

        msgs = client.read_all_messages()
        resp = client.find_response(msgs, "configurationDone")
        assert resp["success"] is True
