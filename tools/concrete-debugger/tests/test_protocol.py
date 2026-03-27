"""Tests for DAP protocol message I/O."""

import io
import json

import numpy as np
import pytest

from concrete_dap.protocol import (
    make_event,
    make_response,
    read_message,
    write_message,
)


def _encode_dap(msg: dict) -> bytes:
    """Encode a dict as a Content-Length framed DAP message."""
    body = json.dumps(msg).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body


class TestReadMessage:
    def test_basic_read(self):
        msg = {"seq": 1, "type": "request", "command": "initialize"}
        stream = io.BytesIO(_encode_dap(msg))
        result = read_message(stream)
        assert result == msg

    def test_empty_stream(self):
        stream = io.BytesIO(b"")
        result = read_message(stream)
        assert result is None

    def test_multiple_messages(self):
        msg1 = {"seq": 1, "type": "request", "command": "initialize"}
        msg2 = {"seq": 2, "type": "request", "command": "launch"}
        stream = io.BytesIO(_encode_dap(msg1) + _encode_dap(msg2))
        assert read_message(stream) == msg1
        assert read_message(stream) == msg2

    def test_zero_content_length(self):
        stream = io.BytesIO(b"Content-Length: 0\r\n\r\n")
        result = read_message(stream)
        assert result is None


class TestWriteMessage:
    def test_basic_write(self):
        msg = {"seq": 1, "type": "response", "success": True}
        stream = io.BytesIO()
        write_message(msg, stream)
        stream.seek(0)
        result = read_message(stream)
        assert result == msg

    def test_numpy_serialization(self):
        msg = {"value": np.int64(42)}
        stream = io.BytesIO()
        write_message(msg, stream)
        stream.seek(0)
        result = read_message(stream)
        assert result["value"] == 42

    def test_numpy_array_serialization(self):
        msg = {"arr": np.array([1, 2, 3])}
        stream = io.BytesIO()
        write_message(msg, stream)
        stream.seek(0)
        result = read_message(stream)
        assert result["arr"] == [1, 2, 3]

    def test_numpy_bool_serialization(self):
        msg = {"flag": np.bool_(True)}
        stream = io.BytesIO()
        write_message(msg, stream)
        stream.seek(0)
        result = read_message(stream)
        assert result["flag"] is True


class TestMakeResponse:
    def test_success_response(self):
        req = {"seq": 5, "type": "request", "command": "initialize"}
        resp = make_response(req, body={"supportsConfigurationDone": True})
        assert resp["type"] == "response"
        assert resp["request_seq"] == 5
        assert resp["command"] == "initialize"
        assert resp["success"] is True
        assert resp["body"]["supportsConfigurationDone"] is True

    def test_error_response(self):
        req = {"seq": 3, "type": "request", "command": "launch"}
        resp = make_response(req, success=False, message="file not found")
        assert resp["success"] is False
        assert resp["message"] == "file not found"

    def test_no_body(self):
        req = {"seq": 1, "type": "request", "command": "disconnect"}
        resp = make_response(req)
        assert "body" not in resp


class TestMakeEvent:
    def test_event_with_body(self):
        evt = make_event("stopped", {"reason": "breakpoint", "threadId": 1})
        assert evt["type"] == "event"
        assert evt["event"] == "stopped"
        assert evt["body"]["reason"] == "breakpoint"

    def test_event_without_body(self):
        evt = make_event("initialized")
        assert evt["event"] == "initialized"
        assert "body" not in evt
