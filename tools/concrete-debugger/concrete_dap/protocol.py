"""DAP message types and Content-Length framed I/O."""

import json
import sys
from typing import Any, Optional


def read_message(stream=None) -> Optional[dict]:
    """Read a DAP message with Content-Length framing from stream."""
    if stream is None:
        stream = sys.stdin.buffer

    headers = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.decode("utf-8").rstrip("\r\n")
        if line == "":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    content_length = int(headers.get("Content-Length", "0"))
    if content_length == 0:
        return None

    body = stream.read(content_length)
    if not body:
        return None

    return json.loads(body.decode("utf-8"))


def write_message(msg: dict, stream=None) -> None:
    """Write a DAP message with Content-Length framing to stream."""
    if stream is None:
        stream = sys.stdout.buffer

    body = json.dumps(msg, default=_json_default).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header)
    stream.write(body)
    stream.flush()


def _json_default(obj):
    """JSON serializer for numpy types."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def make_response(request: dict, body: Optional[dict] = None, success: bool = True,
                  message: str = "") -> dict:
    """Build a DAP response message."""
    resp = {
        "seq": 0,
        "type": "response",
        "request_seq": request.get("seq", 0),
        "command": request.get("command", ""),
        "success": success,
    }
    if body is not None:
        resp["body"] = body
    if message:
        resp["message"] = message
    return resp


def make_event(event: str, body: Optional[dict] = None) -> dict:
    """Build a DAP event message."""
    evt = {
        "seq": 0,
        "type": "event",
        "event": event,
    }
    if body is not None:
        evt["body"] = body
    return evt
