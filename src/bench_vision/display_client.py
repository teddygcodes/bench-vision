"""Push updates from the MCP tools to the `bench-vision display` server."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .errors import BenchVisionError

PUSH_TIMEOUT = 3.0


def push(url: str, payload: dict[str, Any]) -> None:
    """POST one update; any failure becomes a one-line BenchVisionError."""
    req = urllib.request.Request(
        f"{url}/api/push",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # No proxies: the display is local, and an http_proxy setting must not hijack it.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=PUSH_TIMEOUT) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", "")
        except (ValueError, OSError, AttributeError):
            detail = ""
        raise BenchVisionError(f"The wall display at {url} rejected the update: {detail or e.reason}.") from None
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        port = urllib.parse.urlsplit(url).port
        flag = f" --port {port}" if port and port != 8765 else ""
        raise BenchVisionError(
            f"The wall display isn't running at {url}. Start it with `uv run bench-vision display{flag}` "
            "(captures still work without it)."
        ) from None


def b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()
