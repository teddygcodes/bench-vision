"""MCP tool definitions."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from .app import BenchVision, Result
from .errors import BenchVisionError

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Eyes on a soldering bench. Typically 'scope' (microscope, straight down) and 'side'
(low oblique from the left, shows fillet profile and the iron); call list_cameras for the
actual set, then capture for an overview. capture reports the full-resolution frame size."""


def _to_content(result: Result | str) -> list[Image | str] | str:
    if isinstance(result, str):
        return result
    return [Image(data=item, format="jpeg") if isinstance(item, bytes) else item for item in result]


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn every failure into a one-line message; tracebacks go to the server log only."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return _to_content(fn(*args, **kwargs))
        except BenchVisionError as e:
            raise ToolError(str(e)) from None
        except Exception as e:  # noqa: BLE001 - last line of defence
            log.exception("unexpected error in %s", fn.__name__)
            raise ToolError(
                f"bench-vision hit an unexpected internal error in {fn.__name__} ({type(e).__name__}). "
                "This is a bug; details are in the server's stderr log."
            ) from None

    return wrapper


def _describe_validation(tool: str, err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        field = ".".join(str(p) for p in e["loc"]) or "arguments"
        if e["type"] == "missing":
            parts.append(f"'{field}' is required")
        else:
            got = repr(e.get("input"))
            got = got if len(got) <= 60 else got[:57] + "..."
            msg = e["msg"] or "is invalid"
            parts.append(f"'{field}' {msg[:1].lower()}{msg[1:]} (got {got})")
    return f"Invalid arguments for {tool}: " + "; ".join(parts) + "."


class BenchServer(MCPServer):
    """MCPServer whose argument-validation errors are short sentences, not pydantic dumps."""

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as e:
            if isinstance(e.__cause__, ValidationError):
                raise ToolError(_describe_validation(name, e.__cause__)) from None
            raise


def build_server(bv: BenchVision) -> BenchServer:
    mcp = BenchServer("bench-vision", instructions=INSTRUCTIONS)

    @mcp.tool()
    @_guard
    def list_cameras() -> str:
        """List configured cameras: name, device path, connected, open/closed, resolution."""
        return bv.list_cameras()

    @mcp.tool()
    @_guard
    def capture(cam: str, rotate: int = 0, max_edge: int = 1024) -> list[Image | str]:
        """Capture one still from a camera (JPEG, long edge <= max_edge px).

        cam: camera name from list_cameras (e.g. "scope", "side").
        rotate: extra clockwise rotation in degrees (multiple of 90), added to the
            camera's default_rotation.
        max_edge: long edge of the returned image; keep the default unless you need more.
        """
        return bv.capture(cam, rotate, max_edge)

    return mcp
