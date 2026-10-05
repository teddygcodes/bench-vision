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
actual set, then capture (or grid) for an overview and capture_region / capture_cell to
zoom in. save_reference before rework and compare afterwards to see what changed.
set_control adjusts focus/exposure/gain (real cameras need Linux/v4l2). Region
coordinates are full-resolution pixels of the rotated frame; every reply states the
frame size and the scale from the returned image."""


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

    @mcp.tool()
    @_guard
    def capture_region(
        cam: str, x: int, y: int, w: int, h: int, rotate: int = 0, max_edge: int = 768
    ) -> list[Image | str]:
        """Main inspection tool: crop a fresh full-resolution frame and zoom it.

        x, y, w, h: top-left corner and size in FULL-RESOLUTION pixels of the rotated
            frame (capture/grid report the frame size and the scale from their images).
            Boxes partly outside the frame are clipped; the reply says so.
        rotate: extra clockwise rotation (multiple of 90) added to the camera default;
            use the same value as the capture you measured on.
        max_edge: the crop is scaled (up or down) so its long edge is this many px.
        """
        return bv.capture_region(cam, x, y, w, h, rotate, max_edge)

    @mcp.tool()
    @_guard
    def grid(cam: str, rows: int, cols: int, rotate: int = 0, max_edge: int = 1024) -> list[Image | str]:
        """Capture the full frame with a labeled rows x cols grid (A1 = top-left).

        Rows are letters A.., columns numbers 1... The grid is remembered per camera
        so the user can say "look at B3" and you call capture_cell(cam, "B3").
        """
        return bv.grid(cam, rows, cols, rotate, max_edge)

    @mcp.tool()
    @_guard
    def capture_cell(cam: str, cell: str, max_edge: int = 768, margin: float = 0.0) -> list[Image | str]:
        """capture_region for one cell (e.g. "B3") of the last grid drawn on this camera.

        margin: extra border as a fraction of the cell size on each side (e.g. 0.25),
            useful when a joint straddles a grid line.
        """
        return bv.capture_cell(cam, cell, max_edge, margin)

    @mcp.tool()
    @_guard
    def save_reference(cam: str, name: str, rotate: int = 0, max_edge: int = 1024) -> list[Image | str]:
        """Capture now and keep the full-resolution frame as a named reference (e.g. "u1-before").

        Saving under an existing name replaces it. rotate works as in capture; compare
        later captures in the same orientation.
        """
        return bv.save_reference(cam, name, rotate, max_edge)

    @mcp.tool()
    @_guard
    def compare(cam: str, name: str, max_edge: int = 1024) -> list[Image | str]:
        """Capture now and compare with a saved reference: returns reference|now side by side,
        then an absdiff heatmap, plus the largest changed regions in full-res pixels.
        No alignment is done (the cameras are fixed on booms)."""
        return bv.compare(cam, name, max_edge)

    @mcp.tool()
    @_guard
    def set_control(cam: str, control: str, value: int) -> str:
        """Set a camera control via v4l2-ctl (e.g. focus_absolute, exposure_time_absolute, gain).

        The name is checked against the camera's --list-ctrls; an unknown name lists the
        available ones with their ranges. Manual focus/exposure need their auto mode off
        first (focus_automatic_continuous=0, auto_exposure=1). The value is re-applied each
        time the camera opens until the server restarts. Needs Linux/v4l2 for real cameras.
        """
        return bv.set_control(cam, control, value)

    return mcp
