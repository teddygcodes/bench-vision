"""End-to-end: launch `bench-vision serve --mock` as a subprocess and talk MCP over stdio."""

import asyncio
import os
import sys
from pathlib import Path

from conftest import images_of, jpeg_size

SRC = Path(__file__).resolve().parent.parent / "src"


def test_serve_mock_over_stdio(root):
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "bench_vision", "serve", "--mock", "--root", str(root)],
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )

    async def go():
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("capture", {"cam": "scope"})
            return tools, result

    tools, result = asyncio.run(go())
    assert {"list_cameras", "capture"} <= tools
    assert not result.is_error
    assert jpeg_size(images_of(result.content)[0]) == (1024, 576)
    assert list((root / "captures").rglob("scope_*.jpg"))


def test_stdout_stays_pure_jsonrpc_with_opencv_debug_logging(root):
    import json
    import subprocess

    init = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}},
    }
    proc = subprocess.run(
        [sys.executable, "-m", "bench_vision", "serve", "--mock", "--root", str(root)],
        input=json.dumps(init) + "\n",
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(SRC), "OPENCV_LOG_LEVEL": "DEBUG"},
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, proc.stderr
    for ln in lines:
        assert json.loads(ln)["jsonrpc"] == "2.0"
