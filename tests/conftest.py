import asyncio
import io
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from bench_vision.app import BenchVision
from bench_vision.server import build_server


def make_image(path: Path, w: int, h: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    # a distinct marker in the top-left quadrant so crops/rotations are checkable
    img[: h // 4, : w // 4] = (0, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return cv2.imread(str(path))


class Clock:
    """Deterministic clock advancing one second per call."""

    def __init__(self):
        self.t = datetime(2026, 10, 4, 12, 0, 0)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_image(tmp_path / "mock" / "scope.png", 1920, 1080, seed=1)
    make_image(tmp_path / "mock" / "side.png", 3840, 2160, seed=2)
    return tmp_path


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def bv(root, clock) -> BenchVision:
    return BenchVision(root, mock=True, clock=clock)


def jpeg_size(data: bytes) -> tuple[int, int]:
    img = Image.open(io.BytesIO(data))
    assert img.format == "JPEG"
    return img.size


def call(bv: BenchVision, tool: str, **args):
    """Call a tool through a real in-process MCP client; return (is_error, content blocks)."""
    return asyncio.run(_call(bv, tool, args))


async def _call(bv, tool, args):
    from mcp import Client

    async with Client(build_server(bv)) as client:
        result = await client.call_tool(tool, args)
        return result.is_error, result.content


def list_tool_names(bv: BenchVision) -> list[str]:
    async def go():
        from mcp import Client

        async with Client(build_server(bv)) as client:
            return [t.name for t in (await client.list_tools()).tools]

    return asyncio.run(go())


def text_of(content) -> str:
    return "\n".join(c.text for c in content if c.type == "text")


def images_of(content) -> list[bytes]:
    import base64

    return [base64.b64decode(c.data) for c in content if c.type == "image"]
