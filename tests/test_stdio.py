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


def test_capture_cli_mock(root, capsys):
    from bench_vision.cli import main

    out = root / "x.jpg"
    assert main(["capture", "scope", "--mock", "--root", str(root), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "scope: full frame 1920x1080" in text and "Saved captures/" in text
    assert jpeg_size(out.read_bytes()) == (1024, 576)


def test_capture_cli_errors_are_one_line(root, capsys):
    from bench_vision.cli import main

    assert main(["capture", "nope", "--mock", "--root", str(root)]) == 1
    err = capsys.readouterr().err.strip()
    assert "Unknown camera 'nope'" in err and "\n" not in err and "Traceback" not in err


def test_capture_cli_unwritable_out_is_one_line(root, capsys):
    from bench_vision.cli import main

    assert main(["capture", "scope", "--mock", "--root", str(root), "--out", str(root)]) == 1
    err = capsys.readouterr().err.strip()
    assert "could not write --out" in err and "\n" not in err and "Traceback" not in err


def test_ctrl_c_is_one_line_even_twice(root):
    """SIGINT during startup imports, or twice during a slow capture: one line, exit 130, no traceback."""
    import signal
    import subprocess
    import time

    slow = (
        "import sys\n"
        "from bench_vision.cli import main\n"
        "import bench_vision.app  # noqa: F401  (heavy imports done before patching)\n"
        "import time, bench_vision.camera as c; orig = c.MockBackend.grab\n"
        "c.MockBackend.grab = lambda self, cam, ctl: (time.sleep(6), orig(self, cam, ctl))[1]\n"
        "print('ready', flush=True); sys.exit(main(sys.argv[1:]))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    args = ["capture", "scope", "--mock", "--root", str(root)]

    # 1) the real main(), interrupted while it is importing mcp/cv2: an import hook pauses exactly there
    during_import = (
        "import sys, time\n"
        "class Pause:\n"
        "    done = False\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'mcp' and not Pause.done:\n"
        "            Pause.done = True\n"
        "            print('ready', flush=True)\n"
        "            time.sleep(10)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Pause())\n"
        "from bench_vision.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", during_import, *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=20)
    assert "Traceback" not in err and proc.returncode == 130 and "interrupted" in err, (proc.returncode, err)

    # 2) twice, while a (slow) capture is in flight
    proc = subprocess.Popen([sys.executable, "-c", slow, *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    time.sleep(1.0)
    proc.send_signal(signal.SIGINT)
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=20)
    assert "Traceback" not in err and proc.returncode == 130 and "interrupted" in err, (proc.returncode, err)
