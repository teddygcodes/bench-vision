"""`bench-vision` command line entry point."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path


def _serve(args: argparse.Namespace) -> int:
    # OpenCV writes INFO/DEBUG logs to stdout, which is the MCP channel. Cap it at
    # WARNING (which goes to stderr) before cv2 is first imported.
    os.environ["OPENCV_LOG_LEVEL"] = "WARNING"
    from .app import BenchVision
    from .server import build_server

    root = Path(args.root).resolve()
    bv = BenchVision(root, mock=args.mock, config_path=Path(args.config).resolve() if args.config else None)
    if bv.config_error:
        # Still start, so the client gets the explanation from every tool call.
        logging.getLogger("bench_vision").error("starting with config error: %s", bv.config_error)
    build_server(bv).run("stdio")
    return 0


def _capture(args: argparse.Namespace) -> int:
    """Run the MCP `capture` tool once, in-process, without Claude: a quick camera check."""
    import asyncio
    import base64

    os.environ["OPENCV_LOG_LEVEL"] = "WARNING"
    # Errors are printed once below; keep library/server log lines out of the way.
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("bench_vision").setLevel(logging.CRITICAL)
    from mcp import Client

    from .app import BenchVision
    from .server import build_server

    bv = BenchVision(Path(args.root).resolve(), mock=args.mock,
                     config_path=Path(args.config).resolve() if args.config else None)

    async def go():
        async with Client(build_server(bv)) as client:
            return await client.call_tool("capture", {"cam": args.cam, "rotate": args.rotate})

    result = asyncio.run(go())
    text = "\n".join(c.text for c in result.content if c.type == "text")
    if result.is_error:
        print(f"bench-vision capture: {text.removeprefix('Error executing tool capture: ')}", file=sys.stderr)
        return 1
    if args.out:
        image = next(c for c in result.content if c.type == "image")
        try:
            Path(args.out).write_bytes(base64.b64decode(image.data))
        except OSError as e:
            print(text)
            print(f"bench-vision capture: could not write --out {args.out}: {e.strerror or e} "
                  "(the capture itself was saved, see above)", file=sys.stderr)
            return 1
        text += f" Returned image written to {args.out}."
    print(text)
    return 0


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = -1
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"must be a port number 1-65535, got {text!r}")
    return value


def _display(args: argparse.Namespace) -> int:
    os.environ["OPENCV_LOG_LEVEL"] = "WARNING"
    from . import imaging
    from .app import BenchVision
    from .display import make_server
    from .live import LiveStream

    root = Path(args.root).resolve()
    bv = BenchVision(root, mock=args.mock, config_path=Path(args.config).resolve() if args.config else None)
    lock = bv.cameras.camera_lock
    if bv.config_error is not None:
        live, cam_name, size, idle = LiveStream(None, lock, f"unavailable: {bv.config_error}"), None, None, 20.0
    else:
        cfg = bv.config
        cam = cfg.cameras[cfg.live_camera]
        w, h = bv.cameras.last_size.get(cam.name, cam.resolution)
        size = (h, w) if imaging.normalize_rotation(cam.default_rotation) in (90, 270) else (w, h)
        # the camera is read in a child process, so freeing it for a capture can't be blocked by a stuck read
        command = [sys.executable, "-m", "bench_vision.livereader", "--root", str(root), "--cam", cam.name]
        if args.mock:
            command.append("--mock")
        if args.config:
            command += ["--config", str(Path(args.config).resolve())]
        live, cam_name, idle = LiveStream(command, lock), cam.name, cfg.live_idle_seconds
    try:
        server, _ = make_server(args.host, args.port, root=root, live=live, live_cam=cam_name, live_size=size,
                                live_idle=idle)
    except OSError as e:
        import errno

        hint = " (is another display already running?)" if e.errno == errno.EADDRINUSE else ""
        print(f"bench-vision display: cannot listen on {args.host}:{args.port}: {e.strerror or e}{hint}",
              file=sys.stderr)
        return 2
    print(f"bench-vision display: open http://{args.host}:{args.port} (Ctrl-C to stop)", file=sys.stderr)
    server.serve_forever()
    return 0


def _call(args: argparse.Namespace) -> int:
    """Run any MCP tool once, in-process, without Claude: `bench-vision call show_step '{"title": ...}'`."""
    import asyncio
    import base64
    import json

    os.environ["OPENCV_LOG_LEVEL"] = "WARNING"
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("bench_vision").setLevel(logging.CRITICAL)
    from mcp import Client

    from .app import BenchVision
    from .server import build_server

    try:
        arguments = json.loads(args.arguments) if args.arguments else {}
    except ValueError as e:
        print(f"bench-vision call: arguments must be a JSON object: {e}", file=sys.stderr)
        return 2
    if not isinstance(arguments, dict):
        print("bench-vision call: arguments must be a JSON object, e.g. '{\"cam\": \"scope\"}'", file=sys.stderr)
        return 2
    bv = BenchVision(Path(args.root).resolve(), mock=args.mock,
                     config_path=Path(args.config).resolve() if args.config else None)

    async def go():
        async with Client(build_server(bv)) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            if args.tool not in names:
                return None, sorted(names)
            return await client.call_tool(args.tool, arguments), None

    result, names = asyncio.run(go())
    if result is None:
        print(f"bench-vision call: no tool {args.tool!r}; tools: {', '.join(names)}", file=sys.stderr)
        return 2
    text = "\n".join(c.text for c in result.content if c.type == "text")
    if result.is_error:
        print(f"bench-vision call: {text.removeprefix(f'Error executing tool {args.tool}: ')}", file=sys.stderr)
        return 1
    images = [c for c in result.content if c.type == "image"]
    if images and args.out:
        try:
            Path(args.out).write_bytes(base64.b64decode(images[0].data))
            text += f"\n(first returned image written to {args.out})"
        except OSError as e:
            print(text)
            print(f"bench-vision call: could not write --out {args.out}: {e.strerror or e}", file=sys.stderr)
            return 1
    print(text)
    return 0


def _setup(args: argparse.Namespace) -> int:
    from .errors import BenchVisionError
    from .setup import run_setup

    try:
        return run_setup(Path(args.root).resolve(), force=args.force)
    except BenchVisionError as e:
        print(f"bench-vision setup: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench-vision", description="MCP server for soldering-bench cameras.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the MCP server on stdio")
    serve.add_argument("--mock", action="store_true", help="serve images from ./mock/ instead of cameras")
    serve.add_argument("--root", default=".", help="project directory holding config.toml, mock/, captures/")
    serve.add_argument("--config", help="config file (default: <root>/config.toml)")
    serve.set_defaults(func=_serve)

    cap = sub.add_parser("capture", help="take one still through the MCP capture tool (camera check, no Claude)")
    cap.add_argument("cam", help="camera name from config.toml (or a mock image name)")
    cap.add_argument("--mock", action="store_true", help="use the images in ./mock/ instead of cameras")
    cap.add_argument("--rotate", type=int, default=0, help="extra clockwise rotation (multiple of 90)")
    cap.add_argument("--root", default=".", help="project directory holding config.toml, mock/, captures/")
    cap.add_argument("--config", help="config file (default: <root>/config.toml)")
    cap.add_argument("--out", help="also write the returned (downscaled) JPEG here")
    cap.set_defaults(func=_capture)

    disp = sub.add_parser("display", help="serve the wall-display page (open it in Chromium --kiosk)")
    disp.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    disp.add_argument("--port", type=_port, default=8765, help="port (default 8765)")
    disp.add_argument("--mock", action="store_true", help="live view from the images in ./mock/")
    disp.add_argument("--root", default=".", help="project directory (config.toml, boards/, mock/)")
    disp.add_argument("--config", help="config file (default: <root>/config.toml)")
    disp.set_defaults(func=_display)

    callp = sub.add_parser("call", help="run any MCP tool once without Claude, e.g. show_step")
    callp.add_argument("tool", help="tool name, e.g. show_step, capture, list_cameras")
    callp.add_argument("arguments", nargs="?", help="JSON object of arguments, e.g. '{\"cam\": \"scope\"}'")
    callp.add_argument("--mock", action="store_true", help="use the images in ./mock/ instead of cameras")
    callp.add_argument("--root", default=".", help="project directory holding config.toml, mock/, captures/")
    callp.add_argument("--config", help="config file (default: <root>/config.toml)")
    callp.add_argument("--out", help="write the first returned image here")
    callp.set_defaults(func=_call)

    setup = sub.add_parser("setup", help="list cameras, print a diagnostic report, write a starter config.toml")
    setup.add_argument("--root", default=".", help="directory to write config.toml into")
    setup.add_argument("--force", action="store_true", help="overwrite an existing config.toml")
    setup.set_defaults(func=_setup)

    args = parser.parse_args(argv)
    # stdout is the MCP channel; all logging goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="bench-vision %(levelname)s: %(message)s")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        # One line, no traceback, whatever was running (imports, a hung camera worker, ...).
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not start a traceback
        print(f"bench-vision {args.command}: interrupted", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(130)  # skip interpreter shutdown, which would wait on (and report) worker threads
