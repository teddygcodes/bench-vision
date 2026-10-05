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
