"""`bench-vision` command line entry point."""

from __future__ import annotations

import argparse
import logging
import os
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench-vision", description="MCP server for soldering-bench cameras.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the MCP server on stdio")
    serve.add_argument("--mock", action="store_true", help="serve images from ./mock/ instead of cameras")
    serve.add_argument("--root", default=".", help="project directory holding config.toml, mock/, captures/")
    serve.add_argument("--config", help="config file (default: <root>/config.toml)")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    # stdout is the MCP channel; all logging goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="bench-vision %(levelname)s: %(message)s")
    return args.func(args)
