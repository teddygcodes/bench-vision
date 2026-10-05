"""bench-vision: an MCP server that gives Claude eyes on a soldering bench."""

import sys


def main() -> None:
    from .cli import main as cli_main

    sys.exit(cli_main())
