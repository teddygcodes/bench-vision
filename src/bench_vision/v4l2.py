"""Thin wrapper around the `v4l2-ctl` command line tool."""

from __future__ import annotations

import difflib
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from .errors import CameraError

# (argv) -> (returncode, stdout, stderr)
Runner = Callable[[list[str]], tuple[int, str, str]]

CTRL_LINE_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")
FLAGS_RE = re.compile(r"flags=([\w\-, ]+?)\s*(?:\(|$)")
KV_RE = re.compile(r"(\w+)=(-?\w+)")


@dataclass(frozen=True)
class ControlInfo:
    name: str
    type: str  # int, bool, menu, intmenu, button, ...
    min: int | None = None
    max: int | None = None
    step: int | None = None
    default: int | None = None
    value: int | None = None
    flags: str = ""

    @property
    def inactive(self) -> bool:
        return "inactive" in self.flags

    def describe(self) -> str:
        if self.type == "bool":
            rng = "0/1"
        elif self.min is not None and self.max is not None:
            rng = f"{self.min}..{self.max}"
        else:
            rng = "?"
        extra = f", {self.flags}" if self.flags else ""
        return f"{self.name} ({self.type} {rng}, value={self.value}{extra})"


SUBPROCESS_TIMEOUT = 5.0


def require_v4l2() -> None:
    """Raise a one-line CameraError unless this is Linux with v4l2-ctl installed."""
    if not sys.platform.startswith("linux"):
        raise CameraError(
            f"This needs Linux/v4l2 (v4l2-ctl from v4l-utils); this machine is {platform.system() or sys.platform}. "
            "Run it on the bench PC (the MCP server itself can run here with `serve --mock`)."
        )
    if shutil.which("v4l2-ctl") is None:
        raise CameraError("This needs Linux/v4l2: `v4l2-ctl` is not installed. Install it with `sudo apt install v4l-utils`.")


def subprocess_runner(argv: list[str]) -> tuple[int, str, str]:
    require_v4l2()
    try:
        # errors="replace": uvcvideo truncates USB product names to 32 bytes, which can split a UTF-8 character
        proc = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=SUBPROCESS_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise CameraError(
            f"`{' '.join(argv)}` timed out after {SUBPROCESS_TIMEOUT:g} s; is the device hung?"
        ) from None
    except OSError as e:
        raise CameraError(f"Could not run `{argv[0]}`: {e.strerror or e}.") from None
    return proc.returncode, proc.stdout, proc.stderr


def parse_list_ctrls(text: str) -> dict[str, ControlInfo]:
    """Parse `v4l2-ctl --list-ctrls` output into {name: ControlInfo}."""
    controls: dict[str, ControlInfo] = {}
    for line in text.splitlines():
        m = CTRL_LINE_RE.match(line)
        if not m:
            continue
        name, typ, rest = m.groups()
        kv: dict[str, str] = {}
        for k, v in KV_RE.findall(rest):
            kv[k] = v

        def num(key: str) -> int | None:
            try:
                return int(kv[key])
            except (KeyError, ValueError):
                return None

        controls[name] = ControlInfo(
            name=name,
            type=typ,
            min=num("min"),
            max=num("max"),
            step=num("step"),
            default=num("default"),
            value=num("value"),
            flags=(m2.group(1).strip() if (m2 := FLAGS_RE.search(rest)) else ""),  # e.g. "update, inactive"
        )
    return controls


class V4L2:
    def __init__(self, runner: Runner = subprocess_runner):
        self.runner = runner

    def require(self) -> None:
        """Platform check for the real runner; injected (test) runners skip it."""
        if self.runner is subprocess_runner:
            require_v4l2()

    def _run(self, argv: list[str], what: str) -> str:
        code, out, err = self.runner(argv)
        if code != 0:
            msg = (err or out).strip().splitlines()
            detail = msg[-1] if msg else f"exit code {code}"
            raise CameraError(f"{what} failed: {detail}")
        return out

    def list_devices(self) -> str:
        code, out, err = self.runner(["v4l2-ctl", "--list-devices"])
        # v4l2-ctl exits non-zero if any node can't be opened (or none exist) but still lists the rest.
        if not out.strip() and code != 0:
            return f"(v4l2-ctl --list-devices found nothing: {(err or '').strip() or f'exit code {code}'})"
        return out

    def list_ctrls(self, device: str) -> dict[str, ControlInfo]:
        out = self._run(["v4l2-ctl", "-d", device, "--list-ctrls"], f"Listing controls of {device}")
        return parse_list_ctrls(out)

    def list_formats(self, device: str) -> str:
        return self._run(["v4l2-ctl", "-d", device, "--list-formats-ext"], f"Listing formats of {device}")

    def list_ctrls_menus(self, device: str) -> str:
        """Raw `--list-ctrls-menus` text (controls plus the names of menu entries)."""
        return self._run(["v4l2-ctl", "-d", device, "--list-ctrls-menus"], f"Listing controls of {device}")

    def version(self) -> str:
        code, out, err = self.runner(["v4l2-ctl", "--version"])
        return (out or err).strip().splitlines()[0] if (out or err).strip() else "unknown"

    def set_ctrl(self, device: str, name: str, value: int) -> None:
        self._run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={name}={value}"],
            f"Setting {name}={value} on {device}",
        )

    def get_ctrl(self, device: str, name: str) -> int | None:
        out = self._run(["v4l2-ctl", "-d", device, f"--get-ctrl={name}"], f"Reading {name} on {device}")
        m = re.search(r":\s*(-?\d+)", out)
        return int(m.group(1)) if m else None


def validate_control(
    controls: dict[str, ControlInfo], name: str, value: int, where: str
) -> ControlInfo:
    """Raise a clear CameraError if `name=value` is not valid for this device."""
    info = controls.get(name)
    if info is None:
        close = difflib.get_close_matches(name, list(controls), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise CameraError(
            f"{where}: control '{name}' does not exist on this camera.{hint} "
            f"Available: {', '.join(sorted(controls)) or '(none)'}."
        )
    if info.type == "bool" and value not in (0, 1):
        raise CameraError(f"{where}: {name} is a bool control; use 0 or 1, not {value}.")
    if info.min is not None and info.max is not None and not info.min <= value <= info.max:
        raise CameraError(f"{where}: {name}={value} is out of range {info.min}..{info.max}.")
    return info


FMT_RE = re.compile(r"^\s*\[\d+\]:\s*'(.{4})'\s*\(([^)]*)\)")
SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s+(\d+)x(\d+)")
STEPWISE_RE = re.compile(r"^\s*Size:\s*(?:Stepwise|Continuous)\s+\d+x\d+\s*-\s*(\d+)x(\d+)")
FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")


@dataclass
class PixelFormat:
    fourcc: str
    description: str
    sizes: dict[tuple[int, int], list[float]]  # (w, h) -> frame rates


def parse_formats(text: str) -> list[PixelFormat]:
    """Parse `v4l2-ctl --list-formats-ext` output."""
    formats: list[PixelFormat] = []
    size: tuple[int, int] | None = None
    for line in text.splitlines():
        if m := FMT_RE.match(line):
            formats.append(PixelFormat(m.group(1), m.group(2), {}))
            size = None
        elif formats and ((m := SIZE_RE.match(line)) or (m := STEPWISE_RE.match(line))):
            size = (int(m.group(1)), int(m.group(2)))
            formats[-1].sizes.setdefault(size, [])
        elif formats and size and (m := FPS_RE.search(line)):
            formats[-1].sizes[size].append(float(m.group(1)))
    return formats
