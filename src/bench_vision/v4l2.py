"""Thin wrapper around the `v4l2-ctl` command line tool."""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

from .errors import CameraError

# (argv) -> (returncode, stdout, stderr)
Runner = Callable[[list[str]], tuple[int, str, str]]

CTRL_LINE_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")
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


def subprocess_runner(argv: list[str]) -> tuple[int, str, str]:
    if shutil.which(argv[0]) is None:
        raise CameraError(
            f"`{argv[0]}` is not installed. Install it with `sudo apt install v4l-utils`."
        )
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        raise CameraError(f"`{' '.join(argv)}` timed out after 15 s; is the device hung?") from None
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
            flags=kv.get("flags", ""),
        )
    return controls


class V4L2:
    def __init__(self, runner: Runner = subprocess_runner):
        self.runner = runner

    def _run(self, argv: list[str], what: str) -> str:
        code, out, err = self.runner(argv)
        if code != 0:
            msg = (err or out).strip().splitlines()
            detail = msg[-1] if msg else f"exit code {code}"
            raise CameraError(f"{what} failed: {detail}")
        return out

    def list_devices(self) -> str:
        code, out, err = self.runner(["v4l2-ctl", "--list-devices"])
        # v4l2-ctl exits non-zero if *any* node can't be opened but still lists the rest.
        if not out.strip() and code != 0:
            raise CameraError(f"`v4l2-ctl --list-devices` failed: {(err or '').strip()}")
        return out

    def list_ctrls(self, device: str) -> dict[str, ControlInfo]:
        out = self._run(["v4l2-ctl", "-d", device, "--list-ctrls"], f"Listing controls of {device}")
        return parse_list_ctrls(out)

    def list_formats(self, device: str) -> str:
        return self._run(["v4l2-ctl", "-d", device, "--list-formats-ext"], f"Listing formats of {device}")

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
