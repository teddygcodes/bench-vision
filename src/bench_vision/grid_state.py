"""The last grid drawn on each camera.

Each server process answers capture_cell from the grids *it* drew, since those
are the overlays its model saw. The grids are also kept in
.bench-vision/grids[-mock].json so they survive a server restart: loaded once at
startup, and updated one camera at a time (merged under an flock, so another
process's grids for other cameras are kept). A state file that exists but
can't be read is never overwritten.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import imaging
from .errors import BenchVisionError
from .locks import file_lock

log = logging.getLogger(__name__)

LOCK_TIMEOUT = 2.0  # seconds to wait for another process's state-file update
DRAWN_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


class Unreadable(Exception):
    """The state file exists but could not be read (permissions, not a regular file)."""


def _valid_size(size: Any) -> bool:
    return (
        isinstance(size, list)
        and len(size) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 1_000_000 for v in size)
    )


def _clean(entry: Any) -> dict[str, Any] | None:
    """Validate one entry from the file; None if it is unusable."""
    try:
        imaging.check_grid(entry["rows"], entry["cols"])
        rotation = imaging.normalize_rotation(entry["rotation"])
    except (BenchVisionError, KeyError, TypeError, AttributeError):
        return None
    drawn, size = entry.get("drawn_at"), entry.get("full_res_size")
    return {
        "rows": entry["rows"],
        "cols": entry["cols"],
        "rotation": rotation,
        "full_res_size": size if _valid_size(size) else None,
        "drawn_at": drawn if isinstance(drawn, str) and DRAWN_AT_RE.fullmatch(drawn) else None,
    }


class GridStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()  # guards memory only; never held while waiting on files
        self._write_lock = threading.Lock()  # serialises this process's file updates
        self.memory: dict[str, dict[str, Any]] = {}
        self.read_error: str | None = None  # why the file couldn't be read at startup, if it couldn't
        try:
            self.memory = self._read_file()
        except Unreadable as e:
            self.read_error = str(e)
            log.warning("cannot read %s: %s", self.path, e)
        except Exception:  # noqa: BLE001 - grid state must never stop the server
            log.exception("ignoring grid state in %s", self.path)
        else:
            if self._corrupt:
                self.read_error = "corrupt file"

    def _read_file(self) -> dict[str, dict[str, Any]]:
        """Valid entries from the file ({} if missing or corrupt); raises Unreadable."""
        self._corrupt = False
        try:
            if not self.path.exists():
                return {}
            if not self.path.is_file():  # refuses directories, FIFOs and devices
                raise Unreadable("not a regular file")
            text = self.path.read_text(encoding="utf-8")
        except OSError as e:
            raise Unreadable(e.strerror or type(e).__name__) from None
        except UnicodeDecodeError:
            log.warning("ignoring corrupt %s", self.path)
            self._corrupt = True
            return {}
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            log.warning("ignoring corrupt %s", self.path)
            self._corrupt = True
            return {}
        out = {}
        for cam, entry in data.items() if isinstance(data, dict) else []:
            cleaned = _clean(entry)
            if cleaned is None:
                log.warning("ignoring invalid grid for %r in %s", cam, self.path)
            else:
                out[cam] = cleaned
        return out

    def get(self, cam: str) -> dict[str, Any] | None:
        with self._lock:
            return self.memory.get(cam)

    def put(self, cam: str, grid: dict[str, Any]) -> str:
        """Store a grid; returns "" or a warning if it could only be kept in memory."""
        with self._lock:
            self.memory[cam] = grid  # capture_cell sees it immediately, whatever happens to the file
        deadline = time.monotonic() + LOCK_TIMEOUT  # one bound for both locks
        if not self._write_lock.acquire(timeout=LOCK_TIMEOUT):
            return " WARNING: grid kept in memory only (state file busy)."
        try:
            return self._write(cam, deadline)
        finally:
            self._write_lock.release()

    def _write(self, cam: str, deadline: float) -> str:
        with file_lock(self.path.with_suffix(".lock"), deadline) as locked:
            if not locked:
                log.error("timed out waiting for %s", self.path.with_suffix(".lock"))
                return " WARNING: grid kept in memory only (state file busy in another process)."
            try:
                on_disk = self._read_file()
            except Unreadable as e:
                log.error("not overwriting unreadable %s: %s", self.path, e)
                return f" WARNING: grid kept in memory only (state file unreadable: {e})."
            except Exception:  # noqa: BLE001
                log.exception("could not read %s", self.path)
                return " WARNING: grid kept in memory only (state file unreadable)."
            with self._lock:  # write this process's latest grid for cam, even if a newer put raced us
                latest = self.memory[cam]
            snapshot = json.dumps({**on_disk, cam: latest}, indent=2) + "\n"
            tmp: Path | None = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.stem}.", suffix=".tmp")
                tmp = Path(name)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(snapshot)
                tmp.replace(self.path)
                self.read_error = None  # the file is good again
                return ""
            except OSError as e:
                if tmp is not None:
                    with contextlib.suppress(OSError):
                        tmp.unlink()
                log.error("could not persist grid: %s", e)
                return f" WARNING: grid kept in memory only ({e.strerror or e})."
