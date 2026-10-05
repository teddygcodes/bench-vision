"""Boards: a reference image with joint rectangles, joint states, and a verdict log.

Layout under <root>/boards/:
  <name>.json            {"name", "image", "size": [w, h], "joints": [{id, x, y, w, h, state}], ...}
  <name>.jpg             the reference image (full resolution)
  <name>/verdicts.jsonl  one JSON object per record_verdict
  <name>/v<NNNN>.jpg     verdict thumbnails
  .current.json          {"name": ...}: the board the display shows (a name no board can have)
Written by the MCP tools, read by the display (which reloads it on start), all under a file lock.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .errors import BenchVisionError
from .locks import file_lock

STATES = ("todo", "active", "verified", "flagged")
VERDICTS = ("good", "cold", "insufficient", "bridge", "lifted pad", "unsoldered", "solder ball", "unsure")
NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")  # no dots: board files can't collide with each other
CURRENT = ".current.json"
JOINT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,23}")
MAX_JOINTS = 500
LOCK_TIMEOUT = 5.0


def check_name(name: Any) -> str:
    if isinstance(name, str) and NAME_RE.fullmatch(name.lower()):
        return name.lower()
    raise BenchVisionError(
        "Board name must be 1-64 characters of letters, digits, '_' or '-' (e.g. 'mock-board'); "
        f"got {repr(name)[:40]}."
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o644)
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class BoardStore:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def _locked(self):
        deadline = time.monotonic() + LOCK_TIMEOUT
        if not self._lock.acquire(timeout=LOCK_TIMEOUT):
            raise BenchVisionError("Boards are busy (another board update is running); try again.")
        try:
            with file_lock(self.root / ".lock", deadline) as ok:
                if not ok:
                    raise BenchVisionError("Boards are busy in another bench-vision process; try again.")
                yield
        finally:
            self._lock.release()

    def _json(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def image_path(self, name: str) -> Path:
        return self.root / f"{name}.jpg"

    def _dir(self, name: str) -> Path:
        return self.root / name

    # ------------------------------------------------------------- reading

    def current_name(self) -> str | None:
        try:
            data = json.loads((self.root / CURRENT).read_text(encoding="utf-8"))
            return check_name(data.get("name")) if isinstance(data, dict) else None
        except (OSError, ValueError, BenchVisionError, RecursionError):
            return None

    def load(self, name: str) -> dict[str, Any]:
        try:
            data = json.loads(self._json(name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise BenchVisionError(f"No board named '{name}'. Create it with board_init first.") from None
        except (OSError, ValueError, RecursionError) as e:
            raise BenchVisionError(f"Board file {self._json(name)} is unreadable or corrupt ({type(e).__name__}).") from None
        if not isinstance(data, dict) or not isinstance(data.get("joints"), list):
            raise BenchVisionError(f"Board file {self._json(name)} is corrupt; run board_init again.")
        return data

    def verdicts(self, name: str, last: int = 8) -> list[dict[str, Any]]:
        try:
            lines = (self._dir(name) / "verdicts.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        out = []
        for line in lines[-last * 4:]:  # tolerate a few corrupt lines
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if isinstance(v, dict) and isinstance(v.get("joint"), str) and isinstance(v.get("thumb"), str):
                out.append(v)
        return out[-last:]

    def thumb_path(self, name: str, thumb: str) -> Path | None:
        if not re.fullmatch(r"v\d{4,6}\.jpg", thumb):
            return None
        p = self._dir(name) / thumb
        return p if p.is_file() else None

    # ------------------------------------------------------------- writing

    def init(self, name: str, image_jpeg: bytes, size: tuple[int, int], joints: list[dict[str, Any]],
             source: str, camera: str | None = None) -> dict[str, Any]:
        board = {
            "name": name, "image": f"{name}.jpg", "size": list(size), "source": source, "camera": camera,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "joints": [{**j, "state": "todo"} for j in joints],
        }
        with self._locked():
            _atomic_write(self.image_path(name), image_jpeg)
            _atomic_write(self._json(name), (json.dumps(board, indent=2) + "\n").encode())
            # a re-initialised board starts a fresh verdict log
            log = self._dir(name) / "verdicts.jsonl"
            if log.exists():
                stamp = time.strftime("%Y%m%d-%H%M%S")
                with contextlib.suppress(OSError):
                    log.rename(log.with_name(f"verdicts.{stamp}.jsonl"))
            _atomic_write(self.root / CURRENT, (json.dumps({"name": name}) + "\n").encode())
        return board

    def set_state(self, name: str, joint_id: str, state: str) -> tuple[dict[str, Any], str | None]:
        """Returns (board, id of a joint that was active before and got demoted to todo)."""
        with self._locked():
            board = self.load(name)
            joint = next((j for j in board["joints"] if isinstance(j, dict) and j.get("id") == joint_id), None)
            if joint is None:
                ids = [j.get("id") for j in board["joints"] if isinstance(j, dict)]
                shown = ", ".join(map(str, ids[:30])) + (" ..." if len(ids) > 30 else "")
                raise BenchVisionError(f"Board '{name}' has no joint '{joint_id[:30]}'. Joints: {shown}.")
            demoted = None
            if state == "active":
                for j in board["joints"]:
                    if isinstance(j, dict) and j is not joint and j.get("state") == "active":
                        j["state"] = "todo"
                        demoted = j.get("id")
            joint["state"] = state
            _atomic_write(self._json(name), (json.dumps(board, indent=2) + "\n").encode())
            _atomic_write(self.root / CURRENT, (json.dumps({"name": name}) + "\n").encode())
            return board, demoted

    def add_verdict(self, name: str, entry: dict[str, Any], thumb_jpeg: bytes) -> dict[str, Any]:
        with self._locked():
            d = self._dir(name)
            d.mkdir(parents=True, exist_ok=True)
            n = 1 + max((int(p.stem[1:]) for p in d.glob("v*.jpg") if p.stem[1:].isdigit()), default=0)
            thumb = f"v{n:04d}.jpg"
            _atomic_write(d / thumb, thumb_jpeg)
            entry = {**entry, "thumb": thumb}
            with open(d / "verdicts.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            return entry
