"""Named reference frames for compare(): references/<cam>/<name>.png + .json.

Frames are stored losslessly (PNG) at full resolution, already rotated, so a
later compare diffs pixels rather than JPEG artefacts. A PNG and its sidecar
are written and read as a pair under a thread lock plus a bounded flock, so
neither threads nor several server processes can see a half-replaced pair.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import load_image_strict
from .errors import BenchVisionError
from .locks import file_lock

NAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
LOCK_TIMEOUT = 5.0
MAX_LISTED = 20


def check_name(name: Any) -> str:
    """Validate a reference name; names are case-insensitive and stored lowercase."""
    if isinstance(name, str) and NAME_RE.fullmatch(name.lower()) and ".." not in name:
        return name.lower()
    shown = repr(name) if len(repr(name)) <= 40 else repr(name)[:37] + "..."
    raise BenchVisionError(
        f"Reference name must be 1-64 characters of letters, digits, '_', '-' or '.' (no '..'), "
        f"starting with a letter or digit (e.g. 'u1-before'); got {shown}."
    )


def _pair_hash(png_bytes: bytes, meta: dict[str, Any]) -> str:
    """sha256 over the image bytes plus the metadata compare relies on (rotation, size)."""
    h = hashlib.sha256(png_bytes)
    h.update(json.dumps([meta.get("rotation"), meta.get("full_res_size")]).encode())
    return h.hexdigest()


def _short(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ReferenceStore:
    def __init__(self, root: Path):
        self.root = root
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _paths(self, cam: str, name: str) -> tuple[Path, Path]:
        d = self.root / cam
        return d / f"{name}.png", d / f"{name}.json"

    @contextlib.contextmanager
    def _pair_lock(self, cam: str):
        deadline = time.monotonic() + LOCK_TIMEOUT  # one bound covering both locks
        busy = BenchVisionError("References are busy (another save_reference/compare is running); try again.")
        with self._locks_guard:
            lock = self._locks.setdefault(cam, threading.Lock())  # cameras never wait on each other
        if not lock.acquire(timeout=LOCK_TIMEOUT):
            raise busy
        try:
            with file_lock(self.root / cam / ".lock", deadline) as locked:
                if not locked:
                    raise busy
                yield
        finally:
            lock.release()

    def path_of(self, cam: str, name: str) -> Path:
        return self._paths(cam, name)[0]

    def names(self, cam: str) -> list[str]:
        try:
            return sorted(
                p.stem
                for p in (self.root / cam).glob("*.png")
                if NAME_RE.fullmatch(p.stem) and p.is_file() and p.with_suffix(".json").is_file()
            )
        except OSError:
            return []

    def _inaccessible(self, cam: str, e: OSError) -> BenchVisionError:
        return BenchVisionError(f"Reference directory {self.root / cam} is not accessible: {e.strerror or e}.")

    def save(self, cam: str, name: str, image: np.ndarray, meta: dict[str, Any]) -> tuple[Path, dict | None]:
        """Write atomically; returns (png path, metadata of the reference it replaced or None)."""
        png, js = self._paths(cam, name)
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise BenchVisionError(f"Could not encode reference '{name}' as PNG.")
        png_bytes = buf.tobytes()
        meta = {**meta, "pair_sha256": _pair_hash(png_bytes, meta)}  # ties image, rotation and size together
        payload = {png: png_bytes, js: (json.dumps(meta, indent=2) + "\n").encode()}
        try:
            png.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            raise BenchVisionError(f"Could not save reference '{name}': {png.parent} exists but is not a directory.") from None
        except OSError as e:
            raise BenchVisionError(f"Could not save reference '{name}' under {self.root}: {e.strerror or e}.") from None
        with self._pair_lock(cam):
            try:
                # {} = a reference existed but its sidecar couldn't be read; None = nothing there
                previous = (self.load_meta(cam, name) or {}) if png.exists() else None
            except OSError as e:
                raise self._inaccessible(cam, e) from None
            self._sweep_stale_temps(png.parent)
            temps: dict[Path, str] = {}
            try:
                # Write both temp files first, then swap them in back to back.
                for path, data in payload.items():
                    fd, tmp = tempfile.mkstemp(dir=png.parent, prefix=f".{name}.", suffix=".tmp")
                    temps[path] = tmp
                    with os.fdopen(fd, "wb") as f:
                        os.fchmod(f.fileno(), 0o644)  # mkstemp's 0600 would differ from captures/
                        f.write(data)
                for path, tmp in temps.items():
                    os.replace(tmp, path)
            except OSError as e:
                raise BenchVisionError(
                    f"Could not save reference '{name}' under {self.root}: {e.strerror or e}."
                ) from None
            finally:
                for tmp in temps.values():
                    with contextlib.suppress(OSError):
                        os.unlink(tmp)
        return png, previous

    @staticmethod
    def _sweep_stale_temps(d: Path, max_age: float = 3600.0) -> None:
        """Remove temp files left by a save that was killed mid-write (older than an hour)."""
        cutoff = time.time() - max_age
        with contextlib.suppress(OSError):
            for p in d.glob(".*.tmp"):
                with contextlib.suppress(OSError):
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()

    def load_meta(self, cam: str, name: str, strict: bool = False) -> dict[str, Any] | None:
        _, js = self._paths(cam, name)
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except PermissionError as e:
            if strict:
                raise BenchVisionError(f"Reference sidecar {js} is not readable: {e.strerror}.") from None
            return None
        except (OSError, ValueError, RecursionError):
            return None
        return meta if isinstance(meta, dict) else None

    def load(self, cam: str, name: str) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            if not (self.root / cam).is_dir():  # nothing saved yet: don't create dirs/lock files
                raise BenchVisionError(
                    f"No reference named '{name}' for camera '{cam}'. No references saved for '{cam}' yet. "
                    "Use save_reference(cam, name) first."
                )
            with self._pair_lock(cam):
                return self._load(cam, name)
        except OSError as e:
            raise self._inaccessible(cam, e) from None

    def _load(self, cam: str, name: str) -> tuple[np.ndarray, dict[str, Any]]:
        png, _ = self._paths(cam, name)
        if not png.is_file():
            have = self.names(cam)
            if not have:
                listing = f"No references saved for '{cam}' yet."
            else:
                more = f" and {len(have) - MAX_LISTED} more" if len(have) > MAX_LISTED else ""
                listing = f"Saved references for '{cam}': {', '.join(have[:MAX_LISTED])}{more}."
            raise BenchVisionError(
                f"No reference named '{name}' for camera '{cam}'. {listing} Use save_reference(cam, name) first."
            )
        meta = self.load_meta(cam, name, strict=True)
        if meta is None:
            raise BenchVisionError(
                f"Reference '{name}' for '{cam}' is missing or has a corrupt .json sidecar; save it again."
            )
        expected = meta.get("pair_sha256")
        if not isinstance(expected, str):
            raise BenchVisionError(
                f"Reference '{name}' for '{cam}' has no image checksum in its sidecar; save the reference again."
            )
        try:
            actual = _pair_hash(png.read_bytes(), meta)
        except OSError as e:
            raise BenchVisionError(f"Reference image {png} is not readable: {e.strerror or e}.") from None
        if actual != expected:
            raise BenchVisionError(
                f"Reference '{name}' for '{cam}': image and sidecar don't match (interrupted save or edited "
                "sidecar?); save the reference again."
            )
        img = load_image_strict(png)
        if img is None:
            raise BenchVisionError(f"Reference image {png} is corrupt; save the reference again.")
        size = meta.get("full_res_size")
        if size != [img.shape[1], img.shape[0]]:
            raise BenchVisionError(
                f"Reference '{name}' for '{cam}': image and sidecar disagree (sidecar full_res_size "
                f"{_short(size)}, image {img.shape[1]}x{img.shape[0]}); save the reference again."
            )
        return img, meta
