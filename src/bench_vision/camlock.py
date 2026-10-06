"""The cross-process "one camera open at a time" lock.

Captures (in the MCP server) and the live view (in `bench-vision display`) run in different
processes, so the in-process lock in CameraManager is backed by an flock on
`.bench-vision/camera.lock` under the project root. Captures have priority: before waiting
for the lock, a capture drops a `camera.want.*` marker; the live stream checks for markers
and stops its reader process (killing it if needed) before releasing (see live.py), and a capture
never waits more than CAPTURE_WAIT seconds in any case. Each capture updates `capture.<cam>.json`
before it lets go (the time of the last capture that worked and the failed opens since then), so
the live view can reset or extend its retry back-off (see live.py).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator

from .errors import CameraError, CameraMissingError, CameraOpenError

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX: lock is in-process only
    fcntl = None  # type: ignore[assignment]

CAPTURE_WAIT = 1.5  # seconds a capture waits for the live view to let go (it should take < 0.5 s)
MAX_FAILS = 50  # failed opens kept per camera in capture.<cam>.json
MARKER_STALE = 30.0  # want-markers older than this are leftovers from a crashed process


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


class CameraLock:
    def __init__(self, state_dir: Path, mock: bool = False):
        self.dir = state_dir
        self.mock = mock  # --mock captures keep their own outcome files, away from the real cameras
        self.path = state_dir / "camera.lock"
        self._fd: int | None = None
        self._guard = threading.Lock()  # serialises this process's use of the fd

    # -------------------------------------------------------------- internals

    def _open(self) -> int | None:
        if fcntl is None:
            return None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            return os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return None  # unusable state dir: fall back to the in-process lock only

    def _try_lock(self, fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        except OSError:
            return True  # can't lock this filesystem: behave as if we hold it

    def _markers(self) -> list[Path]:
        try:
            return list(self.dir.glob("camera.want.*"))
        except OSError:
            return []

    # ---------------------------------------------------------- capture side

    @contextlib.contextmanager
    def for_capture(self, cam: str) -> Iterator[None]:
        """Hold the camera lock for one capture, asking the live view to yield first."""
        marker = self.dir / f"camera.want.{os.getpid()}.{threading.get_ident()}"
        fd = self._open()
        if fd is None:
            yield
            return
        try:
            with contextlib.suppress(OSError):
                marker.write_text(f"{cam} {time.time():.3f}\n")
            deadline = time.monotonic() + CAPTURE_WAIT
            while not self._try_lock(fd):
                if time.monotonic() >= deadline:
                    raise CameraError(
                        f"Camera '{cam}': another bench-vision process (the live view or a second server) kept "
                        f"the cameras for more than {CAPTURE_WAIT:g} s. Try again; if it repeats, restart "
                        "`bench-vision display`."
                    )
                time.sleep(0.02)
            result, missing = "failed", False
            try:
                yield
                result = "ok"
            except CameraOpenError as e:
                result, missing = "open_failed", isinstance(e, CameraMissingError)
                raise
            finally:
                self._record(cam, result, missing)  # before unlocking: the live view reads it when it resumes
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            with contextlib.suppress(OSError):
                marker.unlink()
            os.close(fd)

    def outcome_path(self, cam: str) -> Path:
        return self.dir / f"capture{'-mock' if self.mock else ''}.{cam}.json"

    def _load_outcomes(self, cam: str) -> tuple[float, list[float], bool]:
        """(time of the last capture that worked, times of failed opens since then, last failure was missing)."""
        try:
            data = json.loads(self.outcome_path(cam).read_text(encoding="utf-8"))
            limit = time.time() + 60  # ignore times from a clock far ahead (or a hand-edited file)
            ok = float(data.get("ok_time") or 0.0)
            fails = [float(t) for t in data.get("open_failed_times") or []]
            if not (0 <= ok <= limit) or not all(0 <= t <= limit for t in fails):
                return 0.0, [], False
            return ok, fails, data.get("missing") is True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
            return 0.0, [], False

    def _record(self, cam: str, result: str, missing: bool = False) -> None:
        # Called while this capture still holds the camera lock, so updates from captures never interleave.
        if result == "failed":
            return  # opened but failed later: says nothing about whether the live view can open it
        ok, fails, was_missing = self._load_outcomes(cam)
        now = time.time()
        if result == "ok":
            ok, fails = now, []
        else:
            fails, was_missing = (fails + [now])[-MAX_FAILS:], missing
        path = self.outcome_path(cam)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps({"cam": cam, "ok_time": ok, "open_failed_times": fails,
                                       "missing": was_missing}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()

    def outcomes_since(self, cam: str, since: float) -> tuple[bool, list[float], bool, float]:
        """Captures of `cam` that ended after `since` (wall-clock): (one worked, times of failed opens after
        both `since` and that success, the last failure was a missing device, the latest of these times)."""
        ok, fails, missing = self._load_outcomes(cam)
        reset = ok > since
        fails = [t for t in fails if t > max(since, ok)]
        return reset, fails, missing, max([since, ok, *fails])

    # ----------------------------------------------------------- stream side

    def wanted(self) -> bool:
        """True while some capture is waiting for (or using) the cameras."""
        now = time.time()
        for m in self._markers():
            try:
                pid = int(m.name.split(".")[2])
                alive = pid == os.getpid() or _pid_alive(pid)
                if alive and now - m.stat().st_mtime < MARKER_STALE:
                    return True
                m.unlink()  # left behind by a crashed (or long-gone) process
            except (OSError, ValueError, IndexError):
                continue
        return False

    def try_acquire_low(self) -> bool:
        """Low-priority acquire for the live stream: only when no capture wants the cameras."""
        with self._guard:
            if self._fd is not None:
                return True
            if self.wanted():
                return False
            fd = self._open()
            if fd is None:
                self._fd = -1  # no file lock available: stream runs under the in-process rules only
                return True
            if not self._try_lock(fd):
                os.close(fd)
                return False
            if self.wanted():  # a capture arrived while we were locking: give way at once
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False
            self._fd = fd
            return True

    def release(self) -> None:
        """Release the stream's hold (safe to call from a watchdog while the stream is stuck)."""
        with self._guard:
            fd, self._fd = self._fd, None
        if fd is not None and fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)

    @property
    def held(self) -> bool:
        return self._fd is not None
