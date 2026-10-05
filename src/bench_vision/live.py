"""Live MJPEG view for the wall display (runs inside `bench-vision display`).

The camera is read by a child process (livereader.py) that streams JPEG frames back. While at
least one page watches /live.mjpg, this class holds the shared camera lock (camlock.CameraLock)
as a low-priority user and keeps a reader running. When a capture wants the cameras (or the last
viewer leaves), it terminates the reader, kills it if it hasn't exited after STOP_GRACE, and only
releases the lock once the process is gone: the kernel has closed the device by then, so a
capture never runs while the live view still has a camera open, and never waits on a stuck
reader for more than ~STOP_GRACE + KILL_WAIT (well inside camlock.CAPTURE_WAIT).
A missing or unplugged camera is retried inside the reader with back-off (3 s up to 30 s; its
`status:` stderr lines set the status), so no new process is started per attempt.
Frames go only to the local page, never to the model.
"""

from __future__ import annotations

import logging
import struct
import subprocess
import threading
import time
from typing import Any

from .camlock import CameraLock
from .livereader import STATUS, retry_delay

log = logging.getLogger(__name__)

POLL = 0.02  # how often the lock-wanted flag and the reader are checked
STOP_GRACE = 0.3  # seconds after SIGTERM before SIGKILL
KILL_WAIT = 0.9  # seconds to wait for a killed reader to be reaped (a reader stuck in the kernel may linger)
IDLE_STOP = 2.0  # stop the camera this long after the last viewer leaves
MAX_VIEWERS = 4  # more MJPEG connections than this: the oldest is dropped (browsers can leak them)


class LiveStream:
    def __init__(self, command: list[str] | None, lock: CameraLock, unavailable: str | None = None):
        self.command = command
        self.lock = lock
        self._cond = threading.Condition()
        self.frame: bytes | None = None
        self.seq = 0
        self.status = unavailable or "starting"
        self._viewers: list[int] = []  # tokens, oldest first
        self._kicked: set[int] = set()
        self._next_token = 0
        self._last_viewer = 0.0
        self._thread: threading.Thread | None = None
        self._attempts = 0  # failed opens reported by readers since the last frame (carried across readers)
        self._fixed_unavailable = unavailable is not None or command is None
        self.on_status: Any = None  # set by the display server to refresh pages when the status changes

    # -------------------------------------------------------------- viewers

    def attach(self) -> int:
        with self._cond:
            self._next_token += 1
            token = self._next_token
            self._viewers.append(token)
            while len(self._viewers) > MAX_VIEWERS:
                self._kicked.add(self._viewers.pop(0))
                self._cond.notify_all()
            if not self._fixed_unavailable and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="live-stream", daemon=True)
                self._thread.start()
            return token

    def detach(self, token: int) -> None:
        with self._cond:
            if token in self._viewers:
                self._viewers.remove(token)
            self._kicked.discard(token)
            self._last_viewer = time.monotonic()

    def kicked(self, token: int) -> bool:
        with self._cond:
            return token in self._kicked

    @property
    def viewers(self) -> int:
        with self._cond:
            return len(self._viewers)

    def _watched(self) -> bool:
        with self._cond:
            return bool(self._viewers) or time.monotonic() - self._last_viewer < IDLE_STOP

    def wait_frame(self, seen: int, timeout: float) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self.seq != seen, timeout=timeout)
            return self.seq, self.frame

    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            log.info("live view: %s", status)
            if self.on_status is not None:
                self.on_status()

    def _publish(self, jpeg: bytes) -> None:
        with self._cond:
            self.frame = jpeg
            self.seq += 1
            self._cond.notify_all()

    # ------------------------------------------------------------- producer

    def _read_frames(self, proc: subprocess.Popen) -> None:
        out = proc.stdout
        try:
            while True:
                head = out.read(4)
                if len(head) < 4:
                    return
                (n,) = struct.unpack(">I", head)
                if n > 20 * 1024 * 1024:
                    return
                data = out.read(n)
                if len(data) < n:
                    return
                self._publish(data)
                self._attempts = 0
                self._set_status("live")
        except (OSError, ValueError):
            return

    def _drain_stderr(self, proc: subprocess.Popen, last: list[str]) -> None:
        """Read the reader's stderr as it comes (libjpeg/V4L2 warnings would otherwise fill the pipe and
        stall it). `status: ...` lines set the status at once (the reader retries a missing camera
        itself); the last other line explains an exit."""
        try:
            for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").strip()[:300]
                if line.startswith(STATUS):
                    status = line[len(STATUS):]
                    if status.startswith("unavailable"):
                        self._attempts += 1
                    self._set_status(status)
                elif line:
                    last[0] = line
                    log.debug("live reader: %s", line)
        except (OSError, ValueError):
            return

    def _stop(self, proc: subprocess.Popen) -> None:
        """Terminate the reader and wait until it is really gone (so the device is closed)."""
        if proc.poll() is None:
            try:
                proc.stdin.close()
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(STOP_GRACE)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(KILL_WAIT)
                except subprocess.TimeoutExpired:
                    log.error("live reader pid %s did not exit after SIGKILL (stuck in the kernel?)", proc.pid)

    def _run(self) -> None:
        assert self.command is not None
        failures = 0  # reader exits in a row without a frame: back off like the reader does
        while self._watched():
            if not self.lock.try_acquire_low():
                self._set_status("paused for a capture")
                time.sleep(0.05)
                continue
            proc = None
            error = None
            seq0 = self.seq
            try:
                proc = subprocess.Popen(self.command + ["--attempt", str(self._attempts)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                threading.Thread(target=self._read_frames, args=(proc,), daemon=True).start()
                last_err = [""]
                drain = threading.Thread(target=self._drain_stderr, args=(proc, last_err), daemon=True)
                drain.start()
                if self.status != "live" and not self.status.startswith("unavailable"):
                    self._set_status("starting")  # (while backing off, keep saying why until frames come)
                while self._watched() and not self.lock.wanted() and proc.poll() is None:
                    time.sleep(POLL)
                if proc.poll() is not None and not self.lock.wanted():
                    drain.join(1.0)  # pick up the reader's last line
                    error = last_err[0] if proc.returncode != 0 and last_err[0] else (
                        f"camera reader exited ({proc.returncode})")
            except OSError as e:
                error = f"could not start the camera reader: {e.strerror or e}"
            finally:
                if proc is not None:
                    self._stop(proc)
                self.lock.release()  # only now: the reader (and its open device) is gone
            if self.seq != seq0:
                failures = 0
            if error:
                delay = retry_delay(failures)
                failures += 1
                self._set_status(f"unavailable: {error} (retrying in {delay:g} s)")
                end = time.monotonic() + delay
                while time.monotonic() < end and self._watched():
                    time.sleep(0.1)
        self._set_status("stopped (no viewers)")
