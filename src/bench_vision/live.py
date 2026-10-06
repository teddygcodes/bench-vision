"""Live MJPEG view for the wall display (runs inside `bench-vision display`).

The camera is read by a child process (livereader.py) that streams JPEG frames back. While at
least one page watches /live.mjpg, this class holds the shared camera lock (camlock.CameraLock)
as a low-priority user and keeps a reader running. When a capture wants the cameras (or the last
viewer leaves), it terminates the reader, kills it if it hasn't exited after STOP_GRACE, and only
releases the lock once the process is gone: the kernel has closed the device by then, so a
capture never runs while the live view still has a camera open, and never waits on a stuck
reader for more than ~STOP_GRACE + KILL_WAIT (well inside camlock.CAPTURE_WAIT).
A missing or unplugged camera is retried inside the reader with back-off (3 s up to 30 s; its
`status:` stderr lines set the status), so no new process is started per attempt. When a capture
interrupts the back-off, its outcome (camlock `capture.<cam>.json`) decides what happens next: a
capture of the live camera that worked resets the back-off and retries at once; one that couldn't
open the camera counts as a failed try; anything else (e.g. a capture of another camera) leaves the
back-off where it was, and the next reader waits out the rest of the current delay. Outcomes are
checked before every reader start, by time (all captures since the last failed try or the last
outcome applied), so captures made while no page watched (e.g. for 20 s after a `show`) count too.
Frames go only to the local page, never to the model.
"""

from __future__ import annotations

import logging
import math
import re
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
RETRY_IN = re.compile(r" \(retrying in ([0-9.]+) s\)$")
BACKOFF = re.compile(r"backoff: missing=(yes|no) retry=(\d+(?:\.\d+)?)")


class LiveStream:
    def __init__(self, command: list[str] | None, lock: CameraLock, unavailable: str | None = None,
                 cam: str | None = None):
        self.command = command
        self.cam = cam  # the live camera, whose capture outcomes adjust the back-off
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
        self._attempts = 0  # failed opens since the last frame (carried across readers)
        self._next_try = 0.0  # monotonic time of the next open attempt while backing off (0: at once)
        self._outcome_since = time.time()  # captures that ended before this (wall-clock) are already applied
        self._reason = ""  # the last "unavailable: ..." text, to show again while waiting out a delay
        self._missing = False  # the last failed try found the device absent (so a replug ends the wait)
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
                self._attempts, self._next_try = 0, 0.0
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
                backoff = None if line.startswith(STATUS) else BACKOFF.search(line)
                if backoff:  # the reader failed a try and will wait `retry` seconds
                    self._missing = backoff.group(1) == "yes"
                    self._attempts += 1
                    self._next_try = time.monotonic() + float(backoff.group(2))
                    self._outcome_since = max(self._outcome_since, time.time())  # older captures are superseded
                elif line.startswith(STATUS):
                    status = line[len(STATUS):]
                    if status.startswith("unavailable"):
                        self._reason = RETRY_IN.sub("", status)
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

    def _apply_capture_outcome(self) -> None:
        """Before starting a reader: adjust the back-off by the live camera's latest capture since the last
        failed try (made during a pause we noticed or not, e.g. while no page watched)."""
        if self.cam:
            reset, fails, missing, latest = self.lock.outcomes_since(self.cam, self._outcome_since)
            self._outcome_since = latest  # apply each capture once
            if reset:  # a capture worked: the camera is there, so retry now, back-off from the start
                self._attempts, self._next_try = 0, 0.0
            if fails:  # each failed open since then counts as a failed try; wait from the last one
                self._attempts += len(fails)
                ago = max(0.0, time.time() - fails[-1])
                self._next_try = time.monotonic() + max(0.0, retry_delay(self._attempts - 1) - ago)
                self._missing = missing
                why = "is not connected" if missing else "could not be opened"
                self._reason = f"unavailable: Camera '{self.cam}' {why} (last capture)"
        left = max(0.0, self._next_try - time.monotonic())
        if self._attempts and left >= 0.5 and self._reason:
            self._set_status(f"{self._reason} (retrying in {math.ceil(left)} s)")
        elif self.status != "live" and not (self._attempts and self.status.startswith("unavailable")):
            self._set_status("starting")

    def _run(self) -> None:
        assert self.command is not None
        failures = 0  # reader exits in a row without a frame: back off like the reader does
        while self._watched():
            if not self.lock.try_acquire_low():
                self._set_status("paused for a capture")
                time.sleep(0.05)
                continue
            self._apply_capture_outcome()
            proc = drain = None
            error = None
            seq0 = self.seq
            try:
                wait = max(0.0, self._next_try - time.monotonic())
                extra = ["--attempt", str(self._attempts), "--wait", f"{wait:.2f}"]
                if wait > 0 and self._missing:
                    extra.append("--wait-missing")  # a replug while it was stopped ends the wait at once
                proc = subprocess.Popen(self.command + extra,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                threading.Thread(target=self._read_frames, args=(proc,), daemon=True).start()
                last_err = [""]
                drain = threading.Thread(target=self._drain_stderr, args=(proc, last_err), daemon=True)
                drain.start()
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
                if drain is not None:
                    drain.join(0.5)  # its last backoff: line must land before the next outcome check
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
