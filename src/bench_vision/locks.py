"""A bounded, best-effort cross-process lock on a sibling .lock file."""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


@contextlib.contextmanager
def file_lock(lock_path: Path, deadline: float) -> Iterator[bool]:
    """Hold an exclusive flock on lock_path until the block exits.

    Yields False if another process still holds it at `deadline` (time.monotonic()),
    and True otherwise, including when the lock file itself can't be used (then the
    block runs unlocked rather than failing the tool call).
    """
    fd = None
    if fcntl is not None:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NONBLOCK, 0o644)
        except OSError:
            fd = None
    if fd is not None:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    yield False
                    return
                time.sleep(0.05)
            except OSError:
                os.close(fd)
                fd = None
                break
    try:
        yield True
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
