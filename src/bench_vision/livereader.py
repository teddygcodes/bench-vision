"""Child process that reads the live camera for `bench-vision display`.

    python -m bench_vision.livereader --root DIR [--mock] [--config FILE] --cam NAME [--attempt N] [--wait S [--wait-missing]]

Writes frames to stdout as <4-byte big-endian length><JPEG>, at most MAX_FPS, rotated like
captures and scaled to a LONG_EDGE long edge. If the camera is missing, won't open, or stops
delivering frames (unplugged), it prints a `status: unavailable: ...` line to stderr and tries
again itself after RETRY_DELAYS (3 s, 6 s, 12 s, 24 s, then every 30 s; back to 3 s once frames
flow), so the display doesn't start a new process for each attempt. --attempt and --wait (seconds
before the first try) continue the back-off of a reader the display stopped for a capture;
--wait-missing ends that wait as soon as the (last seen missing) device is present. Before each
`status: unavailable: ...` line (for people) it prints `backoff: missing=yes|no retry=<seconds>` (for
the display). Errors no retry can fix (bad
config, unknown camera) print one line and exit with status 3. It exits when stdin closes (the
display went away). The display kills this process to free the camera for a capture, so the kernel
has closed the device before the camera lock is released.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import threading
import time
from pathlib import Path

LONG_EDGE = 960
MAX_FPS = 15.0
NO_FRAMES = 3.0  # seconds of failed reads before the reader reports the camera gone
RETRY_DELAYS = (3.0, 6.0, 12.0, 24.0, 30.0)  # waits between attempts to (re)open; the last repeats
PRESENCE_CHECK = 1.0  # seconds between checks for a replugged camera while waiting to retry
STATUS = "status: "  # stderr lines with this prefix set the display's live-view status


def retry_delay(attempt: int) -> float:
    """Wait before retry number `attempt` (0-based)."""
    return RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]


def _say(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench_vision.livereader")
    p.add_argument("--root", required=True)
    p.add_argument("--cam", required=True)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--config")
    p.add_argument("--attempt", type=int, default=0)
    p.add_argument("--wait", type=float, default=0.0)
    p.add_argument("--wait-missing", action="store_true")
    args = p.parse_args(argv)
    os.environ["OPENCV_LOG_LEVEL"] = "WARNING"

    # Exit as soon as the parent goes away (its end of our stdin closes).
    # Raw fd reads: a thread blocked in sys.stdin's buffered reader makes the interpreter abort at
    # shutdown ("Fatal Python error: _enter_buffered_busy"), which would replace our error line.
    def watch_parent() -> None:
        try:
            while os.read(0, 4096):
                pass
        finally:
            os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()

    import cv2

    from . import imaging
    from .app import BenchVision
    from .errors import BenchVisionError

    out = sys.stdout.buffer
    try:
        bv = BenchVision(Path(args.root), mock=args.mock, config_path=Path(args.config) if args.config else None)
        if bv.config_error is not None:
            raise bv.config_error
        cam = bv.cameras.get(args.cam)
    except BenchVisionError as e:
        _say(str(e))
        return 3
    except Exception as e:  # noqa: BLE001 - reported as a status line, not a traceback
        _say(f"internal error ({type(e).__name__})")
        return 3

    def present() -> bool:
        try:
            return bv.cameras.backend.device_present(cam)
        except Exception:  # noqa: BLE001
            return False

    def pause(seconds: float, missing: bool) -> None:
        # Sleep, but retry at once if a missing device reappears (replugged) instead of waiting up to 30 s.
        end = time.monotonic() + seconds
        while not (missing and present()):
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(PRESENCE_CHECK, left))

    def unavailable(why: str, attempt: int) -> None:
        missing = not present()
        why = " ".join(why.split())
        why = why if len(why) <= 240 else why[:239] + "…"  # the display shows 300 chars: keep the retry time
        _say(f"backoff: missing={'yes' if missing else 'no'} retry={retry_delay(attempt):g}")
        _say(f"{STATUS}unavailable: {why} (retrying in {retry_delay(attempt):g} s)")
        pause(retry_delay(attempt), missing)

    attempt = max(0, args.attempt)
    if args.wait > 0:  # the rest of a back-off delay that a capture interrupted
        pause(min(args.wait, RETRY_DELAYS[-1]), args.wait_missing)
    while True:
        try:
            bv.load_shared_overrides()  # pick up set_control values made since the last attempt
            stream = bv.cameras.backend.open_stream(cam, bv.cameras.controls_for(cam))
        except BenchVisionError as e:
            unavailable(str(e), attempt)
            attempt += 1
            continue
        except Exception as e:  # noqa: BLE001
            unavailable(f"could not open the camera ({type(e).__name__})", attempt)
            attempt += 1
            continue
        why = None
        next_at = 0.0
        failing_since: float | None = None
        try:
            while True:
                img = stream.read()
                if img is None:
                    # e.g. unplugged: read() fails at once, so don't spin; give up after NO_FRAMES seconds
                    now = time.monotonic()
                    failing_since = failing_since or now
                    if now - failing_since > NO_FRAMES:
                        why = f"Camera '{cam.name}' stopped delivering frames (unplugged?)."
                        break
                    time.sleep(0.05)
                    continue
                failing_since = None
                attempt = 0  # frames flow again: the next failure starts the back-off from the beginning
                img = imaging.rotate(img, cam.default_rotation)
                h, w = img.shape[:2]
                scale = LONG_EDGE / max(h, w)
                if scale < 1:
                    img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                                     interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    data = buf.tobytes()
                    out.write(struct.pack(">I", len(data)) + data)
                    out.flush()
                next_at = max(next_at + 1.0 / MAX_FPS, time.monotonic())
                time.sleep(max(0.0, next_at - time.monotonic()))
        except BrokenPipeError:
            return 0  # the display went away
        except Exception as e:  # noqa: BLE001
            why = f"camera stopped ({type(e).__name__})"
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        unavailable(why, attempt)
        attempt += 1


if __name__ == "__main__":
    code = main()
    sys.stderr.flush()
    os._exit(code)  # not sys.exit: skip interpreter shutdown while the stdin watcher is blocked
