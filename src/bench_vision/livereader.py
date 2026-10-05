"""Child process that reads the live camera for `bench-vision display`.

    python -m bench_vision.livereader --root DIR [--mock] [--config FILE] --cam NAME

Writes frames to stdout as <4-byte big-endian length><JPEG>, at most MAX_FPS, rotated like
captures and scaled to a LONG_EDGE long edge. On an error it prints one line to stderr and exits
with status 3. It exits when stdin closes (the display went away). The display kills this process
to free the camera for a capture, so the kernel has closed the device before the camera lock is
released.
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench_vision.livereader")
    p.add_argument("--root", required=True)
    p.add_argument("--cam", required=True)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--config")
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
        bv.load_shared_overrides()
        stream = bv.cameras.backend.open_stream(cam, bv.cameras.controls_for(cam))
    except BenchVisionError as e:
        print(str(e), file=sys.stderr, flush=True)
        return 3
    except Exception as e:  # noqa: BLE001 - reported as a status line, not a traceback
        print(f"internal error ({type(e).__name__})", file=sys.stderr, flush=True)
        return 3
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
                    print(f"Camera '{cam.name}' stopped delivering frames (unplugged?).", file=sys.stderr, flush=True)
                    return 3
                time.sleep(0.05)
                continue
            failing_since = None
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
    except (BrokenPipeError, OSError):
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"camera stopped: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
        return 3
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    code = main()
    sys.stderr.flush()
    os._exit(code)  # not sys.exit: skip interpreter shutdown while the stdin watcher is blocked
