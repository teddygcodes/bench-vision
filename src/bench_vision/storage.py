"""Persist every image returned to the client under ./captures/YYYY-MM-DD/."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class CaptureStore:
    def __init__(self, root: Path, clock: Callable[[], datetime] = datetime.now):
        self.root = root
        self.clock = clock

    def save(self, camera: str, jpeg: bytes, meta: dict[str, Any]) -> Path:
        now = self.clock()
        day = self.root / now.strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = f"{camera}_{now.strftime('%H%M%S')}"
        n = 0
        while True:  # several captures in the same second get _1, _2, ...
            path = day / (f"{stem}.jpg" if n == 0 else f"{stem}_{n}.jpg")
            try:
                with open(path, "xb") as f:  # exclusive create: safe across threads
                    try:
                        f.write(jpeg)
                    except OSError:
                        f.close()
                        path.unlink(missing_ok=True)
                        raise
                break
            except FileExistsError:
                n += 1
        sidecar = {"camera": camera, "timestamp": now.isoformat(timespec="seconds"), **meta}
        try:
            path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        except OSError:
            path.unlink(missing_ok=True)  # never leave an image without its sidecar
            raise
        return path
