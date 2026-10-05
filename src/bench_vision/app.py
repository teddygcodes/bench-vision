"""The bench-vision service: everything the MCP tools do, minus the MCP plumbing."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

from . import imaging
from .camera import Backend, CameraManager, Frame, MockBackend, OpenCVBackend, load_image_strict
from .config import CameraConfig, Config, load_config
from .errors import BenchVisionError, ConfigError
from .storage import CaptureStore

log = logging.getLogger(__name__)

# A tool result: JPEG bytes become image blocks, strings become text blocks.
Result = list[Union[bytes, str]]

DEFAULT_MAX_EDGE = 1024


def _mock_config(mock: MockBackend) -> Config:
    names = mock.available_cameras()
    if not names:
        raise ConfigError(
            f"--mock was given but {mock.mock_dir} has no images. Add e.g. scope.jpg and side.jpg there."
        )
    cams = {}
    for name in names:
        try:
            img = load_image_strict(mock.images_for(name)[0])
        except (BenchVisionError, IndexError):
            img = None
        if img is None:
            log.warning("mock image for '%s' does not decode; its resolution is unknown", name)
        size = (img.shape[1], img.shape[0]) if img is not None else (0, 0)  # (0, 0) = unknown
        cams[name] = CameraConfig(name=name, device=f"mock:{mock.mock_dir / name}", resolution=size)
    return Config(cameras=cams)


class BenchVision:
    def __init__(
        self,
        root: Path,
        mock: bool = False,
        config_path: Path | None = None,
        backend: Backend | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.root = root
        self.mock = mock
        explicit_config = config_path is not None
        self.config_path = config_path or root / "config.toml"
        self.store = CaptureStore(root / "captures", clock)
        self.config_error: ConfigError | None = None
        self.config: Config | None = None

        if backend is None:
            backend = MockBackend(root / "mock") if mock else OpenCVBackend()
        self.backend = backend
        try:
            if mock and not explicit_config and not self.config_path.exists() and isinstance(backend, MockBackend):
                self.config = _mock_config(backend)
            else:
                self.config = load_config(self.config_path, mock=mock)
        except ConfigError as e:
            self.config_error = e
            log.error("%s", e)
        except (BenchVisionError, OSError) as e:
            msg = str(e) if isinstance(e, BenchVisionError) else f"{e.strerror or 'error'} accessing {e.filename or self.root}"
            self.config_error = ConfigError(f"bench-vision could not start: {msg}")
            log.error("%s", self.config_error)
        except Exception as e:  # noqa: BLE001 - keep the server up so the client sees why
            log.exception("unexpected startup error")
            self.config_error = ConfigError(
                f"bench-vision could not start because of an internal error ({type(e).__name__}). "
                "This is a bug; details are in the server's stderr log."
            )
        cams = self.config.cameras if self.config else {}
        self.cameras = CameraManager(cams, backend)

    # ------------------------------------------------------------------ helpers

    def _ready(self) -> Config:
        if self.config_error is not None:
            raise self.config_error
        assert self.config is not None
        return self.config

    def _relpath(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def _save(self, cam: str, jpeg: bytes, meta: dict[str, Any]) -> str:
        """Save to captures/; on failure, report it rather than losing the image."""
        try:
            return f"Saved {self._relpath(self.store.save(cam, jpeg, meta))}"
        except OSError as e:
            log.error("could not save capture: %s", e)
            return f"WARNING: could not save capture to {self.store.root}: {e.strerror or e}"

    def _frame(self, cam: str, rotate: int) -> tuple[CameraConfig, Frame, int, Any]:
        """Capture one frame; return (config, raw frame, effective rotation, rotated BGR image)."""
        self._ready()
        rotate = imaging.normalize_rotation(rotate)
        cfg = self.cameras.get(cam)
        frame = self.cameras.capture(cam)
        rotation = (cfg.default_rotation + rotate) % 360
        return cfg, frame, rotation, imaging.rotate(frame.image, rotation)

    def _meta(self, cfg: CameraConfig, frame: Frame, rotation: int, **extra: Any) -> dict[str, Any]:
        return {
            "device": cfg.device,
            "mock": self.mock,
            "controls": frame.controls,
            "rotation": rotation,
            "frame_size": list(frame.size),
            **extra,
        }

    # -------------------------------------------------------------------- tools

    def list_cameras(self) -> str:
        cfg = self._ready()
        rows = []
        for name, cam in cfg.cameras.items():
            try:
                present = self.cameras.backend.device_present(cam)
            except (BenchVisionError, OSError):
                present = False
            last = self.cameras.last_size.get(name)
            if last:
                res, source = list(last), "last capture"
            elif cam.resolution == (0, 0):
                res, source = None, "unknown (mock image does not decode)"
            else:
                res, source = list(cam.resolution), "config" if cfg.path else "mock image"
            rows.append(
                {
                    "name": name,
                    "device": cam.device,
                    "connected": present,
                    "status": "open" if self.cameras.open_camera == name else "closed",
                    "resolution": res,
                    "resolution_source": source,
                    "default_rotation": cam.default_rotation,
                }
            )
        return json.dumps({"mode": "mock" if self.mock else "live", "cameras": rows}, indent=2)

    def capture(self, cam: str, rotate: int = 0, max_edge: int = DEFAULT_MAX_EDGE) -> Result:
        max_edge = imaging.check_max_edge(max_edge)
        cfg, frame, rotation, img = self._frame(cam, rotate)
        full_h, full_w = img.shape[:2]
        out = imaging.fit_long_edge(imaging.to_pil(img), max_edge)
        jpeg = imaging.encode_jpeg(out, self._ready().jpeg_quality)
        sx, sy = full_w / out.size[0], full_h / out.size[1]
        saved = self._save(
            cam, jpeg, self._meta(cfg, frame, rotation, tool="capture", crop_box=None, returned_size=list(out.size))
        )
        text = (
            f"{cam}: full frame {full_w}x{full_h} (rotation {rotation}°), returned {out.size[0]}x{out.size[1]}. "
            f"Full-res x = x in this image * {sx:.3f}, full-res y = y * {sy:.3f}. {saved}"
        )
        return [jpeg, text]
