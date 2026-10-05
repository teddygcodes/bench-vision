"""Camera access: lazy, one device at a time, real (OpenCV/V4L2) or mock."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image as PILImage

from .config import NAME_RE, CameraConfig
from .errors import CameraError
from .v4l2 import V4L2, ControlInfo, validate_control

log = logging.getLogger(__name__)

MOCK_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Frame:
    camera: str
    image: np.ndarray  # BGR, full resolution, before rotation
    controls: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class Backend(Protocol):
    def grab(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> Frame: ...

    def device_present(self, cam: CameraConfig) -> bool: ...

    def list_ctrls(self, cam: CameraConfig) -> dict[str, ControlInfo]: ...

    def set_ctrl(self, cam: CameraConfig, name: str, value: int) -> int | None: ...


def missing_device_error(cam: CameraConfig) -> CameraError:
    return CameraError(
        f"Camera '{cam.name}' is not connected: {cam.source} [cameras.{cam.name}] device = "
        f"\"{cam.device}\" does not exist. Check the USB cable, or run `uv run bench-vision setup` "
        "to list the by-id paths that are present now."
    )


class OpenCVBackend:
    """Real UVC cameras via OpenCV's V4L2 backend, controls via v4l2-ctl."""

    def __init__(self, v4l2: V4L2 | None = None):
        self.v4l2 = v4l2 or V4L2()

    def device_present(self, cam: CameraConfig) -> bool:
        try:
            return Path(cam.device).exists()
        except OSError:
            return False

    def list_ctrls(self, cam: CameraConfig) -> dict[str, ControlInfo]:
        if not self.device_present(cam):
            raise missing_device_error(cam)
        return self.v4l2.list_ctrls(cam.device)

    def set_ctrl(self, cam: CameraConfig, name: str, value: int) -> int | None:
        self.v4l2.set_ctrl(cam.device, name, value)
        return self.v4l2.get_ctrl(cam.device, name)

    def _apply_controls(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> dict[str, int]:
        if not controls:
            return {}
        available = self.v4l2.list_ctrls(cam.device)
        applied: dict[str, int] = {}
        # Order matters: e.g. focus_automatic_continuous=0 must precede focus_absolute.
        for name, value in controls:
            validate_control(available, name, value, f"{cam.source} [cameras.{cam.name}] v4l2_controls")
            self.v4l2.set_ctrl(cam.device, name, value)
            applied[name] = value
            if name.endswith("_auto") or "automatic" in name or name == "auto_exposure":
                # Dependent controls become active only after the auto mode changes.
                available = self.v4l2.list_ctrls(cam.device)
        return applied

    def grab(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> Frame:
        try:
            return self._grab(cam, controls)
        except (cv2.error, OSError) as e:
            log.error("camera %s: capture failed: %s", cam.name, e)
            kind = "the V4L2 driver reported an error" if isinstance(e, cv2.error) else (
                f"the operating system reported: {e.strerror}" if getattr(e, "strerror", None) else "I/O failed"
            )
            raise CameraError(
                f"Camera '{cam.name}' ({cam.device}) stopped delivering frames ({kind}). "
                "Try unplugging and replugging it; details are in the server's stderr log."
            ) from None

    def _grab(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> Frame:
        if not self.device_present(cam):
            raise missing_device_error(cam)
        real = str(Path(cam.device).resolve())
        cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise CameraError(
                    f"Camera '{cam.name}' ({cam.device}) exists but could not be opened. "
                    "Is another program (cheese, OBS, a second bench-vision) using it, and is your "
                    "user in the 'video' group?"
                )
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.resolution[1])
            applied = self._apply_controls(cam, controls)
            frame = None
            for _ in range(cam.warmup_frames):
                ok, img = cap.read()
                if ok and img is not None and img.size > 0:
                    frame = img
            if frame is None:
                raise CameraError(
                    f"Camera '{cam.name}' ({cam.device}) opened but returned no frames. "
                    "Try unplugging it, or lower [cameras.{0}] resolution.".format(cam.name)
                )
            h, w = frame.shape[:2]
            if (w, h) != cam.resolution:
                log.warning(
                    "camera %s: requested %sx%s, got %sx%s", cam.name, *cam.resolution, w, h
                )
            return Frame(cam.name, frame, applied)
        finally:
            cap.release()


# Controls a mock camera pretends to have, so set_control is testable offline.
MOCK_CONTROLS: dict[str, ControlInfo] = {
    c.name: c
    for c in (
        ControlInfo("brightness", "int", -64, 64, 1, 0, 0),
        ControlInfo("contrast", "int", 0, 64, 1, 32, 32),
        ControlInfo("gain", "int", 0, 100, 1, 0, 0),
        ControlInfo("auto_exposure", "menu", 0, 3, 1, 3, 3),
        ControlInfo("exposure_time_absolute", "int", 1, 5000, 1, 157, 157),
        ControlInfo("focus_automatic_continuous", "bool", 0, 1, 1, 1, 1),
        ControlInfo("focus_absolute", "int", 0, 1023, 1, 0, 0),
    )
}


def load_image_strict(path: Path) -> np.ndarray | None:
    """Decode an image file to BGR, returning None if it is unreadable, corrupt or truncated."""
    try:
        with PILImage.open(path) as im:
            im.load()  # raises on truncated data, unlike cv2.imread
            rgb = np.asarray(im.convert("RGB"))
    except (PermissionError, IsADirectoryError) as e:
        raise CameraError(f"Image {path} is not readable: {e.strerror}. Check its permissions.") from None
    except PILImage.DecompressionBombError:
        raise CameraError(f"Image {path} is too large to decode safely; use one under ~89 megapixels.") from None
    except (OSError, ValueError):
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class MockBackend:
    """Serves still images from a directory instead of opening devices.

    For camera `name` it uses mock/<name>.jpg|.jpeg|.png, or cycles through the
    images in mock/<name>/ on successive captures.
    """

    def __init__(self, mock_dir: Path):
        self.mock_dir = mock_dir
        self._counters: dict[str, int] = {}
        self._values: dict[str, dict[str, int]] = {}

    def images_for(self, name: str) -> list[Path]:
        """mock/<name>/* if that folder has images, else mock/<name>.<jpg|jpeg|png> (any case)."""
        sub = self.mock_dir / name
        try:
            if sub.is_dir():
                found = sorted(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in MOCK_EXTS)
                if found:
                    return found
            return sorted(
                p
                for p in self.mock_dir.iterdir()
                if p.stem == name and p.suffix.lower() in MOCK_EXTS and p.is_file()
            )
        except OSError as e:
            raise CameraError(f"Could not read mock images for '{name}' in {self.mock_dir}: {e.strerror or e}.") from None

    def available_cameras(self) -> list[str]:
        """Camera names found in the mock directory (only names valid in config.toml)."""
        names = set()
        try:
            if not self.mock_dir.is_dir():
                return []
            for p in self.mock_dir.iterdir():
                name = p.name if p.is_dir() else p.stem
                if not NAME_RE.fullmatch(name):
                    log.warning("ignoring mock image %s: name must match %s", p, NAME_RE.pattern)
                elif p.is_dir() or p.suffix.lower() in MOCK_EXTS:
                    try:
                        if self.images_for(name):
                            names.add(name)
                    except CameraError as e:  # one bad folder must not hide the other cameras
                        log.warning("%s", e)
        except OSError as e:
            raise CameraError(f"Could not read mock directory {self.mock_dir}: {e.strerror or e}.") from None
        return sorted(names)

    def device_present(self, cam: CameraConfig) -> bool:
        return any(os.access(p, os.R_OK) for p in self.images_for(cam.name))

    def _require(self, cam: CameraConfig) -> list[Path]:
        imgs = self.images_for(cam.name)
        if not imgs:
            raise CameraError(
                f"Mock camera '{cam.name}' has no image: put {cam.name}.jpg (or a {cam.name}/ folder "
                f"of images) in {self.mock_dir}."
            )
        return imgs

    def list_ctrls(self, cam: CameraConfig) -> dict[str, ControlInfo]:
        self._require(cam)
        vals = self._values.get(cam.name, {})
        ctrls = {
            n: ControlInfo(c.name, c.type, c.min, c.max, c.step, c.default, vals.get(n, c.value))
            for n, c in MOCK_CONTROLS.items()
        }
        for n, v in cam.v4l2_controls:  # whatever a real config names exists here too
            ctrls.setdefault(n, ControlInfo(n, "int", value=vals.get(n, v)))
        return ctrls

    def set_ctrl(self, cam: CameraConfig, name: str, value: int) -> int | None:
        self._values.setdefault(cam.name, {})[name] = value
        return value

    def grab(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> Frame:
        imgs = self._require(cam)
        i = self._counters.get(cam.name, 0)
        self._counters[cam.name] = i + 1
        path = imgs[i % len(imgs)]
        img = load_image_strict(path)
        if img is None:
            raise CameraError(f"Mock image {path} could not be decoded (corrupt or truncated?); use a valid JPEG/PNG.")
        # Record config controls as applied without checking them: the mock control
        # table is invented, and a real config.toml must keep working under --mock.
        applied = dict(controls)
        for name, value in controls:
            self.set_ctrl(cam, name, value)
        return Frame(cam.name, img, applied)


class CameraManager:
    """Opens cameras lazily and never more than one at a time."""

    def __init__(self, cameras: dict[str, CameraConfig], backend: Backend):
        self.cameras = cameras
        self.backend = backend
        self._lock = threading.Lock()
        self.open_camera: str | None = None
        self.last_size: dict[str, tuple[int, int]] = {}
        # Values changed at runtime with set_control; re-applied after config controls on open.
        self.overrides: dict[str, dict[str, int]] = {}

    def get(self, name: str) -> CameraConfig:
        cam = self.cameras.get(name)
        if cam is None:
            names = ", ".join(self.cameras) or "(none configured)"
            shown = name if len(name) <= 40 else name[:37] + "..."
            raise CameraError(f"Unknown camera '{shown}'. Configured cameras: {names}.")
        return cam

    def controls_for(self, cam: CameraConfig) -> list[tuple[str, int]]:
        merged = dict(cam.v4l2_controls)
        merged.update(self.overrides.get(cam.name, {}))
        # Keep config order (auto-mode switches first), then any new overrides.
        order = [n for n, _ in cam.v4l2_controls] + [
            n for n in self.overrides.get(cam.name, {}) if n not in dict(cam.v4l2_controls)
        ]
        return [(n, merged[n]) for n in order]

    def capture(self, name: str) -> Frame:
        cam = self.get(name)
        with self._lock:
            self.open_camera = name
            try:
                frame = self.backend.grab(cam, self.controls_for(cam))
            finally:
                self.open_camera = None
        self.last_size[name] = frame.size
        return frame
