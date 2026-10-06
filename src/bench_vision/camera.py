"""Camera access: lazy, one device at a time, real (OpenCV/V4L2) or mock."""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image as PILImage

from .config import NAME_RE, CameraConfig
from .errors import CameraError, CameraMissingError, CameraOpenError
from .v4l2 import SUBPROCESS_TIMEOUT, V4L2, ControlInfo, validate_control

log = logging.getLogger(__name__)

MOCK_EXTS = (".jpg", ".jpeg", ".png")
OPEN_TIMEOUT = 5.0  # seconds for the device to open
FRAME_TIMEOUT = 5.0  # seconds for each frame
RELEASE_TIMEOUT = 5.0  # seconds to wait for the device to close after a capture
# Auto-exposure needs time after the device opens: the Arducam's first ~10 frames keep the previous
# exposure, then it takes ~1 s to settle (measured: mean brightness 5.5 at frame 5, 11 by 1.9 s).
# So read until AE_SETTLE seconds after the first frame, then until the brightness holds steady
# (AE_STEADY relative change over AE_STEADY_FRAMES), until at most AE_SETTLE_MAX seconds after the
# first frame; it also stops after AE_BAD_READS failed reads in a row (e.g. unplugged mid-settle).
# Skipped when the config/set_control puts exposure in manual mode.
AE_SETTLE = 1.5
AE_SETTLE_MAX = 2.5
AE_STEADY = 0.03
AE_STEADY_FRAMES = 5
AE_BAD_READS = 3
LOCK_TIMEOUT = 90.0  # longest a capture waits for another one to finish


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


def is_auto_switch(name: str) -> bool:
    """Controls that switch an automatic mode on/off and gate other controls (e.g. focus_absolute)."""
    return name.endswith("_auto") or "automatic" in name or name.startswith("auto_")


def missing_device_error(cam: CameraConfig) -> CameraError:
    return CameraMissingError(
        f"Camera '{cam.name}' is not connected: {cam.source} [cameras.{cam.name}] device = "
        f"\"{cam.device}\" does not exist. Check the USB cable, or run `uv run bench-vision setup` "
        "to list the by-id paths that are present now."
    )


class OpenCVBackend:
    """Real UVC cameras via OpenCV's V4L2 backend, controls via v4l2-ctl."""

    def __init__(self, v4l2: V4L2 | None = None):
        self.v4l2 = v4l2 or V4L2()
        self._worker: tuple[str, threading.Thread] | None = None  # last capture thread (may be stuck)

    def device_present(self, cam: CameraConfig) -> bool:
        try:
            return Path(cam.device).exists()
        except OSError:
            return False

    def list_ctrls(self, cam: CameraConfig) -> dict[str, ControlInfo]:
        self.v4l2.require()  # "needs Linux/v4l2" comes before "not connected"
        if not self.device_present(cam):
            raise missing_device_error(cam)
        return self.v4l2.list_ctrls(cam.device)

    def set_ctrl(self, cam: CameraConfig, name: str, value: int) -> int | None:
        self.v4l2.require()
        if not self.device_present(cam):
            raise missing_device_error(cam)
        self.v4l2.set_ctrl(cam.device, name, value)
        return self.v4l2.get_ctrl(cam.device, name)

    def _apply_controls(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> dict[str, int]:
        if not controls:
            return {}
        available = self.v4l2.list_ctrls(cam.device)
        applied: dict[str, int] = {}
        # Order matters: e.g. focus_automatic_continuous=0 must precede focus_absolute.
        for name, value in controls:
            info = validate_control(available, name, value, f"{cam.source} [cameras.{cam.name}] v4l2_controls")
            if info.read_only:  # the driver refuses it (EACCES); don't fail every capture over it
                log.warning("camera %s: skipping %s=%s (read-only on this camera)", cam.name, name, value)
                continue
            if info.inactive:  # e.g. focus_absolute while autofocus is on: the driver would refuse it
                log.warning("camera %s: skipping %s=%s (inactive while its auto mode is on)", cam.name, name, value)
                continue
            self.v4l2.set_ctrl(cam.device, name, value)
            applied[name] = value
            if is_auto_switch(name):
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

    def open_stream(self, cam: CameraConfig, controls: list[tuple[str, int]] = ()) -> OpenCVStream:
        if not self.device_present(cam):
            raise missing_device_error(cam)
        if self._worker is not None and self._worker[1].is_alive():
            raise CameraError(f"Camera '{self._worker[0]}' is stuck in an earlier capture.")
        stream = OpenCVStream(cam)
        try:
            self._apply_controls(cam, list(controls))  # same controls a capture applies on open
        except BaseException:
            stream.close()
            raise
        return stream

    def _grab(self, cam: CameraConfig, controls: list[tuple[str, int]]) -> Frame:
        if self._worker is not None and self._worker[1].is_alive():
            raise CameraError(
                f"Camera '{self._worker[0]}' is still stuck in an earlier capture that timed out, so no camera "
                "can be opened safely. Unplug and replug it, or restart the bench-vision server."
            )
        if not self.device_present(cam):
            raise missing_device_error(cam)
        real = str(Path(cam.device).resolve())
        events: queue.Queue = queue.Queue()
        abandon = threading.Event()

        settle = not _manual_exposure(controls)

        def work() -> None:
            # Runs the blocking OpenCV calls so the caller can give up on a hung device.
            cap = None
            try:
                cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
                events.put(("opened", cap.isOpened()))
                if not cap.isOpened() or abandon.is_set():
                    return
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.resolution[1])
                events.put(("controls", self._apply_controls(cam, controls)))
                first = None
                means: list[float] = []
                n = bad = 0
                while True:
                    if abandon.is_set():
                        return
                    ok, img = cap.read()
                    good = ok and img is not None and img.size > 0
                    events.put(("frame", img if good else None))
                    n += 1
                    bad = 0 if good else bad + 1
                    if n >= cam.warmup_frames and first is not None and bad >= AE_BAD_READS:
                        log.warning("camera %s: reads failing while exposure settled; keeping the last good frame",
                                    cam.name)
                        return
                    if good:
                        first = first or time.monotonic()
                        means.append(float(cv2.resize(img, (32, 24), interpolation=cv2.INTER_AREA).mean()))
                    if n >= cam.warmup_frames and (not settle or first is None or _exposure_settled(means, first)):
                        return
            except BaseException as e:  # noqa: BLE001 - handed to the caller
                events.put(("error", e))
            finally:
                if cap is not None:
                    cap.release()
                events.put(("released", None))

        worker = threading.Thread(target=work, name=f"capture-{cam.name}", daemon=True)
        self._worker = (cam.name, worker)
        worker.start()

        def wait(expect: str, timeout: float, what: str) -> Any:
            try:
                kind, value = events.get(timeout=timeout)
            except queue.Empty:
                nonlocal timed_out
                timed_out = True
                abandon.set()
                raise (CameraOpenError if expect == "opened" else CameraError)(
                    f"Camera '{cam.name}' ({cam.device}) did not {what} within {timeout:g} s; it may be hung. "
                    "Unplug and replug it."
                ) from None
            if kind == "error":
                raise value
            if kind != expect:  # worker ended early (e.g. could not open)
                return None
            return value

        def next_event(timeout: float, what: str) -> tuple[str, Any]:
            nonlocal timed_out
            try:
                kind, value = events.get(timeout=timeout)
            except queue.Empty:
                timed_out = True
                abandon.set()
                raise CameraError(
                    f"Camera '{cam.name}' ({cam.device}) did not {what} within {timeout:g} s; it may be hung. "
                    "Unplug and replug it."
                ) from None
            if kind == "error":
                raise value
            return kind, value

        timed_out = False
        try:
            if not wait("opened", OPEN_TIMEOUT, "open"):
                raise CameraOpenError(
                    f"Camera '{cam.name}' ({cam.device}) exists but could not be opened. "
                    "Is another program (cheese, OBS, a second bench-vision) using it, and is your "
                    "user in the 'video' group?"
                )
            # Each v4l2-ctl call has its own timeout; allow for all of them.
            applied = wait("controls", OPEN_TIMEOUT + 2 * SUBPROCESS_TIMEOUT * (len(controls) + 1), "accept its controls")
            frame = None
            while True:  # frames until the worker is done (warm-up, then auto-exposure settling)
                kind, img = next_event(FRAME_TIMEOUT, "deliver a frame")
                if kind != "frame":
                    break
                if img is not None:
                    frame = img
        finally:
            # Don't let the next capture start until this device is really closed. After a timeout
            # the worker is hung, so don't wait: the next capture refuses while it is still alive.
            worker.join(0 if timed_out else RELEASE_TIMEOUT)
        if frame is None:
            raise CameraError(
                f"Camera '{cam.name}' ({cam.device}) opened but returned no frames. "
                f"Try unplugging it, or lower [cameras.{cam.name}] resolution."
            )
        h, w = frame.shape[:2]
        if (w, h) != cam.resolution:
            log.warning("camera %s: requested %sx%s, got %sx%s", cam.name, *cam.resolution, w, h)
        return Frame(cam.name, frame, applied or {})


def _manual_exposure(controls: list[tuple[str, int]]) -> bool:
    # auto_exposure (newer kernels) / exposure_auto (older): 1 = manual mode; the last setting applied wins
    modes = [value for name, value in controls if name in ("auto_exposure", "exposure_auto")]
    return bool(modes) and modes[-1] == 1


def _exposure_settled(means: list[float], first: float) -> bool:
    elapsed = time.monotonic() - first
    if elapsed >= AE_SETTLE_MAX:
        return True
    if elapsed < AE_SETTLE or len(means) < AE_STEADY_FRAMES:
        return False
    recent = means[-AE_STEADY_FRAMES:]
    return max(recent) - min(recent) <= max(1.0, AE_STEADY * max(recent))


class OpenCVStream:
    """A held-open low-resolution reader for the live view (one per open; close() releases the device)."""

    def __init__(self, cam: CameraConfig):
        self.cam = cam
        real = str(Path(cam.device).resolve())
        self.cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            raise CameraOpenError(f"Camera '{cam.name}' ({cam.device}) could not be opened for the live view.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # we read slower than the camera: keep frames fresh

    def read(self) -> np.ndarray | None:
        ok, img = self.cap.read()
        return img if ok and img is not None and img.size > 0 else None

    def close(self) -> None:
        self.cap.release()


class MockStream:
    """Live view in --mock mode: the mock image with a running clock, so motion is visible."""

    def __init__(self, image: np.ndarray, fps: float = 15.0):
        self.image = image
        self.period = 1.0 / fps

    def read(self) -> np.ndarray | None:
        import time as _time

        _time.sleep(self.period)
        frame = self.image.copy()
        h, w = frame.shape[:2]
        text = f"LIVE (mock) {_time.strftime('%H:%M:%S')}.{int(_time.time() * 10) % 10}"
        scale = max(w, h) / 1400
        cv2.putText(frame, text, (int(w * 0.02), int(h - h * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), max(2, int(scale * 6)), cv2.LINE_AA)
        cv2.putText(frame, text, (int(w * 0.02), int(h - h * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, int(scale * 2)), cv2.LINE_AA)
        return frame

    def close(self) -> None:
        pass


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
            raise CameraMissingError(
                f"Mock camera '{cam.name}' has no image: put {cam.name}.jpg (or a {cam.name}/ folder "
                f"of images) in {self.mock_dir}."
            )
        return imgs

    def list_ctrls(self, cam: CameraConfig) -> dict[str, ControlInfo]:
        self._require(cam)
        vals = self._values.get(cam.name, {})

        def flags(n: str) -> str:
            # Like real UVC cameras: manual values are inactive while the auto mode is on.
            if n == "focus_absolute" and vals.get("focus_automatic_continuous", 1) == 1:
                return "inactive"
            if n == "exposure_time_absolute" and vals.get("auto_exposure", 3) not in (1, 2):  # manual, shutter prio
                return "inactive"
            return ""

        ctrls = {
            n: ControlInfo(c.name, c.type, c.min, c.max, c.step, c.default, vals.get(n, c.value), flags(n))
            for n, c in MOCK_CONTROLS.items()
        }
        for n, v in cam.v4l2_controls:  # whatever a real config names exists here too
            ctrls.setdefault(n, ControlInfo(n, "int", value=vals.get(n, v)))
        return ctrls

    def set_ctrl(self, cam: CameraConfig, name: str, value: int) -> int | None:
        self._values.setdefault(cam.name, {})[name] = value
        return value

    def open_stream(self, cam: CameraConfig, controls: list[tuple[str, int]] = ()) -> MockStream:
        img = load_image_strict(self._require(cam)[0])
        if img is None:
            raise CameraError(f"Mock image for '{cam.name}' could not be decoded.")
        for name, value in controls:
            self.set_ctrl(cam, name, value)
        return MockStream(img)

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

    def __init__(self, cameras: dict[str, CameraConfig], backend: Backend, camera_lock: Any = None):
        self.cameras = cameras
        self.backend = backend
        self.camera_lock = camera_lock  # cross-process lock shared with the live view (camlock.CameraLock)
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
        overrides = dict(self.overrides.get(cam.name, {}))  # snapshot: set_control may edit concurrently
        # Config controls without an override (config order), then overrides (the order they were set),
        # then a stable sort that puts auto-mode switches first: a manual value is only accepted by
        # the driver once its auto mode is off, e.g. after a replug reset the camera to auto.
        ordered = [(n, v) for n, v in cam.v4l2_controls if n not in overrides] + list(overrides.items())
        return sorted(ordered, key=lambda nv: not is_auto_switch(nv[0]))

    def capture(self, name: str) -> Frame:
        cam = self.get(name)
        if not self._lock.acquire(timeout=LOCK_TIMEOUT):
            raise CameraError(
                f"Camera '{name}' is waiting on another capture that has not finished after {LOCK_TIMEOUT:g} s."
            )
        try:
            # The live view (another process) yields within ~0.5 s; it never blocks this for long.
            with self.camera_lock.for_capture(name) if self.camera_lock else contextlib.nullcontext():
                self.open_camera = name
                frame = self.backend.grab(cam, self.controls_for(cam))
        finally:
            self.open_camera = None
            self._lock.release()  # always, even if the grab failed or timed out
        self.last_size[name] = frame.size
        return frame
