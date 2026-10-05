"""The bench-vision service: everything the MCP tools do, minus the MCP plumbing."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

from . import imaging
from .camera import Backend, CameraManager, Frame, MockBackend, OpenCVBackend, load_image_strict
from .config import CameraConfig, Config, load_config
from .errors import BenchVisionError, ConfigError
from .grid_state import GridStore
from .storage import CaptureStore

log = logging.getLogger(__name__)

# A tool result: JPEG bytes become image blocks, strings become text blocks.
Result = list[Union[bytes, str]]

DEFAULT_MAX_EDGE = 1024
REGION_LONG_EDGE = 768
CELL_RE = re.compile(r"^\s*([A-Za-z])\s*0*([1-9][0-9]{0,2})\s*$")
ARCHIVE_JPEG_QUALITY = 92  # captures/ keeps full resolution at high quality


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
        self.clock = clock
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
        # Kept out of captures/ (only images + sidecars there); mock and live never share grids.
        self.grids = GridStore(root / ".bench-vision" / ("grids-mock.json" if mock else "grids.json"))

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

    def _save(self, cam: str, full_res: Any, meta: dict[str, Any]) -> str:
        """Save the full-resolution image (BGR) to captures/; on failure, report it rather than fail."""
        try:
            jpeg = imaging.encode_jpeg(imaging.to_pil(full_res), ARCHIVE_JPEG_QUALITY)
            return f"Saved {self._relpath(self.store.save(cam, jpeg, meta))}"
        except OSError as e:
            log.error("could not save capture: %s", e)
            return f"WARNING: this capture was not saved to {self._relpath(self.store.root)}/ ({e.strerror or e})."

    def _frame(self, cam: str, rotate: int) -> tuple[CameraConfig, Frame, int, Any]:
        """Capture one frame; return (config, raw frame, effective rotation, rotated BGR image)."""
        self._ready()
        rotate = imaging.normalize_rotation(rotate)
        cfg = self.cameras.get(cam)
        frame = self.cameras.capture(cam)
        rotation = (cfg.default_rotation + rotate) % 360
        return cfg, frame, rotation, imaging.rotate(frame.image, rotation)

    def _meta(
        self, cfg: CameraConfig, frame: Frame, rotation: int, rotated: Any, returned: Any,
        max_edge: int, **extra: Any,
    ) -> dict[str, Any]:
        """Sidecar contents. Sizes are [w, h]; the saved image is always the rotated full frame
        (full_res_size), and crop_box says which part of it was returned to the model."""
        return {
            "device": cfg.device,
            "mock": self.mock,
            "controls": frame.controls,
            "rotation": rotation,
            "full_res_size": [rotated.shape[1], rotated.shape[0]],
            "max_edge": max_edge,
            "returned_size": list(returned.size),
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
        self._ready()  # a broken config is reported before any argument problem
        max_edge = imaging.check_max_edge(max_edge)
        cfg, frame, rotation, img = self._frame(cam, rotate)
        full_h, full_w = img.shape[:2]
        out = imaging.fit_long_edge(imaging.to_pil(img), max_edge)
        jpeg = imaging.encode_jpeg(out, self._ready().jpeg_quality)
        sx, sy = full_w / out.size[0], full_h / out.size[1]
        saved = self._save(
            cam, img, self._meta(cfg, frame, rotation, img, out, max_edge, tool="capture", crop_box=None)
        )
        text = (
            f"{cam}: full frame {full_w}x{full_h} (rotation {rotation}°), returned {out.size[0]}x{out.size[1]}. "
            f"Full-res x = x in this image * {sx:.3f}, full-res y = y * {sy:.3f}. {saved}"
        )
        return [jpeg, text]

    # -------------------------------------------------------------- regions

    def _region(
        self, cam: str, cfg: CameraConfig, frame: Frame, rotation: int, img: Any,
        box: tuple[int, int, int, int], requested: tuple[int, int, int, int], max_edge: int, tool: str,
        **extra: Any,
    ) -> Result:
        x, y, w, h = box
        crop = img[y : y + h, x : x + w]
        out = imaging.fit_long_edge(imaging.to_pil(crop), max_edge, upscale=True)
        jpeg = imaging.encode_jpeg(out, self._ready().jpeg_quality)
        clamped = box != requested
        meta = self._meta(
            cfg, frame, rotation, img, out, max_edge, tool=tool, crop_box=list(box),
            requested_box=list(requested) if clamped else None, **extra,
        )
        saved = self._save(cam, img, meta)
        full_h, full_w = img.shape[:2]
        note = (
            f" (requested x={requested[0]}, y={requested[1]}, w={requested[2]}, h={requested[3]} "
            "was clipped to the frame)" if clamped else ""
        )
        text = (
            f"{cam}: region x={x}, y={y}, w={w}, h={h} of the {full_w}x{full_h} frame (rotation {rotation}°)"
            f"{note}, returned {out.size[0]}x{out.size[1]} ({out.size[0] / w:.2f}x zoom). "
            f"Full-res x = {x} + x in this image * {w / out.size[0]:.3f}, "
            f"y = {y} + y * {h / out.size[1]:.3f}. {saved}"
        )
        return [jpeg, text]

    def capture_region(
        self, cam: str, x: int, y: int, w: int, h: int, rotate: int = 0, max_edge: int = REGION_LONG_EDGE
    ) -> Result:
        self._ready()
        max_edge = imaging.check_max_edge(max_edge)
        imaging.validate_box(x, y, w, h)  # before opening the camera
        cfg, frame, rotation, img = self._frame(cam, rotate)
        full_h, full_w = img.shape[:2]
        box = imaging.clamp_box(x, y, w, h, full_w, full_h)
        return self._region(cam, cfg, frame, rotation, img, box, (x, y, w, h), max_edge, "capture_region")

    # ----------------------------------------------------------------- grids

    def grid(self, cam: str, rows: int, cols: int, rotate: int = 0, max_edge: int = DEFAULT_MAX_EDGE) -> Result:
        self._ready()
        imaging.check_grid(rows, cols)
        max_edge = imaging.check_max_edge(max_edge)
        cfg, frame, rotation, img = self._frame(cam, rotate)
        full_h, full_w = img.shape[:2]
        if full_w / cols < imaging.MIN_REGION or full_h / rows < imaging.MIN_REGION:
            raise BenchVisionError(
                f"A {rows}x{cols} grid is too fine for the {full_w}x{full_h} frame "
                f"(cells must be at least {imaging.MIN_REGION} px)."
            )
        small = imaging.fit_long_edge(imaging.to_pil(img), max_edge)
        if small.size[0] / cols < imaging.MIN_OVERLAY_CELL or small.size[1] / rows < imaging.MIN_OVERLAY_CELL:
            hint = " or a larger max_edge" if small.size[0] < full_w else ""
            raise BenchVisionError(
                f"A {rows}x{cols} grid would make cells smaller than {imaging.MIN_OVERLAY_CELL} px in the "
                f"{small.size[0]}x{small.size[1]} overlay image, too small to read. Use fewer rows/cols{hint}."
            )
        out = imaging.draw_grid(small, rows, cols)
        jpeg = imaging.encode_jpeg(out, self._ready().jpeg_quality)
        geometry = {"rows": rows, "cols": cols, "rotation": rotation, "full_res_size": [full_w, full_h],
                    "drawn_at": self.clock().isoformat(timespec="seconds")}
        warn = self.grids.put(cam, geometry)
        saved = self._save(cam, img, self._meta(cfg, frame, rotation, img, out, max_edge,
                                                tool="grid", crop_box=None, grid=geometry))
        _, _, cw, ch = imaging.cell_box(0, 0, rows, cols, full_w, full_h)
        last = imaging.cell_name(rows - 1, cols - 1)
        example = imaging.cell_name(min(1, rows - 1), min(1, cols - 1))
        text = (
            f"{cam}: {rows}x{cols} grid on the {full_w}x{full_h} frame (rotation {rotation}°), returned "
            f"{out.size[0]}x{out.size[1]} (full-res x = x * {full_w / out.size[0]:.3f}, "
            f"y = y * {full_h / out.size[1]:.3f}), "
            f"cells A1 (top-left) to {last} (bottom-right), each about {cw}x{ch} full-res px. "
            f"Rows are letters, columns are numbers. Use capture_cell(cam=\"{cam}\", cell=\"{example}\") to zoom in."
            f"{warn} {saved}"
        )
        return [jpeg, text]

    def capture_cell(self, cam: str, cell: str, max_edge: int = REGION_LONG_EDGE, margin: float = 0.0) -> Result:
        self._ready()
        self.cameras.get(cam)
        max_edge = imaging.check_max_edge(max_edge)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= 1:
            raise BenchVisionError(f"margin must be a fraction of the cell size between 0 and 1, got {margin!r}.")
        grid = self.grids.get(cam)
        if grid is None:
            why = (
                f" (the grid state file {self._relpath(self.grids.path)} could not be read: {self.grids.read_error})"
                if self.grids.read_error else ""
            )
            raise BenchVisionError(
                f"No grid stored for camera '{cam}'{why}. Call grid(cam=\"{cam}\", rows, cols) first."
            )
        rows, cols = grid["rows"], grid["cols"]
        m = CELL_RE.match(cell) if isinstance(cell, str) else None
        last = imaging.cell_name(rows - 1, cols - 1)
        if not m:
            shown = repr(cell) if len(repr(cell)) <= 20 else repr(cell)[:17] + "..."
            raise BenchVisionError(f"Cell must look like 'B3' (row letter, column number); got {shown}.")
        row, col = ord(m.group(1).upper()) - ord("A"), int(m.group(2)) - 1
        if row >= rows or col >= cols:
            raise BenchVisionError(
                f"Cell {m.group(1).upper()}{col + 1} is outside the {rows}x{cols} grid for '{cam}' (A1..{last})."
            )
        cfg = self.cameras.get(cam)
        extra_rotate = (grid["rotation"] - cfg.default_rotation) % 360
        cfg, frame, rotation, img = self._frame(cam, extra_rotate)
        full_h, full_w = img.shape[:2]
        cx, cy, cw, ch = imaging.cell_box(row, col, rows, cols, full_w, full_h)
        name = imaging.cell_name(row, col)
        if cw < imaging.MIN_REGION or ch < imaging.MIN_REGION:
            raise BenchVisionError(f"Cell {name} is too small at {full_w}x{full_h}; use a coarser grid.")
        mx, my = round(cw * margin), round(ch * margin)
        # The margin may extend past the frame edge; that part is simply dropped (not reported as clipping).
        box = imaging.clamp_box(cx - mx, cy - my, cw + 2 * mx, ch + 2 * my, full_w, full_h)
        result = self._region(
            cam, cfg, frame, rotation, img, box, box, max_edge, "capture_cell",
            cell=name, cell_box=[cx, cy, cw, ch], margin=margin, grid=grid,
        )
        drawn = f" drawn {grid['drawn_at']}" if grid.get("drawn_at") else ""
        result[1] = f"Cell {imaging.cell_name(row, col)} of the {rows}x{cols} grid{drawn}. " + result[1]
        if grid.get("full_res_size") and list(grid["full_res_size"]) != [full_w, full_h]:
            result[1] += (
                f" Note: the frame is now {full_w}x{full_h} but the grid was drawn on "
                f"{grid['full_res_size'][0]}x{grid['full_res_size'][1]}; cells were scaled proportionally."
            )
        return result
