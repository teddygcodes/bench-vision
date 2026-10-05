"""The bench-vision service: everything the MCP tools do, minus the MCP plumbing."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

from . import imaging
from .camera import Backend, CameraManager, Frame, MockBackend, OpenCVBackend, load_image_strict
from .config import CameraConfig, Config, load_config
from .errors import BenchVisionError, ConfigError
from .grid_state import GridStore
from .references import ReferenceStore, check_name
from .storage import CaptureStore
from .v4l2 import validate_control

log = logging.getLogger(__name__)

CONTROLS_FILE = "controls.json"  # set_control values shared with the live reader

# A tool result: JPEG bytes become image blocks, strings become text blocks.
Result = list[Union[bytes, str]]

DEFAULT_MAX_EDGE = 1024
REGION_LONG_EDGE = 768
CONTROL_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
CELL_RE = re.compile(r"^\s*([A-Za-z])\s*0*([1-9][0-9]{0,2})\s*$")
DISPLAY_LONG_EDGE = 1920  # images sent to the wall display
STEP_IMAGE_LONG_EDGE = 1024
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
        cams[name] = CameraConfig(name=name, device=f"mock:{name}", resolution=size)
    return Config(cameras=cams, live_camera="scope" if "scope" in cams else next(iter(cams)))


def _check_text(value: Any, name: str, limit: int, required: bool) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise BenchVisionError(f"{name} is required.")
        return None
    if not isinstance(value, str):
        raise BenchVisionError(f"{name} must be text.")
    if len(value) > limit:
        raise BenchVisionError(f"{name} is {len(value)} characters; keep it under {limit} so it reads on the wall.")
    return value.strip()


def _one_line(value: str | None, name: str) -> str | None:
    if value is not None and ("\n" in value or "\r" in value):
        raise BenchVisionError(f"{name} must be one line (what's wrong, what to do).")
    return value


def _num(v: float, pct: bool = False) -> str:
    """Fixed-point with enough decimals that small non-zero values never print as 0 or 1e-05."""
    if v == 0:
        return "0"
    if v < 1e-6:
        return "<0.000001"
    for places in (1, 2, 3, 4, 5, 6):
        if round(v, places) != 0:
            text = f"{v:.{places}f}"
            break
    if pct and v < 100 and float(text) >= 100:  # e.g. 99.96%: don't claim every pixel changed
        return ">99.9"
    return text


def _short(value: Any, limit: int = 60) -> str:
    """A sidecar value echoed to the client, bounded and printable (sidecars can be hand-edited)."""
    text = value if isinstance(value, str) and value.isprintable() else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
        from .camlock import CameraLock

        self.cameras = CameraManager(cams, backend, CameraLock(root / ".bench-vision"))
        # Kept out of captures/ (only images + sidecars there); mock and live never share grids.
        self.references = ReferenceStore(root / ("references-mock" if mock else "references"))
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
            # Every image-producing tool ends its reply with this, so show() can use the path directly.
            return f"image_path: {self._relpath(self.store.save(cam, jpeg, meta))} (full-resolution frame)"
        except OSError as e:
            log.error("could not save capture: %s", e)
            return (
                f"image_path: none (WARNING: this capture was not saved to {self._relpath(self.store.root)}/: "
                f"{e.strerror or e})"
            )

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
        result = self._region(cam, cfg, frame, rotation, img, box, (x, y, w, h), max_edge, "capture_region")
        result[1] += self._target_hint(cam, cfg, rotation, box, "here")
        return result

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
        result[1] += self._target_hint(cam, cfg, rotation, (cx, cy, cw, ch), name)
        if grid.get("full_res_size") and list(grid["full_res_size"]) != [full_w, full_h]:
            result[1] += (
                f" Note: the frame is now {full_w}x{full_h} but the grid was drawn on "
                f"{grid['full_res_size'][0]}x{grid['full_res_size'][1]}; cells were scaled proportionally."
            )
        return result

    # ------------------------------------------------------------ references

    def save_reference(self, cam: str, name: str, rotate: int = 0, max_edge: int = DEFAULT_MAX_EDGE) -> Result:
        self._ready()
        self.cameras.get(cam)
        name = check_name(name)
        max_edge = imaging.check_max_edge(max_edge)
        cfg, frame, rotation, img = self._frame(cam, rotate)
        full_h, full_w = img.shape[:2]
        stamp = self.clock().isoformat(timespec="seconds")
        ref_meta = {
            "camera": cam, "name": name, "saved_at": stamp, "device": cfg.device, "mock": self.mock,
            "rotation": rotation, "full_res_size": [full_w, full_h], "controls": frame.controls,
        }
        png, previous = self.references.save(cam, name, img, ref_meta)
        out = imaging.fit_long_edge(imaging.to_pil(img), max_edge)
        jpeg = imaging.encode_jpeg(out, self._ready().jpeg_quality)
        saved = self._save(cam, img, self._meta(cfg, frame, rotation, img, out, max_edge,
                                                tool="save_reference", crop_box=None, reference=name))
        if previous:
            replaced = f" Replaced the reference saved {_short(previous.get('saved_at', 'earlier'))}."
        elif previous is not None:
            replaced = " Replaced an existing reference (its old sidecar was missing or unreadable)."
        else:
            replaced = ""
        text = (
            f"{cam}: saved reference '{name}' ({full_w}x{full_h}, rotation {rotation}°) to "
            f"{self._relpath(png)}.{replaced} Later, compare(cam=\"{cam}\", name=\"{name}\") diffs a fresh "
            f"frame against it. Returned {out.size[0]}x{out.size[1]}. {saved}; reference_path: {self._relpath(png)}"
        )
        return [jpeg, text]

    def compare(self, cam: str, name: str, max_edge: int = DEFAULT_MAX_EDGE) -> Result:
        self._ready()
        cfg = self.cameras.get(cam)
        name = check_name(name)
        max_edge = imaging.check_max_edge(max_edge)
        ref, ref_meta = self.references.load(cam, name)
        ref_rotation = ref_meta.get("rotation", 0)
        try:
            ref_rotation = imaging.normalize_rotation(ref_rotation)
        except BenchVisionError:
            raise BenchVisionError(f"Reference '{name}' has an invalid rotation in its sidecar; save it again.") from None
        # Capture in the reference's orientation so the two frames line up pixel for pixel.
        cfg, frame, rotation, cur = self._frame(cam, (ref_rotation - cfg.default_rotation) % 360)
        if cur.shape[:2] != ref.shape[:2]:
            rh, rw = ref.shape[:2]
            ch, cw = cur.shape[:2]
            raise BenchVisionError(
                f"Reference '{name}' is {rw}x{rh} but '{cam}' now delivers {cw}x{ch} (resolution or rotation "
                f"changed). Save the reference again at the current settings."
            )
        heat_out, stats = imaging.diff_heatmap(ref, cur, max_edge)
        full_h, full_w = cur.shape[:2]
        pair = imaging.side_by_side(
            imaging.to_pil(ref), imaging.to_pil(cur),
            (f"REFERENCE '{name}'", "NOW"), max_edge,
        )
        q = self._ready().jpeg_quality
        pair_jpeg = imaging.encode_jpeg(pair, q)
        heat_jpeg = imaging.encode_jpeg(heat_out, max(q, 75), full_chroma=True)  # small warm marks must survive
        saved = self._save(cam, cur, self._meta(cfg, frame, rotation, cur, heat_out, max_edge, tool="compare",
                                                crop_box=None, reference=name,
                                                reference_saved_at=ref_meta.get("saved_at"),
                                                reference_sha256=ref_meta.get("pair_sha256"), diff=stats,
                                                returned_sizes={"side_by_side": list(pair.size),
                                                                "heatmap": list(heat_out.size)}))
        n_boxes, n_areas = len(stats["regions"]), stats["region_count"]
        plural = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"  # noqa: E731
        if not n_boxes:
            boxes = "no boxes"
        elif stats["areas_in_regions"] == n_areas:
            covered = {1: "the changed area", 2: "both changed areas"}.get(n_areas, f"all {n_areas} changed areas")
            boxes = f"{n_boxes} box{'es' if n_boxes != 1 else ''} covering {covered}"
        else:
            boxes = (
                f"boxes = the {n_boxes} largest of {stats['hint_count']} zoom regions, covering "
                f"{stats['areas_in_regions']} of {plural(n_areas, 'changed area')}"
            )
        extra_rotate = (rotation - cfg.default_rotation) % 360
        rot_arg = f", rotate={extra_rotate}" if extra_rotate else ""
        if stats["regions"]:
            regions = "; ".join(
                f"#{i} x={x}, y={y}, w={w}, h={h}" for i, (x, y, w, h) in enumerate(stats["regions"], 1)
            )
            more = "" if stats["areas_in_regions"] == n_areas else f" ({plural(n_areas, 'changed area')} in total)"
            where = (
                f" {'Changed region' if n_boxes == 1 else 'Largest changed regions'} in this {full_w}x{full_h} "
                "frame, outlined and numbered on the heatmap"
                f"{more}: {regions} (inspect with capture_region(cam=\"{cam}\", x, y, w, h{rot_arg}))."
            )
        else:
            where = " No changed regions above the noise threshold."
        shift = stats["brightness_shift"]
        if abs(shift) >= imaging.BRIGHTNESS_NOTE:
            where += f" Median brightness is {abs(shift)} levels {'higher' if shift > 0 else 'lower'} than the reference."
        where += (
            " Caveat: absdiff can't tell rework from a moved board, a focus change or an exposure change; "
            "if the heatmap lights up edges or bright areas all over, check those first."
        )
        if ref_meta.get("device") not in (None, cfg.device):
            where += (
                f" WARNING: this reference was saved from device {_short(ref_meta.get('device'))!r}, but "
                f"'{cam}' is now {cfg.device!r}; differences may just be a different camera."
            )
        text = (
            f"{cam}: compared now vs reference '{name}' saved {_short(ref_meta.get('saved_at', '?'))} "
            f"({full_w}x{full_h}, rotation {rotation}°, no alignment). "
            f"Image 1: reference | now side by side. Image 2: absdiff heatmap over the current frame "
            f"(blue-cyan = not counted: below {imaging.DIFF_THRESHOLD} levels, or isolated specks under "
            f"{imaging.MIN_CHANGED_PIXELS} px; yellow-red = counted changes; {boxes}; "
            f"{heat_out.size[0]}x{heat_out.size[1]}, full-res x = x * {full_w / heat_out.size[0]:.3f}, "
            f"y = y * {full_h / heat_out.size[1]:.3f}). Mean diff {_num(stats['mean_diff'])}/255; "
            f"{stats['changed_px']} px ({_num(stats['changed_pct'], pct=True)}%) changed by >= {imaging.DIFF_THRESHOLD} levels "
            f"after a light blur, counting groups of >= {imaging.MIN_CHANGED_PIXELS} px (isolated specks "
            f"ignored).{where} {saved}; reference_path: {self._relpath(self.references.path_of(cam, name))}"
        )
        return [pair_jpeg, heat_jpeg, text]

    # -------------------------------------------------------------- controls

    def set_control(self, cam: str, control: str, value: int) -> str:
        self._ready()
        cfg = self.cameras.get(cam)
        if not isinstance(control, str) or not CONTROL_NAME_RE.fullmatch(control):
            raise BenchVisionError(
                f"control must be a v4l2 control name like 'focus_absolute' or 'gain'; got {_short(control, 40)}."
            )
        if not isinstance(value, int) or isinstance(value, bool):
            raise BenchVisionError(f"value must be an integer; got {_short(value, 40)}.")
        controls = self.cameras.backend.list_ctrls(cfg)  # needs Linux/v4l2 for real cameras
        info = validate_control(controls, control, value, f"set_control on '{cam}'")
        if info.inactive:
            focus_auto = "focus_auto" if "focus_auto" in controls else "focus_automatic_continuous"
            hint = {
                "focus_absolute": f"set {focus_auto}=0 first",
                "exposure_time_absolute": "set auto_exposure=1 (manual) first",
                "exposure_absolute": "set exposure_auto=1 (manual) first",
            }.get(control, "turn off the matching automatic control first")
            raise BenchVisionError(f"{control} is inactive on '{cam}' right now; {hint}.")
        before = info.value
        after = self.cameras.backend.set_ctrl(cfg, control, value)
        overrides = self.cameras.overrides.setdefault(cam, {})
        overrides[control] = value
        dropped = ""
        # Turning an auto mode back on makes its manual value inactive; stop re-applying it.
        # (auto switch -> manual control it gates, values where the manual control is inactive);
        # older kernels use the names focus_auto / exposure_auto / exposure_absolute.
        dependent = {"focus_automatic_continuous": ("focus_absolute", lambda v: v == 1),
                     "focus_auto": ("focus_absolute", lambda v: v == 1),
                     "auto_exposure": ("exposure_time_absolute", lambda v: v not in (1, 2)),
                     "exposure_auto": ("exposure_absolute", lambda v: v not in (1, 2))}.get(control)
        if dependent and dependent[1](value) and overrides.pop(dependent[0], None) is not None:
            dropped = f" (stopped re-applying {dependent[0]}, which is inactive in this mode)"
        self._save_overrides()
        rng = f"{info.min}..{info.max}" if info.min is not None and info.max is not None else "?"
        readback = f", camera now reports {after}" if after is not None and after != value else ""
        return (
            f"{cam}: {control} set to {value} (was {before}; range {rng}{readback}){dropped}. It is re-applied every time "
            f"'{cam}' opens until the server restarts; to keep it, add \"{control}={value}\" to the "
            f"[cameras.{cam}] v4l2_controls list in config.toml."
        )

    def _save_overrides(self) -> None:
        """Share set_control values with the live view's reader (another process), so opening the camera
        for the live view doesn't reset them to the config values. Tagged with this pid: they last only
        as long as this server."""
        path = self._controls_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"pid": os.getpid(), "overrides": self.cameras.overrides}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            log.warning("could not save set_control values for the live view: %s", e)

    def _controls_file(self) -> Path:
        # a --mock server's invented controls must never reach the real cameras' live view
        return self.root / ".bench-vision" / (f"{CONTROLS_FILE[:-5]}-mock.json" if self.mock else CONTROLS_FILE)

    def load_shared_overrides(self) -> None:
        """In the live reader: use the set_control values of a running MCP server (see _save_overrides),
        replacing any loaded earlier (that server may have exited since)."""
        from .camlock import _pid_alive

        self.cameras.overrides = {}

        try:
            data = json.loads(self._controls_file().read_text(encoding="utf-8"))
            pid, overrides = int(data["pid"]), data["overrides"]
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            return
        if not 0 < pid < 2**31 or not _pid_alive(pid) or not isinstance(overrides, dict):
            return
        for cam, values in overrides.items():
            if isinstance(values, dict):
                self.cameras.overrides[cam] = {
                    k: v for k, v in values.items()
                    if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
                }

    # ------------------------------------------------------------- wall display

    def _image_file(self, image_path: Any, what: str = "image_path") -> Path:
        """Resolve a path from a tool reply (relative to the project root); must stay inside it."""
        if not isinstance(image_path, str) or not image_path.strip():
            raise BenchVisionError(f"{what} must be a path such as one returned in a capture reply's image_path.")
        p = Path(image_path.strip())
        p = (p if p.is_absolute() else self.root / p)
        try:
            resolved = p.resolve()
            inside = resolved.is_relative_to(self.root.resolve())
            exists = resolved.is_file()
        except (OSError, RuntimeError, ValueError):  # ValueError: e.g. a NUL byte in the path
            inside, exists = True, False  # unusable path: report it as not existing
        shown = _short(image_path, 80)
        if not inside:
            raise BenchVisionError(f"{what} {shown} is outside the bench-vision folder; use a path from a capture reply.")
        if not exists:
            raise BenchVisionError(f"{what} {shown} does not exist; use the image_path from a capture reply.")
        return resolved

    def _load_marked(self, image_path: Any, marks: Any, what: str = "image_path") -> tuple[Any, Path | None, Path]:
        """(PIL image with marks drawn, annotated copy path or None, source path)."""
        from . import marks as marks_mod

        src = self._image_file(image_path, what)
        bgr = load_image_strict(src)
        if bgr is None:
            raise BenchVisionError(f"{what} {_short(image_path, 80)} is not a readable JPEG/PNG image.")
        img = imaging.to_pil(bgr)
        checked = marks_mod.validate(marks, *img.size)
        if not checked:
            return img, None, src
        img = marks_mod.draw(img, checked)
        try:
            out = self._annotated_path(src)
            jpeg = imaging.encode_jpeg(img, ARCHIVE_JPEG_QUALITY)
            while True:  # exclusive create: two concurrent show() calls never share a file
                try:
                    with open(out, "xb") as f:
                        f.write(jpeg)
                    break
                except FileExistsError:
                    out = self._annotated_path(src)
        except OSError as e:
            log.error("could not save annotated copy: %s", e)
            out = None
        return img, out, src

    def _annotated_path(self, src: Path) -> Path:
        """<stem>.marked.jpg next to the original when it is a capture; _1, _2... if that exists.

        Originals elsewhere (mock/, references) are not written next to: their copy goes into
        today's captures/ folder so the repo and the reference store stay clean.
        """
        folder = src.parent
        if not src.is_relative_to(self.store.root.resolve()):
            folder = self.store.root / self.clock().strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
        n, out = 0, folder / f"{src.stem}.marked.jpg"
        while out.exists():
            n += 1
            out = folder / f"{src.stem}.marked_{n}.jpg"
        return out

    def _display_jpeg(self, img: Any, long_edge: int) -> str:
        from .display_client import b64

        return b64(imaging.encode_jpeg(imaging.fit_long_edge(img, long_edge), 85))

    def _push(self, payload: dict[str, Any]) -> None:
        from .display_client import push

        push(self._ready().display_url, payload)

    def show(self, image_path: str, caption: str, marks: Any = None) -> str:
        self._ready()
        caption = _one_line(_check_text(caption, "caption", 200, required=True), "caption")
        img, marked, src = self._load_marked(image_path, marks)
        saved = f" Annotated copy: {self._relpath(marked)}." if marked else ""
        try:
            self._push({"area": "image", "caption": caption,
                        "images": [{"jpeg_b64": self._display_jpeg(img, DISPLAY_LONG_EDGE), "label": None}]})
        except BenchVisionError as e:
            raise BenchVisionError(f"{e}{saved}") from None
        return f"Shown on the wall display: {self._relpath(src)} — {caption.rstrip('.')}.{saved}"

    def show_compare(self, left_path: str, right_path: str, caption: str, left_label: str, right_label: str) -> str:
        self._ready()
        caption = _one_line(_check_text(caption, "caption", 200, required=True), "caption")
        labels = [_check_text(left_label, "left_label", 60, required=True),
                  _check_text(right_label, "right_label", 60, required=True)]
        images = [self._load_marked(p, None, what)[0] for p, what in ((left_path, "left_path"), (right_path, "right_path"))]
        self._push({"area": "image", "caption": caption,
                    "images": [{"jpeg_b64": self._display_jpeg(im, DISPLAY_LONG_EDGE), "label": lab}
                               for im, lab in zip(images, labels)]})
        return f"Shown side by side on the wall display: {labels[0]} | {labels[1]} — {caption.rstrip('.')}."

    def show_step(self, title: str, body: str, image_path: str | None = None, marks: Any = None,
                  progress: str | None = None) -> str:
        self._ready()
        title = _one_line(_check_text(title, "title", 80, required=True), "title")
        body = _check_text(body, "body", 500, required=True)
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) > 5:
            raise BenchVisionError(f"body has {len(lines)} lines; keep a step to 3-5 short lines so it reads from 3 ft.")
        progress = _check_text(progress, "progress", 60, required=False)
        if image_path is None and marks:
            raise BenchVisionError("marks need an image_path to draw on.")
        payload: dict[str, Any] = {"area": "step", "title": title, "body": body, "progress": progress}
        saved = ""
        if image_path is not None:
            img, marked, _ = self._load_marked(image_path, marks)
            payload["image_jpeg_b64"] = self._display_jpeg(img, STEP_IMAGE_LONG_EDGE)
            saved = f" Annotated copy: {self._relpath(marked)}." if marked else ""
        try:
            self._push(payload)
        except BenchVisionError as e:
            raise BenchVisionError(f"{e}{saved}") from None
        return f"Step shown on the wall display: {title}{f' ({progress})' if progress else ''}.{saved}"

    def show_clear(self, area: str = "all") -> str:
        self._ready()
        if area not in ("all", "image", "step"):
            raise BenchVisionError(f"area must be 'all', 'image' or 'step'; got {_short(area, 20)}.")
        self._push({"area": "clear", "which": area})
        return f"Cleared the wall display ({area})."

    # ------------------------------------------------------- live view target

    def _live_frame_size(self, cfg: CameraConfig) -> tuple[int, int]:
        """Full-res size of a camera's frame at its default rotation (what the live view shows)."""
        w, h = self.cameras.last_size.get(cfg.name, cfg.resolution)
        return (h, w) if cfg.default_rotation in (90, 270) else (w, h)

    def _target_hint(self, cam: str, cfg: CameraConfig, rotation: int, box: tuple[int, int, int, int],
                     label: str) -> str:
        if rotation != cfg.default_rotation:
            return ""  # the live view shows the default rotation; these coordinates wouldn't match
        x, y, w, h = box
        return (f"\nTo mark it on the wall's live view: set_target(cam=\"{cam}\", x={x}, y={y}, w={w}, h={h}, "
                f"label=\"{label}\").")

    def set_target(self, cam: str, x: int, y: int, w: int, h: int, label: str = "") -> str:
        cfg_all = self._ready()
        cfg = self.cameras.get(cam)
        label = _check_text(label, "label", 60, required=False) or ""
        if "\n" in label or "\r" in label:
            raise BenchVisionError("label must be one line (e.g. \"J5\").")
        fw, fh = self._live_frame_size(cfg)
        box = imaging.clamp_box(x, y, w, h, fw, fh)
        self._push({"area": "target", "cam": cam, "x": box[0], "y": box[1], "w": box[2], "h": box[3],
                    "label": label, "frame_w": fw, "frame_h": fh})
        clipped = "" if box == (x, y, w, h) else f" (clipped to the {fw}x{fh} frame)"
        where = "" if cam == cfg_all.live_camera else (
            f" The wall's live view shows '{cfg_all.live_camera}', so this reticle appears only if that is "
            f"changed to '{cam}' ([display] live_camera)."
        )
        return f"Reticle set on '{cam}' at x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}{clipped}.{where}"

    def clear_target(self, cam: str) -> str:
        self._ready()
        self.cameras.get(cam)
        self._push({"area": "target", "cam": cam, "clear": True})
        return f"Reticle cleared on '{cam}'."

    # ------------------------------------------------------------- boards

    def _boards(self) -> Any:
        from .boards import BoardStore

        return BoardStore(self.root / "boards")

    def _push_board(self) -> str:
        """Tell the display the board changed; a display that isn't running is reported, not fatal."""
        try:
            self._push({"area": "board"})
            return ""
        except BenchVisionError as e:
            return f" (Saved; the wall display wasn't updated: {e})"

    def board_init(self, name: str, image_path: str, joints: Any) -> str:
        from . import boards

        self._ready()
        name = boards.check_name(name)
        src = self._image_file(image_path)
        bgr = load_image_strict(src)
        if bgr is None:
            raise BenchVisionError(f"image_path {_short(image_path, 80)} is not a readable JPEG/PNG image.")
        H, W = bgr.shape[:2]
        if not isinstance(joints, list) or not joints:
            raise BenchVisionError("joints must be a non-empty list of {id, x, y, w, h} in the image's pixels.")
        if len(joints) > boards.MAX_JOINTS:
            raise BenchVisionError(f"At most {boards.MAX_JOINTS} joints per board; got {len(joints)}.")
        clean, seen = [], set()
        for i, j in enumerate(joints):
            where = f"joints[{i}]"
            if not isinstance(j, dict) or set(j) != {"id", "x", "y", "w", "h"}:
                raise BenchVisionError(f"{where} must be an object with exactly id, x, y, w, h.")
            jid = j.get("id")
            if not isinstance(jid, str) or not boards.JOINT_RE.fullmatch(jid):
                raise BenchVisionError(f"{where}: id must be 1-24 letters/digits (e.g. \"J5\"); got {repr(jid)[:30]}.")
            if jid in seen:
                raise BenchVisionError(f"{where}: duplicate joint id {jid!r}.")
            seen.add(jid)
            nums = {}
            for k in ("x", "y", "w", "h"):
                v = j.get(k)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) > 1e6:
                    raise BenchVisionError(f"{where} ({jid}): {k} must be a number of pixels.")
                nums[k] = round(float(v), 1)
            if nums["w"] <= 0 or nums["h"] <= 0 or nums["x"] < 0 or nums["y"] < 0 or \
                    nums["x"] + nums["w"] > W or nums["y"] + nums["h"] > H:
                raise BenchVisionError(
                    f"{where} ({jid}) is not inside the {W}x{H} image (x, y = top-left; w, h > 0)."
                )
            clean.append({"id": jid, **nums})
        camera = self._source_camera(src)
        jpeg = imaging.encode_jpeg(imaging.to_pil(bgr), ARCHIVE_JPEG_QUALITY)
        self._boards().init(name, jpeg, (W, H), clean, self._relpath(src), camera)
        note = self._push_board()
        cam_note = f" Reference came from '{camera}'." if camera else ""
        return (f"Board '{name}' recorded: {len(clean)} joints on a {W}x{H} reference "
                f"(boards/{name}.json), all todo; it is now the current board.{cam_note}{note}")

    def _source_camera(self, src: Path) -> str | None:
        """The camera a capture came from, if its sidecar says so and it used the default rotation."""
        try:
            meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
            cam, rotation = meta.get("camera"), meta.get("rotation")
            cfg = self.cameras.cameras.get(cam) if isinstance(cam, str) else None
            if cfg is not None and rotation == cfg.default_rotation and meta.get("crop_box") in (None, []):
                return cam
        except (OSError, ValueError, AttributeError, RecursionError):
            pass
        return None

    def board_set(self, name: str, joint_id: str, state: str) -> str:
        from . import boards

        self._ready()
        name = boards.check_name(name)
        if state not in boards.STATES:
            raise BenchVisionError(f"state must be one of {', '.join(boards.STATES)}; got {_short(state, 20)}.")
        if not isinstance(joint_id, str) or not joint_id:
            raise BenchVisionError("joint_id must be a joint id from board_init, e.g. \"J5\".")
        board, demoted = self._boards().set_state(name, joint_id, state)
        note = self._push_board()
        extra = f" {demoted} went back to todo (one active joint at a time)." if demoted else ""
        hint = ""
        if state == "active":
            j = next(j for j in board["joints"] if j.get("id") == joint_id)
            box = tuple(round(j[k]) for k in ("x", "y", "w", "h"))
            cam = board.get("camera")
            if isinstance(cam, str) and cam in self.cameras.cameras:
                hint = (f" Now mark it: set_target(cam=\"{cam}\", x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}, "
                        f"label=\"{joint_id}\").")
            else:
                hint = (f" Its box in the reference image is x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}; use "
                        "set_target with those if the reference came from that camera's full frame.")
        counts = {s: sum(1 for j in board["joints"] if j.get("state") == s) for s in boards.STATES}
        return (f"Board '{name}': {joint_id} is {state}.{extra} "
                f"({counts['verified']} verified, {counts['flagged']} flagged, {counts['todo']} todo.){hint}{note}")

    def record_verdict(self, joint_id: str, verdict: str, image_path: str, note: str = "") -> str:
        from . import boards

        self._ready()
        store = self._boards()
        name = store.current_name()
        if name is None:
            raise BenchVisionError("No current board: call board_init first.")
        board = store.load(name)
        joint = next((j for j in board["joints"] if isinstance(j, dict) and j.get("id") == joint_id), None)
        if not isinstance(joint_id, str) or joint is None:
            raise BenchVisionError(f"Board '{name}' has no joint {_short(joint_id, 30)}.")
        v = verdict.strip().lower() if isinstance(verdict, str) else None
        if v not in boards.VERDICTS:
            raise BenchVisionError(f"verdict must be one of {', '.join(boards.VERDICTS)}; got {_short(verdict, 30)}.")
        note = _one_line(_check_text(note, "note", 200, required=False), "note") or ""
        src = self._image_file(image_path)
        bgr = load_image_strict(src)
        if bgr is None:
            raise BenchVisionError(f"image_path {_short(image_path, 80)} is not a readable JPEG/PNG image.")
        img = imaging.to_pil(bgr)
        if list(img.size) == list(board.get("size", [])):
            # same frame as the board reference: crop the joint, with context around it
            x, y, w, h = (float(joint[k]) for k in ("x", "y", "w", "h"))
            m = max(w, h) * 0.6 + 16
            img = img.crop((max(0, x - m), max(0, y - m), min(img.size[0], x + w + m), min(img.size[1], y + h + m)))
        thumb = imaging.encode_jpeg(imaging.fit_long_edge(img, 240), 85)
        entry = store.add_verdict(name, {"joint": joint_id, "verdict": v, "note": note,
                                         "image_path": self._relpath(src),
                                         "time": self.clock().isoformat(timespec="seconds")}, thumb)
        shown = self._push_board()
        return f"Recorded on '{name}': {joint_id} = {v}{f' ({note})' if note else ''} [{entry['thumb']}].{shown}"
