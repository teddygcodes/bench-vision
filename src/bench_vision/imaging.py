"""Image helpers: rotation, resizing, JPEG encoding."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

from .errors import BenchVisionError

ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def normalize_rotation(degrees: int) -> int:
    if not isinstance(degrees, int) or isinstance(degrees, bool) or degrees % 90 != 0:
        raise BenchVisionError(f"rotate must be a multiple of 90 degrees (clockwise), got {degrees!r}.")
    return degrees % 360


def rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    degrees = normalize_rotation(degrees)
    return img if degrees == 0 else cv2.rotate(img, ROTATIONS[degrees])


def to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def fit_long_edge(img: Image.Image, long_edge: int, upscale: bool = False) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale >= 1 and not upscale:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def check_max_edge(max_edge: int, default_cap: int = 4096) -> int:
    if not isinstance(max_edge, int) or isinstance(max_edge, bool) or not 64 <= max_edge <= default_cap:
        raise BenchVisionError(f"max_edge must be between 64 and {default_cap} pixels, got {max_edge!r}.")
    return max_edge


MIN_REGION = 8  # smallest crop edge, in full-res pixels


MAX_COORD = 100_000


def validate_box(x: int, y: int, w: int, h: int) -> None:
    for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise BenchVisionError(f"{name} must be an integer number of pixels, got {type(v).__name__}.")
        if abs(v) > MAX_COORD:
            raise BenchVisionError(f"{name} is out of range (|{name}| must be <= {MAX_COORD} pixels).")
    if w <= 0 or h <= 0:
        raise BenchVisionError(f"Region width and height must be positive, got w={w}, h={h}.")


def clamp_box(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Intersect a crop box with the frame; raise a clear error if nothing usable is left."""
    validate_box(x, y, w, h)
    if w < MIN_REGION or h < MIN_REGION:
        raise BenchVisionError(f"Region must be at least {MIN_REGION}x{MIN_REGION} px, got w={w}, h={h}.")
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame_w), min(y + h, frame_h)
    if x1 - x0 < MIN_REGION or y1 - y0 < MIN_REGION:
        raise BenchVisionError(
            f"Region x={x}, y={y}, w={w}, h={h} lies (almost) entirely outside the {frame_w}x{frame_h} frame "
            f"(at least {MIN_REGION}x{MIN_REGION} px must overlap). Coordinates are full-resolution pixels "
            "of the rotated frame; x/y is the top-left corner."
        )
    return x0, y0, x1 - x0, y1 - y0


GRID_MAX_ROWS = 26  # rows are lettered A..Z
GRID_MAX_COLS = 50
MIN_OVERLAY_CELL = 12  # px per cell in the returned overlay image


def check_grid(rows: int, cols: int) -> None:
    for name, v, hi in (("rows", rows, GRID_MAX_ROWS), ("cols", cols, GRID_MAX_COLS)):
        if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= hi:
            raise BenchVisionError(f"{name} must be an integer 1-{hi}, got {v!r}.")


def cell_name(row: int, col: int) -> str:
    return f"{chr(ord('A') + row)}{col + 1}"


def cell_box(row: int, col: int, rows: int, cols: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Full-res (x, y, w, h) of a grid cell; edges are rounded so cells tile the frame exactly."""
    x0, x1 = round(col * frame_w / cols), round((col + 1) * frame_w / cols)
    y0, y1 = round(row * frame_h / rows), round((row + 1) * frame_h / rows)
    return x0, y0, x1 - x0, y1 - y0


def draw_grid(img: Image.Image, rows: int, cols: int) -> Image.Image:
    """Overlay grid lines and cell labels (A1 = top-left) on an already-downscaled image."""
    from PIL import ImageDraw, ImageFont

    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    line = max(1, round(max(w, h) / 600))
    # Dark outline keeps lines visible on bright solder; dropped on dense grids so it doesn't hide the board.
    outline = line + 2 if min(w / cols, h / rows) >= 40 else 0
    for c in range(1, cols):
        x = round(c * w / cols)
        if outline:
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0), width=outline)
        draw.line([(x, 0), (x, h)], fill=(255, 230, 0), width=line)
    for r in range(1, rows):
        y = round(r * h / rows)
        if outline:
            draw.line([(0, y), (w, y)], fill=(0, 0, 0), width=outline)
        draw.line([(0, y), (w, y)], fill=(255, 230, 0), width=line)

    cw, ch = w / cols, h / rows
    size = int(max(9, min(28, ch * 0.28, cw * 0.22)))
    font = ImageFont.load_default(size=size)
    pad = max(1, size // 6)
    widest = draw.textbbox((0, 0), cell_name(rows - 1, cols - 1), font=font)
    label_w = widest[2] - widest[0] + 2 * pad + line + 1
    label_h = widest[3] - widest[1] + 2 * pad + line + 1
    # Label every cell only if a label leaves most of the cell visible.
    every_cell = label_w <= cw * 0.6 and label_h <= ch * 0.4

    def label(text: str, x: float, y: float) -> None:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        box = (x, y, x + right - left + 2 * pad, y + bottom - top + 2 * pad)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x + pad - left, y + pad - top), text, fill=(255, 230, 0), font=font)

    if every_cell:
        for r in range(rows):
            for c in range(cols):
                label(cell_name(r, c), c * cw + line + 1, r * ch + line + 1)
    else:
        # Too dense: label the top row and left column, skipping cells so labels never overlap.
        col_step = max(1, math.ceil(label_w / cw))
        row_step = max(1, math.ceil(label_h / ch))
        for c in range(0, cols, col_step):
            label(cell_name(0, c), c * cw + line + 1, line + 1)
        for r in range(row_step, rows, row_step):
            label(cell_name(r, 0), line + 1, r * ch + line + 1)
    return out
