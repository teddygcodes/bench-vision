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


def encode_jpeg(img: Image.Image, quality: int = 85, full_chroma: bool = False) -> bytes:
    extra = {"subsampling": 0} if full_chroma else {}  # keep thin coloured detail (heatmaps)
    rgb = img.convert("RGB")
    try:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, optimize=True, **extra)
    except OSError:
        # Pillow's optimize pass uses a fixed-size buffer that very noisy images can overflow.
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, **extra)
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


DIFF_THRESHOLD = 30  # per-pixel absdiff (0-255, max over channels) counted as "changed"
MIN_CHANGED_PIXELS = 4  # a region needs at least this many changed pixels (after the 5x5 blur)


def _tight(labels: np.ndarray, mask: np.ndarray, label: int, box: tuple) -> tuple[int, int, int, int, int]:
    """Bounding box of the changed (undilated) pixels of one grouped component."""
    x, y, w, h, count = box
    sub = (labels[y : y + h, x : x + w] == label) & (mask[y : y + h, x : x + w] > 0)
    ys, xs = np.nonzero(sub)
    return x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), count


BRIGHTNESS_NOTE = 8  # median luma change (levels) worth mentioning


def _merge_boxes(boxes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Merge (x, y, w, h, changed_px, areas) boxes that overlap or touch, until stable."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        out: list[tuple[int, ...]] = []
        for b in boxes:
            for i, o in enumerate(out):
                if b[0] <= o[0] + o[2] and o[0] <= b[0] + b[2] and b[1] <= o[1] + o[3] and o[1] <= b[1] + b[3]:
                    x0, y0 = min(b[0], o[0]), min(b[1], o[1])
                    x1, y1 = max(b[0] + b[2], o[0] + o[2]), max(b[1] + b[3], o[1] + o[3])
                    out[i] = (x0, y0, x1 - x0, y1 - y0, b[4] + o[4], b[5] + o[5])
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def diff_heatmap(ref: np.ndarray, cur: np.ndarray, max_edge: int, top: int = 3) -> tuple[Image.Image, dict]:
    """Compare two same-size BGR frames (no alignment: the cameras are fixed).

    Returns (heatmap image, stats). The heatmap is the absdiff, max-pooled so even
    1-2 px changes stay visible after downscaling to max_edge, blended over the dimmed
    current frame, with the reported regions outlined and numbered. stats has
    mean_diff, changed_pct, brightness_shift (median luma change, signed), regions
    (padded full-res [x, y, w, h], most changed first, ready for capture_region) and
    region_count. It reports what differs; it does not guess why.
    """
    a = cv2.GaussianBlur(ref, (5, 5), 0)
    b = cv2.GaussianBlur(cur, (5, 5), 0)
    diff = cv2.absdiff(a, b).max(axis=2)
    mask = (diff >= DIFF_THRESHOLD).astype(np.uint8)
    h_img, w_img = diff.shape

    # Changed-pixel groups: a light dilation joins fragments of one thin line; groups with
    # fewer than MIN_CHANGED_PIXELS changed pixels are speckle. Padded boxes that overlap or
    # touch are then merged, so one feature never yields several overlapping hints.
    k = max(5, (min(h_img, w_img) // 150) | 1)
    n, labels, comp, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((k, k), np.uint8)), connectivity=8)
    counts = np.bincount(labels[mask > 0], minlength=n)
    keep = [i for i in range(1, n) if counts[i] >= MIN_CHANGED_PIXELS]
    real = np.isin(labels, keep) & (mask > 0) if keep else np.zeros(mask.shape, bool)
    tight = [_tight(labels, mask, i, (*map(int, comp[i, :4]), int(counts[i]))) for i in keep]
    boxes = _merge_boxes([(*pad_box(t[:4], w_img, h_img), t[4], 1) for t in tight])
    boxes.sort(key=lambda r: r[4], reverse=True)

    step = 4  # subsample: the median of a quarter-million pixels is plenty
    luma_ref = cv2.cvtColor(ref[::step, ::step], cv2.COLOR_BGR2GRAY).astype(np.int16)
    luma_cur = cv2.cvtColor(cur[::step, ::step], cv2.COLOR_BGR2GRAY).astype(np.int16)
    brightness_shift = int(np.median(luma_cur - luma_ref))  # luma, signed: + brighter, - darker
    changed_px = int(real.sum())
    changed_pct = 100.0 * changed_px / real.size
    regions = [list(map(int, bx[:4])) for bx in boxes[:top]]

    # Display layer at the output size.
    out_w, out_h = fit_long_edge(Image.new("L", (w_img, h_img)), max_edge).size
    pool = max(1, math.ceil(w_img / out_w))
    shown = cv2.resize(cv2.dilate(diff, np.ones((pool, pool), np.uint8)), (out_w, out_h), interpolation=cv2.INTER_AREA)
    counted = cv2.resize(cv2.dilate(real.astype(np.uint8), np.ones((pool, pool), np.uint8)), (out_w, out_h),
                         interpolation=cv2.INTER_NEAREST)
    counted = cv2.dilate(counted, np.ones((3, 3), np.uint8)) > 0  # every counted group >= 3x3 px on screen
    # Two-part colour scale: below the threshold blue -> cyan (seen, not counted); at or above it
    # yellow -> orange-red (counted). JET's top end is dark red, so stop at orange-red (205).
    d = shown.astype(np.float32)
    # Warm colours only where pixels were actually counted, so the map always matches the numbers.
    level = np.where(
        counted,
        160.0 + (np.clip(d, DIFF_THRESHOLD, 128.0) - DIFF_THRESHOLD) * (45.0 / (128.0 - DIFF_THRESHOLD)),
        # 4 coarse steps: enough to show where small differences are, without colour noise bloating the JPEG
        np.floor(np.minimum(d, DIFF_THRESHOLD - 1) * (4.0 / DIFF_THRESHOLD)) * 22.5,
    )
    heat = cv2.applyColorMap(level.astype(np.uint8), cv2.COLORMAP_JET)
    gray = cv2.cvtColor(cv2.resize(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), (out_w, out_h), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(heat, 0.7, gray, 0.3, 0)
    sx, sy = out_w / w_img, out_h / h_img
    for idx, (x, y, w, h) in enumerate(regions, 1):
        # outline just outside the padded box so it never covers a change at the frame edge
        p0 = [int(x * sx) - 3, int(y * sy) - 3]
        p1 = [int((x + w) * sx) + 2, int((y + h) * sy) + 2]
        # a box spanning the whole width/height would be outlined off-image: pull that side in
        if w >= w_img:
            p0[0], p1[0] = 0, out_w - 1
        if h >= h_img:
            p0[1], p1[1] = 0, out_h - 1
        p0, p1 = tuple(p0), tuple(p1)
        cv2.rectangle(blended, p0, p1, (0, 0, 0), 3)
        cv2.rectangle(blended, p0, p1, (255, 255, 255), 1)
        # number above the box, or below it when the box touches the top edge (never on the change)
        org = (max(p0[0], 0) + 3, p0[1] - 4 if p0[1] - 4 >= 12 else min(p1[1] + 14, out_h - 2))
        cv2.putText(blended, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    stats = {
        "mean_diff": float(diff.mean()),
        "changed_px": changed_px,
        "changed_pct": changed_pct,
        "brightness_shift": brightness_shift,
        "regions": regions,
        "region_count": len(tight),  # separate groups of changed pixels, before merging hints
        "hint_count": len(boxes),  # zoom boxes after merging (regions holds the top `top` of them)
        "areas_in_regions": sum(bx[5] for bx in boxes[:top]),
    }
    return to_pil(blended), stats


def pad_box(box: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Grow a changed-region box with context (25% + 16 px a side, at most 64 px a side; boxes at
    least 64 px) and clamp it to the frame. The cap keeps big changes from swallowing their neighbours."""
    x, y, w, h = box

    def grow(start: int, size: int, limit: int) -> tuple[int, int]:
        new = max(size + 2 * min(size // 4 + 16, 64), 64)
        new = min(new, limit)
        s = min(max(start - (new - size) // 2, 0), limit - new)
        return s, new

    x, w = grow(x, w, frame_w)
    y, h = grow(y, h, frame_h)
    return x, y, w, h


def side_by_side(left: Image.Image, right: Image.Image, labels: tuple[str, str], max_edge: int) -> Image.Image:
    """Two same-size images next to each other with a label strip, long edge <= max_edge."""
    from PIL import ImageDraw, ImageFont

    w, h = left.size
    gap = max(4, w // 200)
    font_px = max(12, h // 25)
    strip = font_px + 2 * (font_px // 3)
    canvas = Image.new("RGB", (2 * w + gap, h + strip), (24, 24, 24))
    canvas.paste(left.convert("RGB"), (0, strip))
    canvas.paste(right.convert("RGB"), (w + gap, strip))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_px)
    room = w - font_px  # each label must stay within its own half
    for x, text in ((0, labels[0]), (w + gap, labels[1])):
        while text and draw.textlength(text, font=font) > room:
            text = text[:-2] + "…" if len(text) > 2 else ""
        draw.text((x + font_px // 2, font_px // 3), text, fill=(255, 230, 0), font=font)
    return fit_long_edge(canvas, max_edge)
