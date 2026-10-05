"""Annotation marks for show()/show_step(), drawn at full resolution.

A mark is {kind, x, y, w, h, text, color} in the image's full-resolution pixels:
  circle  around the box (x, y, w, h); without w/h, centred on (x, y)
  box     the rectangle (x, y, w, h)
  arrow   pointing AT (x, y); w/h (optional) is the offset back to the tail, default from the upper left
  label   text with a dark background at (x, y)
`text` (optional for circle/box/arrow, required for label) is drawn next to the mark where it covers
nothing; otherwise the mark gets a numbered badge and the text goes in a legend below the image.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .errors import BenchVisionError

KINDS = ("circle", "arrow", "box", "label")
MAX_MARKS = 20
COLORS = {
    "red": (255, 48, 48), "yellow": (255, 212, 0), "green": (0, 220, 90), "cyan": (0, 220, 255),
    "magenta": (255, 60, 220), "white": (255, 255, 255), "orange": (255, 140, 0), "blue": (60, 140, 255),
}
DEFAULT_COLOR = "red"


def _color(value: Any, where: str) -> tuple[int, int, int]:
    if value is None:
        return COLORS[DEFAULT_COLOR]
    if isinstance(value, str):
        v = value.strip().lower()
        if v in COLORS:
            return COLORS[v]
        if len(v) == 7 and v.startswith("#"):
            try:
                return int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)
            except ValueError:
                pass
    raise BenchVisionError(f"{where}: color must be one of {', '.join(COLORS)} or #rrggbb; got {str(value)[:20]!r}.")


def _num(mark: dict, key: str, where: str, required: bool) -> float | None:
    v = mark.get(key)
    if v is None:
        if required:
            raise BenchVisionError(f"{where}: '{key}' is required.")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise BenchVisionError(f"{where}: '{key}' must be a number of pixels; got {str(v)[:20]!r}.")
    return float(v)


def validate(marks: Any, width: int, height: int) -> list[dict[str, Any]]:
    """Check marks against the image size; returns normalised copies."""
    if marks is None:
        return []
    if not isinstance(marks, list):
        raise BenchVisionError("marks must be a list of {kind, x, y, w, h, text, color} objects.")
    if len(marks) > MAX_MARKS:
        raise BenchVisionError(f"At most {MAX_MARKS} marks per image; got {len(marks)}.")
    out = []
    for i, m in enumerate(marks):
        where = f"marks[{i}]"
        if not isinstance(m, dict):
            raise BenchVisionError(f"{where} must be an object like {{\"kind\": \"circle\", \"x\": 10, \"y\": 20}}.")
        unknown = set(m) - {"kind", "x", "y", "w", "h", "text", "color"}
        if unknown:
            raise BenchVisionError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}.")
        kind = m.get("kind")
        if kind not in KINDS:
            raise BenchVisionError(f"{where}: kind must be one of {', '.join(KINDS)}; got {str(kind)[:20]!r}.")
        x, y = _num(m, "x", where, True), _num(m, "y", where, True)
        w, h = _num(m, "w", where, False), _num(m, "h", where, False)
        if kind == "box" and (w is None or h is None):
            raise BenchVisionError(f"{where}: a box needs w and h.")
        if (kind in ("circle", "box")) and ((w is not None and w <= 0) or (h is not None and h <= 0)):
            raise BenchVisionError(f"{where}: w and h must be positive.")
        limit = 2 * max(width, height)
        if any(v is not None and abs(v) > limit for v in (w, h)):
            raise BenchVisionError(f"{where}: w and h must be within {limit} px for this {width}x{height} image.")
        if not (0 <= x <= width and 0 <= y <= height):
            raise BenchVisionError(
                f"{where}: ({x:g}, {y:g}) is outside the {width}x{height} image. Marks use that image's "
                "full-resolution pixel coordinates (the frame size every capture reply states)."
            )
        text = m.get("text")
        if text is not None and (not isinstance(text, str) or len(text) > 80):
            raise BenchVisionError(f"{where}: text must be a string of at most 80 characters.")
        if isinstance(text, str) and ("\n" in text or "\r" in text):
            raise BenchVisionError(f"{where}: text must be one line.")
        if kind == "label" and not text:
            raise BenchVisionError(f"{where}: a label needs text.")
        out.append({"kind": kind, "x": x, "y": y, "w": w, "h": h, "text": text or None,
                    "color": _color(m.get("color"), where)})
    return out


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


def _overlap(a: Rect, b: Rect) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def draw(img: Image.Image, marks: list[dict[str, Any]]) -> Image.Image:
    """Return a copy of img (RGB) with the validated marks drawn, sized to stay legible when scaled down.

    All shapes are drawn first. Each label then goes next to its mark only where it covers no mark,
    no arrow (target or shaft) and no other label; if there is no such spot nearby, the mark gets a
    small numbered badge and its text goes in a legend strip appended below the image. The image
    area itself is never resized, so mark coordinates stay valid in the annotated copy.
    """
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    W, H = out.size
    long_edge = max(W, H)
    line = max(3, round(long_edge / 250))  # ~8 px at 1920 wide: still a few px on a scaled wall image
    font = _font(max(16, round(long_edge / 24)))
    badge_font = _font(max(12, round(long_edge / 40)))
    pad = max(4, line)
    gap = line + 2

    keep_clear: list[Rect] = []  # every mark's area (and arrow targets): labels must not cover these
    pending: list[tuple[str, tuple[int, int, int], Rect]] = []  # (text, color, area of its own mark)

    for m in marks:
        x, y, w, h, c = m["x"], m["y"], m["w"], m["h"], m["color"]
        if m["kind"] == "box":
            d.rectangle((x, y, x + w, y + h), outline=(0, 0, 0), width=line + 4)
            d.rectangle((x, y, x + w, y + h), outline=c, width=line)
            area: Rect = (x - line, y - line, x + w + line, y + h + line)
            keep_clear.append(area)
        elif m["kind"] == "circle":
            if w is not None or h is not None:
                cx, cy = x + (w or h) / 2, y + (h or w) / 2
                r = max(w or h, h or w) / 2 * 1.15 + line
            else:
                cx, cy, r = x, y, long_edge / 20
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=line + 4)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=c, width=line)
            area = (cx - r - line, cy - r - line, cx + r + line, cy + r + line)
            keep_clear.append(area)
        elif m["kind"] == "arrow":
            head = max(line * 4, long_edge / 50)
            dx = dy = None
            if w is not None and h is not None and math.hypot(w, h) >= 2 * head:
                # keep the tail on the image: shorten the arrow along its own direction if needed
                scale = 1.0
                for v, d_, lim in ((x, w, W), (y, h, H)):
                    t = v - d_  # tail coordinate
                    if t < 0:
                        scale = min(scale, v / d_ if d_ else 1.0)
                    elif t > lim:
                        scale = min(scale, (v - lim) / d_ if d_ else 1.0)
                if math.hypot(w * scale, h * scale) >= 2 * head:
                    dx, dy = w * scale, h * scale
            if dx is None:  # default (or too short to see): come in from the upper left, flipped if that leaves the image
                sx_, sy_ = min(long_edge / 10, W / 3), min(long_edge / 10, H / 3)  # stay on narrow images
                dx = sx_ if x - sx_ >= 0 else -sx_
                dy = sy_ if y - sy_ >= 0 else -sy_
            tail = (x - dx, y - dy)
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            head = max(head, min(length / 5, long_edge / 25))
            left_pt = (x - head * ux + head * 0.5 * uy, y - head * uy - head * 0.5 * ux)
            right_pt = (x - head * ux - head * 0.5 * uy, y - head * uy + head * 0.5 * ux)
            d.line((tail, (x, y)), fill=(0, 0, 0), width=line + 4)
            d.line((tail, (x - head * 0.6 * ux, y - head * 0.6 * uy)), fill=c, width=line)
            d.polygon(((x, y), left_pt, right_pt), fill=c, outline=(0, 0, 0))
            target: Rect = (x - head - gap, y - head - gap, x + head + gap, y + head + gap)
            keep_clear.append(target)  # the head and the point it shows
            seg = max(line * 3, 1.0)  # and the shaft, as a chain of small squares
            for i in range(int(length // seg) + 1):
                px, py = tail[0] + ux * i * seg, tail[1] + uy * i * seg
                keep_clear.append((px - seg, py - seg, px + seg, py + seg))
            # the label belongs at the tail end
            area = (tail[0] - gap, tail[1] - gap, tail[0] + gap, tail[1] + gap)
        else:  # label: text at (x, y)
            area = (x, y, x, y)
        if m["text"]:
            pending.append((m["text"], c, area))

    placed: list[Rect] = []
    legend: list[tuple[str, str, tuple[int, int, int]]] = []  # (badge number, text, color)

    def spots(own: Rect, lw: float, lh: float, rings: int = 3) -> list[tuple[float, float]]:
        """Candidate top-left corners around a mark, nearest first (`rings` rings x 8 positions)."""
        x0, y0, x1, y1 = own
        out = [(x0, y0)] if (x0 == x1 and y0 == y1) else []
        for k in range(1, rings + 1):
            g = gap + (k - 1) * lh
            out += [(x0, y1 + g), (x0, y0 - g - lh), (x1 + g, y0), (x0 - g - lw, y0),
                    (x1 - lw, y1 + g), (x1 - lw, y0 - g - lh), (x1 + g, y1 - lh), (x0 - g - lw, y1 - lh)]
        return out

    def covered(r: Rect) -> float:
        return sum(_overlap(r, k) for k in keep_clear) + sum(_overlap(r, q) for q in placed)

    def on_image(x: float, y: float, w: float, h: float) -> bool:
        return x >= 0 and y >= 0 and x + w <= W and y + h <= H

    def boxed_text(x: float, y: float, text: str, color: tuple[int, int, int], f: Any) -> None:
        left, top, _, _ = d.textbbox((0, 0), text, font=f)
        d.text((x + pad - left, y + pad - top), text, fill=color, font=f)

    for text, color, own in pending:
        left, top, right, bottom = d.textbbox((0, 0), text, font=font)
        lw, lh = right - left + 2 * pad, bottom - top + 2 * pad
        # 1) the text right next to its mark, but only where it covers nothing at all
        clean = next(((x, y) for x, y in spots(own, lw, lh)
                      if on_image(x, y, lw, lh) and covered((x, y, x + lw, y + lh)) == 0), None)
        if clean:
            rect = (clean[0], clean[1], clean[0] + lw, clean[1] + lh)
            placed.append(rect)
            d.rectangle(rect, fill=(0, 0, 0))
            boxed_text(clean[0], clean[1], text, color, font)
            continue
        # 2) no room: a small numbered badge by the mark, the text goes in the legend below the image
        number = str(len(legend) + 1)
        bl, bt, br, bb = d.textbbox((0, 0), number, font=badge_font)
        bw, bh = br - bl + 2 * pad, bb - bt + 2 * pad
        # badges are small: search further out before accepting any overlap
        options = [(min(max(x, 0), max(W - bw, 0)), min(max(y, 0), max(H - bh, 0)))
                   for x, y in spots(own, bw, bh, rings=8)]
        bx, by = min(options, key=lambda p: (covered((p[0], p[1], p[0] + bw, p[1] + bh)),
                                             abs(p[0] - own[0]) + abs(p[1] - own[1])))
        rect = (bx, by, bx + bw, by + bh)
        placed.append(rect)
        d.rectangle(rect, fill=(0, 0, 0), outline=color, width=max(1, line // 2))
        boxed_text(bx, by, number, color, badge_font)
        legend.append((number, text, color))

    if not legend:
        return out
    # Legend strip appended below the image: the image itself (and every mark coordinate) is unchanged.
    # A smaller font than on-image labels, so a crowded image doesn't shrink much on the wall.
    lfont = _font(max(14, round(long_edge / 40)))
    row = lfont.size + pad
    strip = Image.new("RGB", (W, H + row * len(legend) + pad), (0, 0, 0))
    strip.paste(out, (0, 0))
    ds = ImageDraw.Draw(strip)
    for i, (number, text, color) in enumerate(legend):
        entry = f"{number}  {text}"
        while len(entry) > 4 and ds.textlength(entry, font=lfont) > W - 2 * pad:  # never cut at the edge
            entry = entry[:-2] + "…"
        y = H + pad + i * row
        left, top, _, _ = ds.textbbox((0, 0), entry, font=lfont)
        ds.text((pad - left, y - top), entry, fill=color, font=lfont)
    return strip
