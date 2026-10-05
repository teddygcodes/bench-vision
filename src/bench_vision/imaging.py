"""Image helpers: rotation, resizing, JPEG encoding."""

from __future__ import annotations

import io

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
