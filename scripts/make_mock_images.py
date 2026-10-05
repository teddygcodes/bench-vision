"""Generate synthetic bench images for --mock mode.

    uv run python scripts/make_mock_images.py

Writes mock/scope.jpg (1920x1080, top-down) and mock/side.jpg (3840x2160,
oblique). The board has a row of through-hole joints with known defects so the
inspection workflow can be exercised without hardware:

  J1 good, J2 good, J3 bridge to J4, J5 cold (grainy, dull), J6 unsoldered pin,
  J7 insufficient, J8 good, plus a solder ball between J7 and J8.
"""

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "mock"
rng = np.random.default_rng(7)


def board(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 95, 30)  # solder mask green (BGR)
    noise = rng.normal(0, 6, (h, w, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(int(h * 0.15), int(h * 0.9), int(h * 0.12)):  # traces
        cv2.line(img, (0, y), (w, y + int(h * 0.03)), (45, 120, 40), max(2, w // 300))
    return img


def fillet(img, c, r, kind):
    x, y = c
    pad = (60, 170, 200)  # copper
    cv2.circle(img, c, int(r * 1.3), pad, -1)
    if kind == "unsoldered":
        cv2.circle(img, c, int(r * 0.45), (20, 20, 20), -1)  # empty hole
        cv2.circle(img, c, int(r * 0.3), (150, 150, 155), -1)  # bare pin
        return
    if kind == "insufficient":
        cv2.circle(img, c, int(r * 0.85), (175, 180, 185), -1)
        cv2.circle(img, c, int(r * 1.3), pad, max(2, r // 5))
    elif kind == "cold":
        cv2.circle(img, c, int(r * 1.15), (120, 125, 125), -1)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(mask, c, int(r * 1.15), 255, -1)
        grain = rng.integers(-40, 40, img.shape[:2]).astype(np.int16)
        for ch in range(3):
            v = img[..., ch].astype(np.int16)
            img[..., ch] = np.where(mask > 0, np.clip(v + grain, 0, 255), v).astype(np.uint8)
    else:  # good: smooth concave cone, bright highlight
        for i in range(20, 0, -1):
            rr = int(r * 1.2 * i / 20)
            shade = int(150 + 90 * (1 - i / 20))
            cv2.circle(img, c, rr, (shade, shade, shade + 5), -1)
        cv2.circle(img, (x - r // 4, y - r // 4), max(2, r // 6), (255, 255, 255), -1)
    cv2.circle(img, c, int(r * 0.3), (200, 200, 205), -1)  # pin tip


def draw_joints(img, xs, y, r):
    kinds = ["good", "good", "good", "good", "cold", "unsoldered", "insufficient", "good"]
    for i, (x, k) in enumerate(zip(xs, kinds)):
        fillet(img, (x, y), r, k)
        cv2.putText(img, f"J{i + 1}", (x - r, y + int(r * 2.6)), cv2.FONT_HERSHEY_SIMPLEX,
                    r / 22, (230, 230, 230), max(1, r // 10))
    # bridge J3-J4
    cv2.rectangle(img, (xs[2], y - r // 2), (xs[3], y + r // 2), (190, 195, 200), -1)
    # solder ball between J7 and J8
    bx = (xs[6] + xs[7]) // 2
    cv2.circle(img, (bx, y - int(r * 1.8)), max(3, r // 4), (215, 215, 220), -1)
    cv2.circle(img, (bx - r // 12, y - int(r * 1.85)), max(1, r // 12), (255, 255, 255), -1)


def scope() -> np.ndarray:
    w, h = 1920, 1080
    img = board(w, h)
    r = 48
    xs = [int(w * (0.12 + 0.108 * i)) for i in range(8)]
    draw_joints(img, xs, h // 2, r)
    cv2.putText(img, "U1  MOCK BOARD", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (235, 235, 235), 4)
    return img


def side() -> np.ndarray:
    w, h = 3840, 2160
    flat = board(w, h)
    r = 90
    xs = [int(w * (0.12 + 0.108 * i)) for i in range(8)]
    draw_joints(flat, xs, h // 2, r)
    # squash into a ~40 degree oblique view from the left
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.10, h * 0.30], [w * 0.98, h * 0.18], [w * 0.98, h * 0.92], [w * 0.0, h * 0.80]])
    img = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(25, 25, 28))
    # soldering iron coming in from the upper left
    tip = (int(w * 0.36), int(h * 0.50))
    cv2.line(img, (int(w * 0.02), int(h * 0.05)), tip, (70, 70, 75), 60)
    cv2.line(img, (int(w * 0.20), int(h * 0.30)), tip, (160, 160, 170), 22)
    cv2.circle(img, tip, 14, (200, 200, 210), -1)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "scope.jpg"), scope(), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / "side.jpg"), side(), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", OUT / "scope.jpg", OUT / "side.jpg")
