"""`bench-vision display`: the wall-monitor page, fed by server-sent events.

A small threaded HTTP server (stdlib only):
  GET  /              the page (black, high contrast, meant for Chromium --kiosk)
  GET  /events        server-sent events: the full state on connect and after every change
  GET  /state         the same state as JSON (handy for checking from a shell)
  GET  /img/<id>      an image pushed by show/show_compare/show_step
  GET  /live.mjpg     the live camera view (MJPEG); only runs while a page is watching
  GET  /board.jpg     the current board's reference image, shrunk for the board-map tile
  GET  /verdict/<f>   a verdict thumbnail of the current board
  POST /api/push      used by the MCP tools: {"area": "image" | "step" | "clear" | "target" | "board", ...}

Image area: a pushed image for `live_idle` seconds after a push (or until show_clear("image")),
otherwise the live view with any set_target reticle. A verdict strip runs along its bottom.
Step panel: the board-map tile on top, then the current step. The step, the targets and the
board (from boards/ on disk) survive a display restart.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_LIVE_IDLE = 20.0
MAX_BODY = 40 * 1024 * 1024  # a couple of 1920 px JPEGs, base64-encoded
KEEPALIVE = 15.0  # seconds between SSE comments, so dead connections are noticed
MAX_TEXT = {"caption": 200, "label": 60, "title": 80, "body": 500, "progress": 60}
BOARD_TILE_EDGE = 720


class PushError(ValueError):
    """A malformed push; reported to the caller as HTTP 400 with this message."""


def _text(payload: dict, key: str, required: bool = False, limit_key: str | None = None) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        if required:
            raise PushError(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise PushError(f"'{key}' must be a string")
    limit = MAX_TEXT[limit_key or key]
    if len(value) > limit:
        raise PushError(f"'{key}' is longer than {limit} characters")
    return value


def _jpeg(item: Any, where: str) -> bytes:
    if not isinstance(item, str):
        raise PushError(f"{where} must be base64 JPEG data")
    try:
        data = base64.b64decode(item, validate=True)
    except (binascii.Error, ValueError):
        raise PushError(f"{where} is not valid base64") from None
    if not data.startswith(b"\xff\xd8"):
        raise PushError(f"{where} is not a JPEG")
    return data


def _number(payload: dict, key: str) -> float:
    v = payload.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise PushError(f"'{key}' must be a number")
    return float(v)


def _finite(v: Any) -> float:
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError("not a finite number")
    return f


class DisplayState:
    def __init__(self, root: Path | None = None, live: Any = None, live_cam: str | None = None,
                 live_size: tuple[int, int] | None = None, live_idle: float = DEFAULT_LIVE_IDLE) -> None:
        self._cond = threading.Condition()
        self.version = 0
        self._next_id = 0
        self.root = root
        self.live = live
        self.live_idle = live_idle
        self.images: dict[str, bytes] = {}
        self.image_area: dict[str, Any] = {"images": [], "caption": None, "mode": "live"}
        self.step: dict[str, Any] = {"title": None, "body": None, "progress": None, "image": None}
        self.targets: dict[str, dict[str, Any]] = {}
        self.board: dict[str, Any] | None = None
        self.verdicts: list[dict[str, Any]] = []
        self.live_info = {"cam": live_cam, "size": list(live_size) if live_size else None}
        self._pushed_at = 0.0
        self._timer: threading.Timer | None = None
        self._board_jpeg: tuple[str, float, bytes] | None = None
        if root is not None:
            self._restore()
            self._reload_board()

    # ---------------------------------------------------------- persistence

    def _state_file(self) -> Path | None:
        return self.root / ".bench-vision" / "display.json" if self.root else None

    def _restore(self) -> None:
        p = self._state_file()
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p else None
        except (OSError, ValueError, RecursionError):
            return
        if not isinstance(data, dict):
            return
        step = data.get("step")
        if isinstance(step, dict) and isinstance(step.get("title"), str) and isinstance(step.get("body"), str):
            self.step = {"title": step["title"][:80], "body": step["body"][:500],
                         "progress": step.get("progress")[:60] if isinstance(step.get("progress"), str) else None,
                         "image": None}  # pushed images are not kept across restarts
        targets = data.get("targets")
        if isinstance(targets, dict):
            for cam, t in list(targets.items())[:16]:
                try:
                    self.targets[str(cam)[:32]] = self._clean_target(t)
                except PushError:
                    continue

    def _persist(self) -> None:
        p = self._state_file()
        if p is None:
            return
        data = {"step": {k: v for k, v in self.step.items() if k != "image"}, "targets": self.targets}
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".display.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data) + "\n", encoding="utf-8")
            tmp.replace(p)
        except OSError as e:
            log.warning("could not save display state: %s", e)

    def _reload_board(self) -> None:
        """Re-read the current board and its last verdicts from boards/ (written by the MCP tools)."""
        if self.root is None:
            return
        from .boards import BoardStore
        from .errors import BenchVisionError

        store = BoardStore(self.root / "boards")
        name = store.current_name()
        if name is None:
            self.board, self.verdicts = None, []
            return
        try:
            b = store.load(name)
            size = [int(b["size"][0]), int(b["size"][1])]
            joints = [
                {"id": str(j["id"])[:24], **{k: _finite(j[k]) for k in ("x", "y", "w", "h")},
                 "state": j.get("state") if j.get("state") in ("todo", "active", "verified", "flagged") else "todo"}
                for j in b["joints"] if isinstance(j, dict)
            ]
            mtime = store.image_path(name).stat().st_mtime
            self.board = {"name": name, "size": size, "joints": joints, "rev": f"{name}-{mtime:.0f}"}
        except (BenchVisionError, KeyError, TypeError, ValueError, IndexError, OSError) as e:
            log.warning("could not load board %s: %s", name, e)
            self.board = None
        self.verdicts = [
            {"joint": v["joint"][:24], "verdict": str(v.get("verdict", ""))[:20], "good": v.get("verdict") == "good",
             "thumb": v["thumb"]}
            for v in store.verdicts(name)
        ] if self.board else []

    def board_jpeg(self) -> bytes | None:
        if self.root is None or self.board is None:
            return None
        from PIL import Image

        from .boards import BoardStore

        path = BoardStore(self.root / "boards").image_path(self.board["name"])
        try:
            mtime = path.stat().st_mtime
            if self._board_jpeg and self._board_jpeg[:2] == (str(path), mtime):
                return self._board_jpeg[2]
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((BOARD_TILE_EDGE, BOARD_TILE_EDGE))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
        except (OSError, ValueError, Image.DecompressionBombError):
            return None
        self._board_jpeg = (str(path), mtime, buf.getvalue())
        return self._board_jpeg[2]

    def verdict_jpeg(self, thumb: str) -> bytes | None:
        if self.root is None or self.board is None:
            return None
        from .boards import BoardStore

        p = BoardStore(self.root / "boards").thumb_path(self.board["name"], thumb)
        try:
            return p.read_bytes() if p else None
        except OSError:
            return None

    # --------------------------------------------------------------- state

    def _store(self, data: bytes) -> str:
        self._next_id += 1
        key = f"{self._next_id}"
        self.images[key] = data
        return key

    def _gc(self) -> None:
        live = {i["id"] for i in self.image_area["images"]} | ({self.step["image"]} if self.step["image"] else set())
        for key in list(self.images):
            if key not in live:
                del self.images[key]

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            live = dict(self.live_info)
            live["status"] = self.live.status if self.live is not None else "unavailable: no live camera configured"
            # copies: the SSE writer serialises this outside the lock while pushes keep arriving
            return {"version": self.version, "image_area": self.image_area, "step": self.step,
                    "targets": {k: dict(v) for k, v in self.targets.items()}, "board": self.board,
                    "verdicts": list(self.verdicts), "live": live}

    def touch(self) -> None:
        """Something outside a push changed (e.g. live-view status): tell the pages."""
        with self._cond:
            self.version += 1
            self._cond.notify_all()

    def _back_to_live(self, pushed_at: float) -> None:
        with self._cond:
            if self._pushed_at == pushed_at and self.image_area["mode"] == "pushed":
                self.image_area = {**self.image_area, "mode": "live"}
                self.version += 1
                self._cond.notify_all()

    @staticmethod
    def _clean_target(t: Any) -> dict[str, Any]:
        if not isinstance(t, dict):
            raise PushError("target must be an object")
        out: dict[str, Any] = {k: _number(t, k) for k in ("x", "y", "w", "h", "frame_w", "frame_h")}
        if out["w"] <= 0 or out["h"] <= 0 or out["frame_w"] <= 0 or out["frame_h"] <= 0:
            raise PushError("target sizes must be positive")
        out["label"] = _text(t, "label", limit_key="label")
        return out

    def apply(self, payload: Any) -> int:
        if not isinstance(payload, dict):
            raise PushError("push must be a JSON object")
        area = payload.get("area")
        with self._cond:
            if area == "image":
                items = payload.get("images")
                if not isinstance(items, list) or not 1 <= len(items) <= 2:
                    raise PushError("'images' must be a list of 1 or 2 images")
                decoded = []
                for n, item in enumerate(items):
                    if not isinstance(item, dict):
                        raise PushError(f"images[{n}] must be an object")
                    decoded.append((_jpeg(item.get("jpeg_b64"), f"images[{n}].jpeg_b64"), _text(item, "label")))
                caption = _text(payload, "caption")
                self.image_area = {
                    "images": [{"id": self._store(data), "label": label} for data, label in decoded],
                    "caption": caption, "mode": "pushed",
                }
                self._pushed_at = time.monotonic()
                if self._timer is not None:
                    self._timer.cancel()
                self._timer = threading.Timer(self.live_idle, self._back_to_live, args=(self._pushed_at,))
                self._timer.daemon = True
                self._timer.start()
            elif area == "step":
                title, body = _text(payload, "title", required=True), _text(payload, "body", required=True)
                progress = _text(payload, "progress")
                image = payload.get("image_jpeg_b64")
                image_id = self._store(_jpeg(image, "image_jpeg_b64")) if image is not None else None
                self.step = {"title": title, "body": body, "progress": progress, "image": image_id}
                self._persist()
            elif area == "clear":
                which = payload.get("which", "all")
                if which not in ("all", "image", "step"):
                    raise PushError("'which' must be 'all', 'image' or 'step'")
                if which in ("all", "image"):
                    self.image_area = {"images": [], "caption": None, "mode": "live"}
                if which in ("all", "step"):
                    self.step = {"title": None, "body": None, "progress": None, "image": None}
                    self._persist()
            elif area == "target":
                cam = payload.get("cam")
                if not isinstance(cam, str) or not cam or len(cam) > 32:
                    raise PushError("'cam' must be a camera name")
                if payload.get("clear"):
                    self.targets.pop(cam, None)
                else:
                    if len(self.targets) >= 16 and cam not in self.targets:
                        raise PushError("too many targets")
                    self.targets[cam] = self._clean_target(payload)
                self._persist()
            elif area == "board":
                self._reload_board()
            else:
                raise PushError("'area' must be 'image', 'step', 'clear', 'target' or 'board'")
            self._gc()
            self.version += 1
            self._cond.notify_all()
            return self.version

    def wait_newer(self, version: int, timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self.version != version, timeout=timeout)


def make_handler(state: DisplayState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "bench-vision-display"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                self._json(HTTPStatus.OK, state.snapshot())
            elif path.startswith("/img/"):
                self._image(state.images.get(path[len("/img/"):]))
            elif path == "/board.jpg":
                self._image(state.board_jpeg())
            elif path.startswith("/verdict/"):
                self._image(state.verdict_jpeg(path[len("/verdict/"):]))
            elif path == "/events":
                self._events()
            elif path == "/live.mjpg":
                self._mjpeg()
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _image(self, data: bytes | None) -> None:
            if data is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "no such image"})
            else:
                self._send(HTTPStatus.OK, data, "image/jpeg")

        def _events(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.close_connection = True
            try:
                snap = state.snapshot()
                while True:
                    self.wfile.write(b"retry: 1000\ndata: " + json.dumps(snap).encode() + b"\n\n")
                    self.wfile.flush()
                    seen = snap["version"]
                    while not state.wait_newer(seen, KEEPALIVE):
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    snap = state.snapshot()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return  # the page went away; it reconnects on its own

        def _mjpeg(self) -> None:
            live = state.live
            if live is None:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no live camera configured"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            token = live.attach()
            try:
                seen = -1
                while not live.kicked(token):
                    seen, frame = live.wait_frame(seen, timeout=1.0)
                    if frame is None or live.kicked(token):
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                live.detach(token)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/api/push":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                # Also blocks plain-form posts from other web pages (no CORS preflight for those).
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Content-Type must be application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"body must be 1..{MAX_BODY} bytes"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                version = state.apply(payload)
            except (ValueError, RecursionError) as e:  # PushError and JSON errors
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e) if isinstance(e, PushError) else "invalid JSON"})
                return
            self._json(HTTPStatus.OK, {"ok": True, "version": version})

    return Handler


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, root: Path | None = None,
                live: Any = None, live_cam: str | None = None, live_size: tuple[int, int] | None = None,
                live_idle: float = DEFAULT_LIVE_IDLE) -> tuple[ThreadingHTTPServer, DisplayState]:
    state = DisplayState(root, live, live_cam, live_size, live_idle)
    if live is not None:
        live.on_status = state.touch
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True  # open SSE/MJPEG connections must not keep the process alive
    return server, state


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bench-vision</title>
<style>
  :root { --bg: #000; --fg: #fff; --dim: #9a9a9a; --accent: #ffd400; --line: #333;
          --todo: #8a8a8a; --active: #ffd400; --verified: #20d060; --flagged: #ff3b3b; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--fg); overflow: hidden; overflow-wrap: anywhere;
    font-family: "DejaVu Sans", "Ubuntu", "Helvetica Neue", Arial, sans-serif; font-size: 30px; line-height: 1.3; }
  /* minmax(0, ...): never let an image's natural width widen a column */
  main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); grid-template-rows: 100vh; height: 100vh; }
  #image-area { display: flex; flex-direction: column; min-width: 0; min-height: 0; border-right: 3px solid var(--line); }
  #images { flex: 1; display: flex; gap: 16px; padding: 16px; min-height: 0; position: relative; }
  figure { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  figure img { flex: 1; min-height: 0; min-width: 0; width: 100%; height: 100%; object-fit: contain; }
  figcaption { font-size: 32px; font-weight: 700; color: var(--accent); padding-bottom: 8px; }
  #live { flex: 1; position: relative; min-width: 0; min-height: 0; }
  #live img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
  #live-badge { position: absolute; left: 12px; top: 12px; font-size: 28px; font-weight: 700; padding: 4px 14px;
    background: #b00000; color: #fff; border-radius: 6px; }
  #live-status { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    text-align: center; color: var(--dim); font-size: 32px; padding: 40px; pointer-events: none; }
  #reticle { position: absolute; border: 5px solid var(--accent); box-shadow: 0 0 0 3px #000, inset 0 0 0 3px #000;
    pointer-events: none; display: none; }
  #reticle span { position: absolute; left: -5px; top: -52px; white-space: nowrap; font-size: 30px; font-weight: 800;
    color: #000; background: var(--accent); padding: 2px 12px; }
  #reticle.below span { top: auto; bottom: -52px; }
  #verdicts { display: none; gap: 12px; padding: 10px 16px; border-top: 3px solid var(--line); overflow: hidden; }
  /* 8 equal slots: every one of the last 8 verdicts always fits */
  #verdicts figure { flex: 0 0 calc((100% - 7 * 12px) / 8); min-width: 0; align-items: center; }
  #verdicts img { flex: none; width: 100%; height: 110px; min-height: 0; object-fit: cover;
    border: 6px solid var(--verified); }
  #verdicts figure.bad img { border-color: var(--flagged); }
  #verdicts figcaption { font-size: 28px; color: var(--fg); padding: 2px 0 0; }
  #caption { min-height: 96px; padding: 20px 28px; font-size: 36px; font-weight: 600;
    border-top: 3px solid var(--line); background: #111; }
  #panel { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  #board { display: none; flex-direction: column; gap: 6px; padding: 16px 20px 12px; border-bottom: 3px solid var(--line); }
  #board-head { font-size: 28px; font-weight: 700; color: var(--accent); }
  #board-map { position: relative; }
  #board-map img { display: block; width: 100%; max-height: 34vh; object-fit: contain; }
  #board-map svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  #board-map rect { fill: none; stroke-width: 5; vector-effect: non-scaling-stroke; }
  rect.todo { stroke: var(--todo); }
  rect.verified { stroke: var(--verified); fill: rgba(32, 208, 96, .25) !important; }
  rect.flagged { stroke: var(--flagged); fill: rgba(255, 59, 59, .30) !important; }
  rect.active { stroke: var(--active); fill: rgba(255, 212, 0, .35) !important; animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
  #step { display: flex; flex-direction: column; gap: 20px; padding: 28px; min-width: 0; min-height: 0; flex: 1; }
  #step-progress { font-size: 32px; font-weight: 700; color: var(--accent); }
  #step-title { font-size: 48px; font-weight: 800; line-height: 1.15; }
  #step-body { font-size: 30px; white-space: pre-line; }
  #step-image { flex: 1; min-height: 0; display: flex; }
  #step-image img { width: 100%; height: 100%; object-fit: contain; object-position: top; }
  .empty { color: var(--dim); font-size: 32px; margin: auto; text-align: center; }
  #offline { position: fixed; left: 0; right: 0; bottom: 0; padding: 12px 28px; font-size: 28px;
    background: #7a0000; color: #fff; display: none; }
</style>
</head>
<body>
<main>
  <section id="image-area">
    <div id="images"></div>
    <div id="verdicts"></div>
    <div id="caption"></div>
  </section>
  <aside id="panel">
    <div id="board"><div id="board-head"></div><div id="board-map"><img alt="board"><svg></svg></div></div>
    <div id="step"><p class="empty">No step yet</p></div>
  </aside>
</main>
<div id="offline">Display server not reachable — reconnecting…</div>
<script>
  const $ = (id) => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  function el(tag, props, text) {
    const e = document.createElement(tag);
    Object.assign(e, props || {});
    if (text) e.textContent = text;
    return e;
  }
  let shown = {image: null, step: null, board: null, verdicts: null};
  let state = null;

  // One persistent live <img>. Chromium keeps an MJPEG connection open after the element is removed,
  // so leaving live mode blanks its src (which closes the stream) instead of dropping the element.
  function liveImg() { return $("live-img"); }
  function startLive() {
    const img = liveImg();
    if (img && !img.dataset.on) { img.dataset.on = "1"; img.src = "/live.mjpg?t=" + Date.now(); }
  }
  function stopLive() {
    const img = liveImg();
    if (img && img.dataset.on) { delete img.dataset.on; img.removeAttribute("src"); img.src = "data:,"; }
  }
  function restartLive() { stopLive(); if ($("live").style.display !== "none") startLive(); }

  function liveView(st) {
    let wrap = $("live");
    if (!wrap) {
      wrap = el("div", {id: "live"});
      const img = el("img", {id: "live-img", alt: "live view"});
      img.onload = placeReticle;
      img.onerror = () => { if (img.dataset.on) setTimeout(restartLive, 1000); };
      const ret = el("div", {id: "reticle"});
      ret.append(el("span"));
      wrap.append(img, ret, el("div", {id: "live-badge"}, "LIVE"), el("div", {id: "live-status"}));
      $("images").append(wrap);
    }
    for (const f of $("images").querySelectorAll("figure")) f.remove();
    wrap.style.display = "block";
    startLive();
    const status = st.live.status || "";
    const ok = status === "live";
    $("live-status").textContent = ok ? "" : "Live view " + status;
    $("live-badge").style.display = ok ? "block" : "none";
    $("caption").textContent = st.live.cam ? "Live: " + st.live.cam : "";
    placeReticle();
  }

  function placeReticle() {
    const ret = $("reticle"), img = $("live-img");
    if (!ret || !img || !state) return;
    const t = state.live.cam ? state.targets[state.live.cam] : null;
    if (!t) { ret.style.display = "none"; return; }
    const cw = img.clientWidth, ch = img.clientHeight;
    const nw = img.naturalWidth || t.frame_w, nh = img.naturalHeight || t.frame_h;
    const s = Math.min(cw / nw, ch / nh), dw = nw * s, dh = nh * s;
    const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
    const kx = dw / t.frame_w, ky = dh / t.frame_h;
    ret.style.left = (ox + t.x * kx) + "px";
    ret.style.top = (oy + t.y * ky) + "px";
    ret.style.width = Math.max(t.w * kx, 12) + "px";
    ret.style.height = Math.max(t.h * ky, 12) + "px";
    ret.classList.toggle("below", oy + t.y * ky < 60);
    ret.firstChild.textContent = t.label || "";
    ret.firstChild.style.display = t.label ? "block" : "none";
    ret.style.display = "block";
  }
  window.addEventListener("resize", placeReticle);

  function renderImages(st) {
    const ia = st.image_area;
    if (ia.mode === "live" || !ia.images.length) { shown.image = null; liveView(st); return; }
    const key = JSON.stringify(ia);
    if (key === shown.image) return;
    shown.image = key;
    const box = $("images");
    stopLive();  // closes the MJPEG stream, so the display frees the camera
    if ($("live")) $("live").style.display = "none";
    for (const f of box.querySelectorAll("figure")) f.remove();
    for (const im of ia.images) {
      const fig = el("figure");
      if (im.label) fig.append(el("figcaption", {}, im.label));
      fig.append(el("img", {src: "/img/" + im.id, alt: im.label || "image"}));
      box.append(fig);
    }
    $("caption").textContent = ia.caption || "";
  }

  function renderVerdicts(st) {
    const key = JSON.stringify(st.verdicts);
    if (key === shown.verdicts) return;
    shown.verdicts = key;
    const strip = $("verdicts");
    strip.replaceChildren();
    strip.style.display = st.verdicts.length ? "flex" : "none";
    for (const v of st.verdicts) {
      const fig = el("figure", {className: v.good ? "good" : "bad"});
      fig.append(el("img", {src: "/verdict/" + v.thumb, alt: v.joint + " " + v.verdict}),
                 el("figcaption", {}, v.joint));
      strip.append(fig);
    }
    placeReticle();
  }

  function renderBoard(st) {
    const key = JSON.stringify(st.board);
    if (key === shown.board) return;
    shown.board = key;
    const b = st.board, tile = $("board");
    if (!b) { tile.style.display = "none"; return; }
    tile.style.display = "flex";
    const n = (s) => b.joints.filter((j) => j.state === s).length;
    $("board-head").textContent = b.name + " · " + n("verified") + " verified · " + n("flagged") + " flagged · "
      + b.joints.length + " joints";
    const img = $("board-map").querySelector("img");
    const src = "/board.jpg?rev=" + encodeURIComponent(b.rev);
    if (img.getAttribute("src") !== src) img.src = src;
    const svg = $("board-map").querySelector("svg");
    svg.setAttribute("viewBox", "0 0 " + b.size[0] + " " + b.size[1]);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.replaceChildren();
    for (const j of b.joints) {
      const r = document.createElementNS(SVG, "rect");
      for (const [k, v] of [["x", j.x], ["y", j.y], ["width", j.w], ["height", j.h]]) r.setAttribute(k, v);
      r.setAttribute("class", j.state);
      svg.append(r);
    }
  }

  function renderStep(st) {
    const s = st.step, key = JSON.stringify(s);
    if (key === shown.step) return;
    shown.step = key;
    const panel = $("step");
    panel.replaceChildren();
    if (!s.title) { panel.append(el("p", {className: "empty"}, "No step yet")); return; }
    if (s.progress) panel.append(el("div", {id: "step-progress"}, s.progress));
    panel.append(el("h1", {id: "step-title"}, s.title));
    panel.append(el("div", {id: "step-body"}, s.body));
    if (s.image) {
      const wrap = el("div", {id: "step-image"});
      wrap.append(el("img", {src: "/img/" + s.image, alt: "step image"}));
      panel.append(wrap);
    }
  }

  function render(st) {
    state = st;
    renderImages(st);
    renderVerdicts(st);
    renderBoard(st);
    renderStep(st);
  }

  function connect() {
    const es = new EventSource("/events");
    es.onopen = () => {
      $("offline").style.display = "none";
      shown = {image: null, step: null, board: null, verdicts: null};
      if (liveImg() && liveImg().dataset.on) restartLive();  // the server may have restarted: reopen the stream
    };
    es.onmessage = (ev) => { render(JSON.parse(ev.data)); };
    es.onerror = () => { $("offline").style.display = "block"; };  // EventSource retries by itself
  }
  connect();
</script>
</body>
</html>
"""
