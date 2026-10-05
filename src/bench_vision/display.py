"""`bench-vision display`: the wall-monitor page, fed by server-sent events.

A small threaded HTTP server (stdlib only):
  GET  /            the page (black, high contrast, meant for Chromium --kiosk)
  GET  /events      server-sent events: the full state on connect and after every change
  GET  /state       the same state as JSON (handy for checking from a shell)
  GET  /img/<id>    an image referenced by the state
  POST /api/push    used by the MCP tools: {"area": "image" | "step" | "clear", ...}

The two areas are independent: pushing an image never touches the step panel and vice versa.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY = 40 * 1024 * 1024  # a couple of 1920 px JPEGs, base64-encoded
KEEPALIVE = 15.0  # seconds between SSE comments, so dead connections are noticed
MAX_TEXT = {"caption": 200, "label": 60, "title": 80, "body": 500, "progress": 60}


class PushError(ValueError):
    """A malformed push; reported to the caller as HTTP 400 with this message."""


def _text(payload: dict, key: str, required: bool = False) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        if required:
            raise PushError(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise PushError(f"'{key}' must be a string")
    if len(value) > MAX_TEXT[key]:
        raise PushError(f"'{key}' is longer than {MAX_TEXT[key]} characters")
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


class DisplayState:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.version = 0
        self._next_id = 0
        self.images: dict[str, bytes] = {}
        self.image_area: dict[str, Any] = {"images": [], "caption": None}
        self.step: dict[str, Any] = {"title": None, "body": None, "progress": None, "image": None}

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
            return {"version": self.version, "image_area": self.image_area, "step": self.step}

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
                    "caption": caption,
                }
            elif area == "step":
                title, body = _text(payload, "title", required=True), _text(payload, "body", required=True)
                progress = _text(payload, "progress")
                image = payload.get("image_jpeg_b64")
                image_id = self._store(_jpeg(image, "image_jpeg_b64")) if image is not None else None
                self.step = {"title": title, "body": body, "progress": progress, "image": image_id}
            elif area == "clear":
                which = payload.get("which", "all")
                if which not in ("all", "image", "step"):
                    raise PushError("'which' must be 'all', 'image' or 'step'")
                if which in ("all", "image"):
                    self.image_area = {"images": [], "caption": None}
                if which in ("all", "step"):
                    self.step = {"title": None, "body": None, "progress": None, "image": None}
            else:
                raise PushError("'area' must be 'image', 'step' or 'clear'")
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

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per push is enough
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
                data = state.images.get(path[len("/img/"):])
                if data is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "no such image"})
                else:
                    self._send(HTTPStatus.OK, data, "image/jpeg")
            elif path == "/events":
                self._events()
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

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


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> tuple[ThreadingHTTPServer, DisplayState]:
    state = DisplayState()
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True  # open SSE connections must not keep the process alive
    return server, state


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bench-vision</title>
<style>
  :root { --bg: #000; --fg: #fff; --dim: #9a9a9a; --accent: #ffd400; --line: #333; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--fg); overflow: hidden; overflow-wrap: anywhere;
    font-family: "DejaVu Sans", "Ubuntu", "Helvetica Neue", Arial, sans-serif; font-size: 30px; line-height: 1.3; }
  /* minmax(0, ...): never let an image's natural width widen a column */
  main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); grid-template-rows: 100vh; height: 100vh; }
  #image-area { display: flex; flex-direction: column; min-width: 0; min-height: 0; border-right: 3px solid var(--line); }
  #images { flex: 1; display: flex; gap: 16px; padding: 16px; min-height: 0; }
  figure { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  figure img { flex: 1; min-height: 0; min-width: 0; width: 100%; height: 100%; object-fit: contain; }
  figcaption { font-size: 32px; font-weight: 700; color: var(--accent); padding-bottom: 8px; }
  #caption { min-height: 96px; padding: 20px 28px; font-size: 36px; font-weight: 600;
    border-top: 3px solid var(--line); background: #111; }
  #step { display: flex; flex-direction: column; gap: 20px; padding: 28px; min-width: 0; min-height: 0; }
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
    <div id="images"><p class="empty">Waiting for an image from Claude…</p></div>
    <div id="caption"></div>
  </section>
  <aside id="step"><p class="empty">No step yet</p></aside>
</main>
<div id="offline">Display server not reachable — reconnecting…</div>
<script>
  const $ = (id) => document.getElementById(id);
  function el(tag, props, text) {
    const e = document.createElement(tag);
    Object.assign(e, props || {});
    if (text) e.textContent = text;
    return e;
  }
  let shown = {image: null, step: null};
  function render(state) {
    const ia = state.image_area, key = JSON.stringify(ia);
    if (key !== shown.image) {
      shown.image = key;
      const box = $("images");
      box.replaceChildren();
      if (!ia.images.length) box.append(el("p", {className: "empty"}, "Waiting for an image from Claude…"));
      for (const im of ia.images) {
        const fig = el("figure");
        if (im.label) fig.append(el("figcaption", {}, im.label));
        fig.append(el("img", {src: "/img/" + im.id, alt: im.label || "image"}));
        box.append(fig);
      }
      $("caption").textContent = ia.caption || "";
    }
    const st = state.step, skey = JSON.stringify(st);
    if (skey !== shown.step) {
      shown.step = skey;
      const panel = $("step");
      panel.replaceChildren();
      if (!st.title) { panel.append(el("p", {className: "empty"}, "No step yet")); return; }
      if (st.progress) panel.append(el("div", {id: "step-progress"}, st.progress));
      panel.append(el("h1", {id: "step-title"}, st.title));
      panel.append(el("div", {id: "step-body"}, st.body));
      if (st.image) {
        const wrap = el("div", {id: "step-image"});
        wrap.append(el("img", {src: "/img/" + st.image, alt: "step image"}));
        panel.append(wrap);
      }
    }
  }
  function connect() {
    const es = new EventSource("/events");
    es.onopen = () => { $("offline").style.display = "none"; };
    es.onmessage = (ev) => { render(JSON.parse(ev.data)); };
    es.onerror = () => { $("offline").style.display = "block"; };  // EventSource retries by itself
  }
  connect();
</script>
</body>
</html>
"""
