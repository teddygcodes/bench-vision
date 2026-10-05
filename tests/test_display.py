import json
import threading
import time
import urllib.request

import numpy as np
import pytest
from PIL import Image

from bench_vision.app import BenchVision
from bench_vision.display import PAGE, make_server

from conftest import call, list_tool_names, text_of


@pytest.fixture
def display():
    server, state = make_server("127.0.0.1", 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url, state
    server.shutdown()
    server.server_close()


@pytest.fixture
def dbv(root, clock, display):
    url, _ = display
    (root / "config.toml").write_text(
        '[display]\nurl = "%s"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\n'
        'resolution = [1920, 1080]\n\n[cameras.side]\ndevice = "/dev/v4l/by-id/side-video-index0"\n'
        'resolution = [3840, 2160]\n' % url
    )
    return BenchVision(root, mock=True, clock=clock)


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.loads(r.read())


def image_path_of(text):
    import re

    return re.search(r"image_path: (\S+)", text).group(1)


class SSE:
    """Minimal SSE client: collects data payloads on a background thread."""

    def __init__(self, url):
        self.events = []
        self.cond = threading.Condition()
        self.resp = urllib.request.urlopen(url + "/events", timeout=10)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        try:
            for line in self.resp:
                if line.startswith(b"data: "):
                    buf = line[6:]
                elif line == b"\n" and buf:
                    with self.cond:
                        self.events.append(json.loads(buf))
                        self.cond.notify_all()
                    buf = b""
        except Exception:  # noqa: BLE001 - connection closed at teardown
            pass

    def wait(self, n, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: len(self.events) >= n, timeout=timeout)
            return list(self.events)


def test_tools_registered(dbv):
    assert {"show", "show_compare", "show_step", "show_clear"} <= set(list_tool_names(dbv))


def test_page_layout_and_text_sizes(display):
    url, _ = display
    with urllib.request.urlopen(url, timeout=3) as r:
        html = r.read().decode()
    assert html == PAGE
    assert "background: var(--bg)" in html and "--bg: #000" in html
    assert "minmax(0, 2fr) minmax(0, 1fr)" in html  # image area ~2/3, step panel ~1/3
    assert "#step-title { font-size: 48px" in html
    import re

    sizes = [int(s) for s in re.findall(r"font-size: (\d+)px", html)]
    assert min(sizes) >= 28  # readable from 3 ft
    assert "new EventSource(\"/events\")" in html


def test_show_step_arrives_by_sse_within_a_second(dbv, display):
    url, _ = display
    sse = SSE(url)
    assert len(sse.wait(1)) == 1  # current state on connect
    t0 = time.monotonic()
    err, content = call(dbv, "show_step", title="Reflow J5", body="Flux J5.\nHeat 2-3 s.\nFeed solder.",
                        progress="joint 5 of 8")
    assert not err, text_of(content)
    events = sse.wait(2)
    assert len(events) == 2 and time.monotonic() - t0 < 1.0
    step = events[-1]["step"]
    assert step["title"] == "Reflow J5" and step["progress"] == "joint 5 of 8" and step["body"].count("\n") == 2


def test_show_with_marks_saves_annotated_copy_and_updates_image_only(dbv, display, root):
    url, state = display
    call(dbv, "show_step", title="Keep me", body="This step must survive image pushes.")
    path = image_path_of(text_of(call(dbv, "capture", cam="scope")[1]))
    marks = [{"kind": "circle", "x": 100, "y": 100, "w": 120, "h": 120, "text": "J5 cold"},
             {"kind": "box", "x": 800, "y": 400, "w": 200, "h": 100, "color": "yellow"},
             {"kind": "arrow", "x": 1500, "y": 500, "text": "ball", "color": "#00ff00"},
             {"kind": "label", "x": 50, "y": 900, "text": "U1"}]
    err, content = call(dbv, "show", image_path=path, caption="J5 cold: reflow with flux", marks=marks)
    text = text_of(content)
    assert not err, text
    marked = root / path.replace(".jpg", ".marked.jpg")
    assert marked.is_file() and f"Annotated copy: {path.replace('.jpg', '.marked.jpg')}" in text
    a = np.asarray(Image.open(root / path).convert("RGB")).astype(int)
    b = np.asarray(Image.open(marked).convert("RGB")).astype(int)
    assert a.shape == b.shape and np.abs(a - b).mean() > 0.5  # same full-res frame, marks drawn
    snap = get_json(url + "/state")
    assert snap["image_area"]["caption"] == "J5 cold: reflow with flux"
    assert snap["step"]["title"] == "Keep me"  # step panel untouched
    img_id = snap["image_area"]["images"][0]["id"]
    with urllib.request.urlopen(f"{url}/img/{img_id}", timeout=3) as r:
        sent = Image.open(r)
        assert sent.format == "JPEG" and max(sent.size) == 1920


def test_show_compare_side_by_side(dbv, display):
    url, _ = display
    call(dbv, "save_reference", cam="scope", name="r")
    text = text_of(call(dbv, "compare", cam="scope", name="r")[1])
    import re

    ref = re.search(r"reference_path: (\S+)", text).group(1)
    err, content = call(dbv, "show_compare", left_path=ref, right_path=image_path_of(text),
                        caption="Before vs now", left_label="BEFORE", right_label="NOW")
    assert not err, text_of(content)
    imgs = get_json(url + "/state")["image_area"]["images"]
    assert [i["label"] for i in imgs] == ["BEFORE", "NOW"]


def test_show_step_with_image_and_clear(dbv, display):
    url, _ = display
    path = image_path_of(text_of(call(dbv, "capture", cam="side")[1]))
    err, content = call(dbv, "show_step", title="Bridge", body="Wick it.", image_path=path,
                        marks=[{"kind": "box", "x": 10, "y": 10, "w": 50, "h": 50}])
    assert not err and get_json(url + "/state")["step"]["image"]
    call(dbv, "show", image_path=path, caption="x")
    call(dbv, "show_clear", area="step")
    snap = get_json(url + "/state")
    assert snap["step"]["title"] is None and snap["image_area"]["images"]
    call(dbv, "show_clear")
    assert get_json(url + "/state")["image_area"]["images"] == []


@pytest.mark.parametrize(
    "tool,args,msg",
    [
        ("show", dict(image_path="captures/nope.jpg", caption="x"), "does not exist"),
        ("show", dict(image_path="../../etc/passwd", caption="x"), "outside the bench-vision folder"),
        ("show", dict(image_path="mock/scope.png", caption=""), "caption is required"),
        ("show", dict(image_path="mock/scope.png", caption="x", marks=[{"kind": "star", "x": 1, "y": 1}]),
         "kind must be one of"),
        ("show", dict(image_path="mock/scope.png", caption="x", marks=[{"kind": "circle", "x": 5000, "y": 1}]),
         "outside the 1920x1080 image"),
        ("show", dict(image_path="mock/scope.png", caption="x", marks=[{"kind": "box", "x": 1, "y": 1}]),
         "a box needs w and h"),
        ("show", dict(image_path="mock/scope.png", caption="x",
                      marks=[{"kind": "circle", "x": 1, "y": 1, "color": "puce"}]), "color must be one of"),
        ("show_step", dict(title="t", body="1\n2\n3\n4\n5\n6"), "keep a step to 3-5 short lines"),
        ("show_step", dict(title="", body="b"), "title is required"),
        ("show_step", dict(title="t", body="b", marks=[{"kind": "circle", "x": 1, "y": 1}]), "need an image_path"),
        ("show_clear", dict(area="left"), "area must be"),
    ],
)
def test_show_argument_errors(dbv, tool, args, msg):
    err, content = call(dbv, tool, **args)
    assert err and msg in text_of(content) and "Traceback" not in text_of(content)


def test_display_down_is_one_line_and_captures_unaffected(root, clock):
    (root / "config.toml").write_text(
        '[display]\nurl = "http://127.0.0.1:9"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\n'
    )
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "capture", cam="scope")
    assert not err
    path = image_path_of(text_of(content))
    for tool, args in [("show", dict(image_path=path, caption="x", marks=[{"kind": "circle", "x": 9, "y": 9}])),
                       ("show_step", dict(title="t", body="b")), ("show_clear", {})]:
        err, content = call(bv, tool, **args)
        msg = text_of(content)
        assert err and "isn't running at http://127.0.0.1:9" in msg and "\n" not in msg, msg
    assert "Annotated copy" in text_of(call(bv, "show", image_path=path, caption="x",
                                            marks=[{"kind": "circle", "x": 9, "y": 9}])[1])


def test_display_rejects_bad_pushes(display):
    url, _ = display

    def post(body, ctype="application/json"):
        req = urllib.request.Request(url + "/api/push", data=body, headers={"Content-Type": ctype}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert post(b"not json")[0] == 400
    assert post(b'{"area": "image", "images": [{"jpeg_b64": "aGVsbG8="}]}')[0] == 400  # not a JPEG
    assert post(b'{"area": "step", "title": "t"}')[1]["error"] == "'body' is required"
    assert post(b'{"area": "nope"}')[0] == 400
    assert post(b'{"area": "clear"}', ctype="text/plain")[0] == 415  # blocks form-style cross-site posts
    assert post(b'{"area": "clear"}')[0] == 200


def test_display_cli_port_in_use_is_one_line(display, capsys):
    from bench_vision.cli import main

    port = int(display[0].rsplit(":", 1)[1])
    assert main(["display", "--port", str(port)]) == 2
    err = capsys.readouterr().err.strip()
    assert "cannot listen" in err and "\n" not in err


def test_call_cli_runs_any_tool(root, capsys, display):
    from bench_vision.cli import main

    url, _ = display
    (root / "config.toml").write_text(
        '[display]\nurl = "%s"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\n' % url
    )
    assert main(["call", "show_step", '{"title": "Hi", "body": "From the CLI"}', "--mock", "--root", str(root)]) == 0
    assert "Step shown on the wall display: Hi" in capsys.readouterr().out
    assert get_json(url + "/state")["step"]["title"] == "Hi"
    assert main(["call", "nope", "--mock", "--root", str(root)]) == 2
    assert main(["call", "show_step", "not json", "--mock", "--root", str(root)]) == 2
    assert main(["call", "show_step", '{"title": ""}', "--mock", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "no tool 'nope'" in err and "must be a JSON object" in err and "Traceback" not in err


def test_display_url_config_validation(root, clock):
    (root / "config.toml").write_text('[display]\nurl = "ftp://x"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/s"\n')
    err, content = call(BenchVision(root, mock=True, clock=clock), "show_clear")
    assert err and "[display]: 'url' must look like" in text_of(content)


def test_marks_on_non_capture_images_never_write_into_mock_or_references(dbv, display, root):
    before = sorted(p.name for p in (root / "mock").iterdir())
    text = text_of(call(dbv, "show", image_path="mock/scope.png", caption="x",
                        marks=[{"kind": "circle", "x": 50, "y": 50}])[1])
    assert sorted(p.name for p in (root / "mock").iterdir()) == before
    assert "Annotated copy: captures/" in text and ".marked.jpg" in text


def test_kiosk_desktop_entry_is_spec_valid():
    """Desktop-entry values allow only \\s \\n \\t \\r \\\\ escapes; GLib rejects anything else (e.g. \\")."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "deploy" / "bench-vision-kiosk.desktop").read_text()
    exec_line = next(line for line in text.splitlines() if line.startswith("Exec="))
    assert not re.search(r"\\[^sntr\\]", exec_line), exec_line
    value = exec_line[len("Exec="):]
    assert value.startswith('sh -c "') and value.endswith('"') and value.count('"') == 2
    inner = value[len('sh -c "'):-1]
    assert not any(c in inner for c in '`$\\'), inner  # must be escaped inside a quoted Exec argument
    assert inner.count("127.0.0.1:8765") == 2


def test_huge_mark_sizes_rejected_quickly(dbv):
    import time as _t

    t0 = _t.monotonic()
    err, content = call(dbv, "show", image_path="mock/scope.png", caption="x",
                        marks=[{"kind": "box", "x": 1, "y": 1, "w": 1e12, "h": 5}])
    assert err and "must be within" in text_of(content) and _t.monotonic() - t0 < 2


def test_default_arrow_near_top_left_stays_on_image():
    from bench_vision import marks

    img = Image.new("RGB", (400, 300), (0, 0, 0))
    out = np.asarray(marks.draw(img, marks.validate([{"kind": "arrow", "x": 5, "y": 5, "color": "red"}], 400, 300)))
    red = (out[..., 0] > 200) & (out[..., 1] < 80)
    assert red.sum() > 50  # the shaft/head is drawn inside the image


def test_proxy_settings_do_not_hijack_the_display(dbv, display, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("no_proxy", "")
    err, content = call(dbv, "show_clear")
    assert not err, text_of(content)


def test_display_cli_bad_port_is_one_line(capsys):
    from bench_vision.cli import main

    with pytest.raises(SystemExit) as e:
        main(["display", "--port", "99999"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "must be a port number" in err and "Traceback" not in err


@pytest.mark.parametrize("mark", [
    {"kind": "box", "x": 300, "y": 1010, "w": 60, "h": 60, "text": "J10"},
    {"kind": "circle", "x": 900, "y": 1000, "w": 60, "h": 60, "text": "J9"},
    {"kind": "box", "x": 1850, "y": 1010, "w": 60, "h": 60, "text": "corner"},
    {"kind": "circle", "x": 0, "y": 0, "w": 60, "h": 60, "text": "top-left"},
    {"kind": "arrow", "x": 1900, "y": 1070, "text": "here"},
])
def test_labels_never_cover_their_mark(mark):
    """The outline of a labelled mark must stay visible, even at the frame edges."""
    from bench_vision import marks

    img = Image.new("RGB", (1920, 1080), (0, 0, 0))
    m = marks.validate([{**mark, "color": "green"}], 1920, 1080)
    with_text = np.asarray(marks.draw(img, m)).astype(int)[:1080]  # image area (a legend may follow)
    without = np.asarray(marks.draw(img, [{**m[0], "text": None}])).astype(int)
    green = lambda a: (a[..., 1] > 180) & (a[..., 0] < 80) & (a[..., 2] < 140)  # noqa: E731
    outline = green(without)
    assert outline.sum() > 50
    # nearly all of the mark's pixels survive drawing the label (text glyphs are green too: compare masks)
    assert (green(with_text) & outline).sum() >= 0.95 * outline.sum()


def test_nul_byte_path_and_multiline_caption_are_clear(dbv):
    err, content = call(dbv, "show", image_path="captures/a\x00b.jpg", caption="x")
    assert err and "does not exist" in text_of(content) and "bug" not in text_of(content)
    err, content = call(dbv, "show", image_path="mock/scope.png", caption="line one\nline two")
    assert err and "caption must be one line" in text_of(content)


def test_https_display_url_rejected(root, clock):
    (root / "config.toml").write_text(
        '[display]\nurl = "https://127.0.0.1:8765"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/s"\n')
    err, content = call(BenchVision(root, mock=True, clock=clock), "show_clear")
    assert err and "doesn't speak https" in text_of(content)


def _visible_fraction(mark_list, size=(1920, 1080), index=None):
    """Fraction of each mark's own pixels still visible once all labels are drawn."""
    from bench_vision import marks

    img = Image.new("RGB", size, (0, 0, 0))
    m = marks.validate(mark_list, *size)
    full = np.asarray(marks.draw(img, m)).astype(int)[: size[1]]  # image area (a legend may follow)
    out = []
    for i, mk in enumerate(m):
        alone = np.asarray(marks.draw(img, [{**mk, "text": None}])).astype(int)
        mask = (np.abs(alone - 0).sum(axis=2) > 60)  # pixels this mark paints
        same = (np.abs(full - alone).sum(axis=2) < 60) & mask
        out.append(same.sum() / max(mask.sum(), 1))
    return out


@pytest.mark.parametrize("marks_in,size", [
    ([{"kind": "arrow", "x": 998, "y": 177, "w": 384, "h": 37, "text": "J3-J4 bridge"}], (3840, 2160)),
    ([{"kind": "arrow", "x": 3302, "y": 2022, "w": 488, "h": -95, "text": "solder ball"}], (3840, 2160)),
    ([{"kind": "circle", "x": 540, "y": 1881, "text": "lifted pad, reflow it"}], (1080, 1920)),
    ([{"kind": "circle", "x": 1850, "y": 1000, "w": 120, "h": 120, "text": "J5"}], (1920, 1080)),
])
def test_labels_find_room_at_edges(marks_in, size):
    assert min(_visible_fraction([{**m, "color": "green"} for m in marks_in], size)) >= 0.95


def test_labels_do_not_cover_neighbouring_marks():
    two = [{"kind": "circle", "x": 900, "y": 400, "w": 80, "h": 80, "text": "J5", "color": "green"},
           {"kind": "circle", "x": 900, "y": 520, "w": 80, "h": 80, "text": "J6", "color": "green"}]
    assert min(_visible_fraction(two)) >= 0.95


@pytest.mark.parametrize("w,h", [(0, 0), (3, 2)])
def test_zero_or_tiny_arrow_offset_still_draws_an_arrow(w, h):
    from bench_vision import marks

    img = Image.new("RGB", (1920, 1080), (0, 0, 0))
    out = np.asarray(marks.draw(img, marks.validate(
        [{"kind": "arrow", "x": 1300, "y": 520, "w": w, "h": h, "color": "red"}], 1920, 1080)))
    red = (out[..., 0] > 200) & (out[..., 1] < 80)
    assert red.sum() > 1000  # a full-size arrow, not a stray dot


def test_display_url_with_bad_port_rejected_and_hint_names_port(root, clock):
    (root / "config.toml").write_text('[display]\nurl = "http://127.0.0.1:abc"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/s"\n')
    err, content = call(BenchVision(root, mock=True, clock=clock), "show_clear")
    assert err and "'url' must look like" in text_of(content)
    (root / "config.toml").write_text('[display]\nurl = "http://127.0.0.1:9"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/s"\n')
    err, content = call(BenchVision(root, mock=True, clock=clock), "show_clear")
    assert err and "`uv run bench-vision display --port 9`" in text_of(content)


def test_step_title_must_be_one_line(dbv):
    err, content = call(dbv, "show_step", title="two\nlines", body="b")
    assert err and "title must be one line" in text_of(content)


def test_crowded_arrows_never_hide_each_others_labels():
    """Two default arrows at neighbouring joints near the top edge (the reviewer's repro)."""
    from bench_vision import marks

    for pair in ([(250, 250, "J4 cold"), (300, 200, "J5 bridge")], [(100, 250, "J4 cold"), (200, 250, "J5 bridge")]):
        m = marks.validate([{"kind": "arrow", "x": x, "y": y, "text": t, "color": "green"} for x, y, t in pair],
                           1920, 1080)
        out = marks.draw(Image.new("RGB", (1920, 1080), (0, 0, 0)), m)
        assert min(_visible_fraction([{**mk, "color": "green", "text": t} for mk, (_, _, t) in
                                      zip([{"kind": "arrow", "x": x, "y": y} for x, y, _ in pair], pair)])) >= 0.95
        if out.size[1] > 1080:  # texts that didn't fit went to the legend, numbered
            legend = np.asarray(out)[1080:]
            assert (legend.sum(axis=2) > 100).any()


def test_legend_keeps_image_area_identical_size():
    from bench_vision import marks

    crowd = [{"kind": "circle", "x": 10 + 70 * (i % 4), "y": 10 + 70 * (i // 4), "w": 30, "h": 30,
              "text": f"joint {i} cold, reflow"} for i in range(12)]  # a dense (non-touching) corner cluster
    out = marks.draw(Image.new("RGB", (1920, 1080), (0, 0, 0)), marks.validate(crowd, 1920, 1080))
    assert out.size[0] == 1920 and out.size[1] > 1080  # legend appended, width unchanged
    assert min(_visible_fraction(crowd)) >= 0.95


def test_mark_text_must_be_one_line(dbv):
    err, content = call(dbv, "show", image_path="mock/scope.png", caption="x",
                        marks=[{"kind": "circle", "x": 10, "y": 10, "text": "a\nb"}])
    assert err and "text must be one line" in text_of(content)


def test_long_legend_entries_are_shortened_not_cut():
    from bench_vision import marks

    long = "J5 cold joint: reflow with fresh flux and more heat please, then recheck"
    crowd = [{"kind": "circle", "x": 10 + 40 * i, "y": 10, "w": 20, "h": 20, "text": long} for i in range(6)]
    out = np.asarray(marks.draw(Image.new("RGB", (640, 480)), marks.validate(crowd, 640, 480)))
    assert out.shape[0] > 480
    legend = out[480:]
    assert not (legend[:, -3:].sum(axis=2) > 100).any()  # nothing drawn into the right edge
