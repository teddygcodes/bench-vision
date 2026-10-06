import json
import re
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from bench_vision import live as live_mod
from bench_vision.app import BenchVision
from bench_vision.camlock import CameraLock
from bench_vision.display import make_server
from bench_vision.live import LiveStream

from conftest import call, list_tool_names, text_of


def image_path_of(text):
    return re.search(r"image_path: (\S+)", text).group(1)


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.loads(r.read())


JOINTS = [{"id": f"J{i + 1}", "x": round(1920 * (0.12 + 0.108 * i)) - 65, "y": 475, "w": 130, "h": 130}
          for i in range(8)]


SRC = Path(__file__).resolve().parent.parent / "src"


def reader_command(root, cam="scope"):
    import sys

    return [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(SRC)!r}); from bench_vision.livereader "
            f"import main; sys.exit(main())", "--root", str(root), "--cam", cam, "--mock"]


@pytest.fixture
def server(root):
    """A display server with a mock live view of 'scope' (real reader child process), sharing the root."""
    lock = CameraLock(root / ".bench-vision")
    stream = LiveStream(reader_command(root), lock)
    srv, state = make_server("127.0.0.1", 0, root=root, live=stream, live_cam="scope", live_size=(1920, 1080),
                             live_idle=0.5)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield url, state, stream
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def lbv(root, clock, server):
    url = server[0]
    (root / "config.toml").write_text(
        f'[display]\nurl = "{url}"\nlive_idle_seconds = 1\n\n[cameras.scope]\n'
        'device = "/dev/v4l/by-id/scope-video-index0"\nresolution = [1920, 1080]\n\n'
        '[cameras.side]\ndevice = "/dev/v4l/by-id/side-video-index0"\nresolution = [3840, 2160]\n'
    )
    return BenchVision(root, mock=True, clock=clock)


def test_tools_registered(lbv):
    assert {"set_target", "clear_target", "board_init", "board_set", "record_verdict"} <= set(list_tool_names(lbv))


# ------------------------------------------------------------------ camera lock


def test_capture_preempts_low_priority_holder(tmp_path):
    lock = CameraLock(tmp_path)
    assert lock.try_acquire_low()
    released = threading.Event()

    def stream_side():  # what the live view does between frames
        while not lock.wanted():
            time.sleep(0.01)
        lock.release()
        released.set()

    threading.Thread(target=stream_side, daemon=True).start()
    t0 = time.monotonic()
    with CameraLock(tmp_path).for_capture("scope"):
        assert released.is_set() and time.monotonic() - t0 < 0.5
        assert not lock.try_acquire_low()  # stream can't come back while the capture runs
    assert lock.try_acquire_low()  # ...but can afterwards
    lock.release()
    assert not list(tmp_path.glob("camera.want.*"))


def test_capture_never_waits_forever_for_a_stuck_holder(tmp_path):
    from bench_vision.errors import CameraError

    stuck = CameraLock(tmp_path)
    assert stuck.try_acquire_low()  # holds and never yields
    t0 = time.monotonic()
    with pytest.raises(CameraError, match="kept the cameras"):
        with CameraLock(tmp_path).for_capture("scope"):
            pass
    assert time.monotonic() - t0 < 2.5
    stuck.release()


def test_stuck_reader_is_killed_before_the_capture_gets_the_lock(tmp_path):
    """A reader that ignores SIGTERM and never yields: it is killed and gone before the capture runs."""
    import sys

    tag = f"bvhang{time.time_ns()}"  # unique, so other runs' processes never match
    hang = [sys.executable, "-c", f"import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"sys.stdout.flush(); time.sleep(30)  # {tag}"]
    stream = LiveStream(hang, CameraLock(tmp_path))
    token = stream.attach()
    deadline = time.monotonic() + 5
    while not CameraLock(tmp_path).held and not stream.lock.held and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)  # reader is running and holding the camera
    import subprocess

    before = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
    assert before
    t0 = time.monotonic()
    with CameraLock(tmp_path).for_capture("scope"):
        waited = time.monotonic() - t0
        still = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True).stdout.split()
        assert not set(before) & set(still)  # the reader process is gone: its device is closed
    assert waited < live_mod.STOP_GRACE + live_mod.KILL_WAIT + 0.3
    stream.detach(token)
    time.sleep(live_mod.IDLE_STOP + 0.5)  # let the stream stop (it kills its reader) before the test ends
    subprocess.run(["pkill", "-9", "-f", tag])


def test_viewer_cap_drops_the_oldest(tmp_path):
    stream = LiveStream(None, CameraLock(tmp_path), "unavailable: test")
    tokens = [stream.attach() for _ in range(live_mod.MAX_VIEWERS + 2)]
    assert stream.kicked(tokens[0]) and stream.kicked(tokens[1]) and not stream.kicked(tokens[-1])
    assert stream.viewers == live_mod.MAX_VIEWERS


def test_live_frames_are_small_and_rate_limited(server):
    url, state, stream = server
    token = stream.attach()
    try:
        seq, frame = stream.wait_frame(0, 15)
        assert frame and max(Image.open(__import__("io").BytesIO(frame)).size) == 960
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < 1.0:
            seq, _ = stream.wait_frame(seq, 1)
            n += 1
        assert n <= 17  # <= 15 fps (+ timing slack)
        assert stream.status == "live"
    finally:
        stream.detach(token)


def test_capture_during_live_view_is_fast_and_stream_resumes(server, lbv):
    url, state, stream = server
    token = stream.attach()
    try:
        seq, _ = stream.wait_frame(0, 15)
        t0 = time.monotonic()
        err, content = call(lbv, "capture", cam="scope")
        assert not err, text_of(content)
        assert time.monotonic() - t0 < 1.5
        seq2, frame = stream.wait_frame(seq, 8)
        assert frame and seq2 != seq  # live view came back after the capture
    finally:
        stream.detach(token)


def test_mjpeg_endpoint_serves_frames(server):
    url, _, _ = server
    with urllib.request.urlopen(url + "/live.mjpg", timeout=15) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        chunk = r.read(4096)
    assert b"--frame" in chunk and b"Content-Type: image/jpeg" in chunk


# ------------------------------------------------------------- live/pushed mode


def test_push_replaces_live_view_until_idle_or_clear(server, lbv):
    url, state, _ = server
    assert get_json(url + "/state")["image_area"]["mode"] == "live"
    path = image_path_of(text_of(call(lbv, "capture", cam="scope")[1]))
    call(lbv, "show", image_path=path, caption="x")
    assert get_json(url + "/state")["image_area"]["mode"] == "pushed"
    time.sleep(0.8)  # live_idle = 0.5 s in this server
    assert get_json(url + "/state")["image_area"]["mode"] == "live"
    call(lbv, "show", image_path=path, caption="x")
    call(lbv, "show_clear", area="image")
    assert get_json(url + "/state")["image_area"]["mode"] == "live"


# --------------------------------------------------------------------- targets


def test_set_and_clear_target(server, lbv):
    url, _, _ = server
    err, content = call(lbv, "set_target", cam="scope", x=995, y=475, w=130, h=130, label="J5")
    assert not err and "Reticle set on 'scope'" in text_of(content)
    t = get_json(url + "/state")["targets"]["scope"]
    assert (t["x"], t["y"], t["w"], t["h"], t["label"], t["frame_w"], t["frame_h"]) == (995, 475, 130, 130, "J5",
                                                                                         1920, 1080)
    text = text_of(call(lbv, "set_target", cam="scope", x=1900, y=1000, w=100, h=100)[1])
    assert "clipped to the 1920x1080 frame" in text
    text = text_of(call(lbv, "set_target", cam="side", x=10, y=10, w=100, h=100, label="x")[1])
    assert "live view shows 'scope'" in text
    call(lbv, "clear_target", cam="scope")
    assert "scope" not in get_json(url + "/state")["targets"]


@pytest.mark.parametrize("args,msg", [
    (dict(cam="nope", x=1, y=1, w=10, h=10), "Unknown camera"),
    (dict(cam="scope", x=5000, y=1, w=10, h=10), "outside the 1920x1080 frame"),
    (dict(cam="scope", x=1, y=1, w=0, h=10), "must be positive"),
    (dict(cam="scope", x=1, y=1, w=10, h=10, label="a\nb"), "label must be one line"),
])
def test_set_target_errors(lbv, args, msg):
    err, content = call(lbv, "set_target", **args)
    assert err and msg in text_of(content)


def test_region_and_cell_replies_offer_set_target(lbv):
    text = text_of(call(lbv, "capture_region", cam="scope", x=100, y=200, w=300, h=150)[1])
    assert 'set_target(cam="scope", x=100, y=200, w=300, h=150, label="here")' in text
    call(lbv, "grid", cam="scope", rows=2, cols=2)
    text = text_of(call(lbv, "capture_cell", cam="scope", cell="B2")[1])
    assert 'set_target(cam="scope", x=960, y=540, w=960, h=540, label="B2")' in text
    text = text_of(call(lbv, "capture_region", cam="scope", x=100, y=200, w=300, h=150, rotate=90)[1])
    assert "set_target" not in text  # rotated coordinates wouldn't match the live view


def test_targets_and_step_survive_display_restart(root, server, lbv):
    call(lbv, "set_target", cam="scope", x=10, y=20, w=30, h=40, label="J9")
    call(lbv, "show_step", title="Keep", body="me")
    srv2, state2 = make_server("127.0.0.1", 0, root=root)
    try:
        snap = state2.snapshot()
        assert snap["targets"]["scope"]["label"] == "J9" and snap["step"]["title"] == "Keep"
    finally:
        srv2.server_close()


# ----------------------------------------------------------------------- boards


def _board(lbv):
    path = image_path_of(text_of(call(lbv, "capture", cam="scope")[1]))
    err, content = call(lbv, "board_init", name="mock-board", image_path=path, joints=JOINTS)
    assert not err, text_of(content)
    return path


def test_board_init_persists_and_shows(root, server, lbv):
    url, _, _ = server
    _board(lbv)
    data = json.loads((root / "boards" / "mock-board.json").read_text())
    assert data["size"] == [1920, 1080] and data["camera"] == "scope" and len(data["joints"]) == 8
    assert all(j["state"] == "todo" for j in data["joints"])
    assert json.loads((root / "boards" / ".current.json").read_text()) == {"name": "mock-board"}
    snap = get_json(url + "/state")
    assert snap["board"]["name"] == "mock-board" and len(snap["board"]["joints"]) == 8
    with urllib.request.urlopen(url + "/board.jpg", timeout=3) as r:
        assert max(Image.open(r).size) <= 720


def test_board_set_one_active_and_hint(root, server, lbv):
    url, _, _ = server
    _board(lbv)
    call(lbv, "board_set", name="mock-board", joint_id="J1", state="verified")
    call(lbv, "board_set", name="mock-board", joint_id="J2", state="verified")
    call(lbv, "board_set", name="mock-board", joint_id="J3", state="flagged")
    call(lbv, "board_set", name="mock-board", joint_id="J4", state="active")
    text = text_of(call(lbv, "board_set", name="mock-board", joint_id="J5", state="active")[1])
    assert "J4 went back to todo" in text and 'set_target(cam="scope", x=995, y=475, w=130, h=130, label="J5")' in text
    states = {j["id"]: j["state"] for j in get_json(url + "/state")["board"]["joints"]}
    assert states == {"J1": "verified", "J2": "verified", "J3": "flagged", "J4": "todo", "J5": "active",
                      "J6": "todo", "J7": "todo", "J8": "todo"}
    assert [j["state"] for j in json.loads((root / "boards" / "mock-board.json").read_text())["joints"]].count(
        "active") == 1


@pytest.mark.parametrize("args,msg", [
    (dict(name="mock-board", joint_id="J99", state="verified"), "has no joint 'J99'"),
    (dict(name="mock-board", joint_id="J1", state="done"), "state must be one of"),
    (dict(name="other", joint_id="J1", state="todo"), "No board named 'other'"),
    (dict(name="../x", joint_id="J1", state="todo"), "Board name must be"),
])
def test_board_set_errors(lbv, args, msg):
    _board(lbv)
    err, content = call(lbv, "board_set", **args)
    assert err and msg in text_of(content)


@pytest.mark.parametrize("joints,msg", [
    ([], "non-empty list"),
    ([{"id": "J1", "x": 0, "y": 0, "w": 10}], "exactly id, x, y, w, h"),
    ([{"id": "J1", "x": 0, "y": 0, "w": 10, "h": 10}, {"id": "J1", "x": 5, "y": 5, "w": 10, "h": 10}], "duplicate"),
    ([{"id": "J 1", "x": 0, "y": 0, "w": 10, "h": 10}], "id must be"),
    ([{"id": "J1", "x": 1900, "y": 0, "w": 100, "h": 10}], "not inside the 1920x1080 image"),
    ([{"id": "J1", "x": 0, "y": 0, "w": "big", "h": 10}], "must be a number"),
])
def test_board_init_errors(lbv, joints, msg):
    path = image_path_of(text_of(call(lbv, "capture", cam="scope")[1]))
    err, content = call(lbv, "board_init", name="b", image_path=path, joints=joints)
    assert err and msg in text_of(content)


def test_record_verdict_log_thumbs_and_strip(root, server, lbv):
    url, _, _ = server
    err, content = call(lbv, "record_verdict", joint_id="J1", verdict="good", image_path="mock/scope.png")
    assert err and "No current board" in text_of(content)
    path = _board(lbv)
    for jid, v in [("J1", "good"), ("J3", "bridge"), ("J5", "Cold")]:
        err, content = call(lbv, "record_verdict", joint_id=jid, verdict=v, image_path=path, note="n")
        assert not err, text_of(content)
    lines = (root / "boards" / "mock-board" / "verdicts.jsonl").read_text().splitlines()
    assert [json.loads(x)["verdict"] for x in lines] == ["good", "bridge", "cold"]
    strip = get_json(url + "/state")["verdicts"]
    assert [(v["joint"], v["good"]) for v in strip] == [("J1", True), ("J3", False), ("J5", False)]
    with urllib.request.urlopen(f"{url}/verdict/{strip[0]['thumb']}", timeout=3) as r:
        thumb = Image.open(r)
        assert max(thumb.size) <= 240 and thumb.size[0] < 1920  # cropped to the joint
    err, content = call(lbv, "record_verdict", joint_id="J9", verdict="good", image_path=path)
    assert err and "has no joint" in text_of(content)
    err, content = call(lbv, "record_verdict", joint_id="J1", verdict="meh", image_path=path)
    assert err and "verdict must be one of" in text_of(content)


def test_verdict_strip_keeps_last_eight(root, server, lbv):
    url, _, _ = server
    path = _board(lbv)
    for i in range(11):
        call(lbv, "record_verdict", joint_id=f"J{i % 8 + 1}", verdict="good" if i % 2 else "cold", image_path=path)
    assert len(get_json(url + "/state")["verdicts"]) == 8


def test_board_and_verdicts_reload_after_display_restart(root, server, lbv):
    path = _board(lbv)
    call(lbv, "board_set", name="mock-board", joint_id="J2", state="verified")
    call(lbv, "record_verdict", joint_id="J2", verdict="good", image_path=path)
    srv2, state2 = make_server("127.0.0.1", 0, root=root)
    try:
        snap = state2.snapshot()
        assert snap["board"]["name"] == "mock-board"
        assert {j["id"]: j["state"] for j in snap["board"]["joints"]}["J2"] == "verified"
        assert snap["verdicts"][0]["joint"] == "J2"
    finally:
        srv2.server_close()


def test_board_tools_work_without_display(root, clock):
    (root / "config.toml").write_text(
        '[display]\nurl = "http://127.0.0.1:9"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\n'
        'resolution = [1920, 1080]\n')
    bv = BenchVision(root, mock=True, clock=clock)
    path = image_path_of(text_of(call(bv, "capture", cam="scope")[1]))
    err, content = call(bv, "board_init", name="b", image_path=path, joints=JOINTS[:2])
    assert not err and "wall display wasn't updated" in text_of(content)
    assert (root / "boards" / "b.json").exists()
    err, content = call(bv, "set_target", cam="scope", x=1, y=1, w=10, h=10)
    assert err and "isn't running" in text_of(content)


def test_live_config_validation(root, clock):
    (root / "config.toml").write_text('[display]\nlive_camera = "overhead"\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/s"\n')
    err, content = call(BenchVision(root, mock=True, clock=clock), "list_cameras")
    assert err and "'live_camera' must be one of the configured cameras" in text_of(content)
    (root / "config.toml").write_text('[display]\nlive_idle_seconds = 0\n\n[cameras.scope]\ndevice = "/dev/v4l/by-id/s"\n')
    err, content = call(BenchVision(root, mock=True, clock=clock), "list_cameras")
    assert err and "'live_idle_seconds' must be a number" in text_of(content)


def test_display_cli_mock_serves_live_view(root):
    import subprocess
    import sys

    src = Path(__file__).resolve().parent.parent / "src"
    import os
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "bench_vision", "display", "--mock", "--root", str(root),
                             "--port", str(port)], env={**os.environ, "PYTHONPATH": str(src)},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                snap = get_json(f"http://127.0.0.1:{port}/state")
                break
            except OSError:
                assert time.monotonic() < deadline, proc.stderr.read() if proc.poll() is not None else "no server"
                time.sleep(0.2)
        assert snap["live"]["cam"] == "scope"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/live.mjpg", timeout=5) as r:
            assert b"image/jpeg" in r.read(2048)
    finally:
        proc.terminate()
        proc.wait(5)


def test_board_name_current_and_dots_are_rejected(lbv):
    path = image_path_of(text_of(call(lbv, "capture", cam="scope")[1]))
    for bad in ("a.jpg", "x.y"):
        err, content = call(lbv, "board_init", name=bad, image_path=path, joints=JOINTS[:1])
        assert err and "Board name must be" in text_of(content)
    err, content = call(lbv, "board_init", name="current", image_path=path, joints=JOINTS[:1])
    assert not err  # allowed now: the pointer lives in .current.json
    err, content = call(lbv, "board_set", name="current", joint_id="J1", state="verified")
    assert not err, text_of(content)


def test_stale_marker_from_dead_process_is_ignored(tmp_path):
    (tmp_path / "camera.want.999999.1").write_text("scope 0\n")  # a pid that doesn't exist
    lock = CameraLock(tmp_path)
    assert not lock.wanted() and lock.try_acquire_low()
    lock.release()


def test_page_keeps_one_live_img_and_reopens_after_restart():
    from bench_vision.display import PAGE

    assert 'img.src = "data:,"' in PAGE and "restartLive" in PAGE  # Chromium: blank src closes the MJPEG stream
    assert "box.replaceChildren()" not in PAGE  # the live <img> is never dropped while streaming


# ------------------------------------------------------------ reader robustness


def test_live_reader_applies_config_controls_on_open(tmp_path, monkeypatch):
    from bench_vision import camera as camera_mod
    from bench_vision.camera import OpenCVBackend
    from bench_vision.config import CameraConfig
    from bench_vision.v4l2 import V4L2
    from fake_v4l2 import SIDE_ID, FakeV4L2Ctl, make_by_id

    class NoCap:
        def __init__(self, cam):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(camera_mod, "OpenCVStream", NoCap)
    fake = FakeV4L2Ctl()
    cam = CameraConfig("side", str(make_by_id(tmp_path) / SIDE_ID),
                       v4l2_controls=(("focus_automatic_continuous", 0), ("focus_absolute", 300)))
    OpenCVBackend(V4L2(fake)).open_stream(cam, list(cam.v4l2_controls))
    sets = [a for a in sum(fake.calls, []) if a.startswith("--set-ctrl=")]
    assert sets == ["--set-ctrl=focus_automatic_continuous=0", "--set-ctrl=focus_absolute=300"]


def test_live_reader_uses_set_control_values_of_a_running_server(root, clock):
    bv = BenchVision(root, mock=True, clock=clock)
    bv.set_control("side", "focus_automatic_continuous", 0)
    bv.set_control("side", "focus_absolute", 300)
    reader = BenchVision(root, mock=True)  # what livereader builds
    reader.load_shared_overrides()
    assert reader.cameras.controls_for(reader.cameras.get("side")) == [
        ("focus_automatic_continuous", 0), ("focus_absolute", 300)]
    live = BenchVision(root, mock=False)  # a mock server's invented controls never reach the real cameras
    live.load_shared_overrides()
    assert live.cameras.overrides == {}
    # values from a server that has exited are ignored
    path = root / ".bench-vision" / "controls-mock.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "pid": 999999}))
    stale = BenchVision(root, mock=True)
    stale.load_shared_overrides()
    assert stale.cameras.overrides == {}
    reader.load_shared_overrides()  # a long-lived reader drops the values of a server that has exited
    assert reader.cameras.overrides == {}


FAST_RETRY = "(0.2, 0.4, 0.8)"


def _start_reader(root, *extra, patch=""):
    """The real reader as the display runs it (stdin kept open), with short back-off delays."""
    import subprocess
    import sys

    code = (f"import os, sys; from bench_vision import livereader, camera; livereader.RETRY_DELAYS = {FAST_RETRY}; "
            f"livereader.NO_FRAMES = 0.3; {patch or 'pass'}; c = livereader.main(); sys.stderr.flush(); os._exit(c)")
    return subprocess.Popen([sys.executable, "-c", code, "--root", str(root), "--cam", "scope", *extra],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _lines_with_times(stream, n, timeout):
    """The first n `status:` lines (with arrival times); `backoff:` lines are skipped."""
    got, done = [], threading.Event()

    def read():
        for raw in stream:
            line = raw.decode().strip()
            if line.startswith("backoff: "):
                continue
            got.append((time.monotonic(), line))
            if len(got) >= n:
                break
        done.set()

    threading.Thread(target=read, daemon=True).start()
    done.wait(timeout)
    return got


def test_retry_delays_back_off_to_thirty_seconds():
    from bench_vision.livereader import retry_delay

    assert [retry_delay(i) for i in range(7)] == [3, 6, 12, 24, 30, 30, 30]


def test_live_reader_retries_a_missing_camera_itself_with_back_off(root):
    (root / "config.toml").write_text('[cameras.scope]\ndevice = "/dev/v4l/by-id/not-plugged-in"\n')
    proc = _start_reader(root)
    try:
        lines = _lines_with_times(proc.stderr, 4, timeout=30)
        assert proc.poll() is None  # the same process keeps trying: no exit, no respawn
    finally:
        proc.kill()
    texts = [t for _, t in lines]
    assert len(texts) == 4 and all(t.startswith("status: unavailable: Camera 'scope' is not connected") for t in texts)
    assert [t.rsplit("(retrying in ", 1)[1] for t in texts] == ["0.2 s)", "0.4 s)", "0.8 s)", "0.8 s)"]
    gaps = [b - a for (a, _), (b, _) in zip(lines, lines[1:])]
    assert gaps[0] < gaps[1] < gaps[2] + 0.1 and gaps[1] > 0.3


def test_live_reader_continues_back_off_after_restart(root):
    (root / "config.toml").write_text('[cameras.scope]\ndevice = "/dev/v4l/by-id/not-plugged-in"\n')
    proc = _start_reader(root, "--attempt", "2")
    try:
        lines = _lines_with_times(proc.stderr, 1, timeout=30)
    finally:
        proc.kill()
    assert lines and lines[0][1].endswith("(retrying in 0.8 s)")


def test_live_reader_resumes_in_the_same_process_when_the_camera_returns(root):
    import shutil

    (root / "config.toml").write_text('[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\n')
    shutil.move(root / "mock" / "scope.png", root / "scope.png")  # camera "unplugged" at start
    # a long back-off: the replug must still be noticed within about a second, not after 30 s
    proc = _start_reader(root, "--mock", patch="livereader.RETRY_DELAYS = (30.0,); livereader.PRESENCE_CHECK = 0.2")
    try:
        first = _lines_with_times(proc.stderr, 1, timeout=30)
        assert first and "unavailable" in first[0][1] and first[0][1].endswith("(retrying in 30 s)")
        shutil.move(root / "scope.png", root / "mock" / "scope.png")  # plugged back in
        head = []
        reader = threading.Thread(target=lambda: head.append(proc.stdout.read(4)), daemon=True)
        reader.start()
        reader.join(10)
        assert head and len(head[0]) == 4 and proc.poll() is None  # a frame from the same process
    finally:
        proc.kill()


def test_live_reader_reports_a_camera_that_stops_delivering_and_retries(root):
    proc = _start_reader(root, "--mock", patch="camera.MockStream.read = lambda self: None")
    try:
        lines = _lines_with_times(proc.stderr, 2, timeout=30)
        assert proc.poll() is None
    finally:
        proc.kill()
    texts = [t for _, t in lines]
    assert len(texts) == 2 and all("stopped delivering frames (unplugged?)" in t for t in texts)
    assert "Fatal" not in " ".join(texts)


def test_live_reader_exits_once_on_a_config_error(root):
    (root / "config.toml").write_text('[cameras.scope]\ndevice = 5\n')
    proc = _start_reader(root)
    try:
        rc = proc.wait(timeout=30)
    finally:
        proc.kill()
    err = proc.stderr.read().decode().strip().splitlines()
    assert rc == 3 and err and "'device' is required" in err[-1]
    assert not any(e.startswith("status:") or "Fatal" in e or "Traceback" in e for e in err)


def test_capture_outcomes_are_recorded_per_camera(tmp_path):
    from bench_vision.errors import CameraError, CameraMissingError, CameraOpenError

    lock = CameraLock(tmp_path)
    t0 = time.time() - 0.001
    assert lock.outcomes_since("scope", t0) == (False, [], False, t0)
    with lock.for_capture("scope"):
        pass
    reset, fails, _, latest = lock.outcomes_since("scope", t0)
    assert reset and fails == [] and latest > t0
    assert lock.outcomes_since("side", t0)[:2] == (False, [])
    with pytest.raises(CameraOpenError), lock.for_capture("scope"):
        raise CameraOpenError("could not be opened")
    with pytest.raises(CameraError), lock.for_capture("scope"):
        raise CameraError("no frames")  # opened, then failed: neither a success nor a failed open
    with pytest.raises(CameraOpenError), lock.for_capture("scope"):
        raise CameraMissingError("not connected")
    reset, fails, missing, latest = lock.outcomes_since("scope", t0)
    assert reset and len(fails) == 2 and missing and latest == fails[-1]  # ok, then two failed opens
    assert lock.outcomes_since("scope", latest) == (False, [], True, latest)  # all applied
    with lock.for_capture("scope"):
        pass
    assert lock.outcomes_since("scope", latest)[:2] == (True, [])  # a success clears the failures
    lock.outcome_path("scope").write_text('{"ok_time": 1' + "0" * 400 + '}')
    assert lock.outcomes_since("scope", t0)[:2] == (False, [])  # corrupt (huge time): ignored, no exception


def test_several_captures_between_reader_starts_all_count(tmp_path):
    from bench_vision.errors import CameraError, CameraOpenError
    from bench_vision.livereader import retry_delay

    def stream_at_step(n):
        stream = LiveStream(None, CameraLock(tmp_path), cam="scope")
        stream._attempts, stream._fixed_unavailable = n, False
        return stream

    # ok, then opened-but-failed: the success still resets the back-off
    stream = stream_at_step(4)
    with stream.lock.for_capture("scope"):
        pass
    with pytest.raises(CameraError), stream.lock.for_capture("scope"):
        raise CameraError("no frames")
    stream._apply_capture_outcome()
    assert stream._attempts == 0 and stream._next_try == 0.0
    # ok, then a failed open: reset, then one failed try (3 s from that failure)
    stream = stream_at_step(4)
    with stream.lock.for_capture("scope"):
        pass
    with pytest.raises(CameraOpenError), stream.lock.for_capture("scope"):
        raise CameraOpenError("busy")
    stream._apply_capture_outcome()
    assert stream._attempts == 1 and 0 < stream._next_try - time.monotonic() <= retry_delay(0)


def test_live_reader_waits_out_the_rest_of_an_interrupted_delay(root):
    (root / "config.toml").write_text('[cameras.scope]\ndevice = "/dev/v4l/by-id/not-plugged-in"\n')
    t0 = time.monotonic()
    proc = _start_reader(root, "--attempt", "1", "--wait", "0.7")  # (capped at the longest delay, 0.8 here)
    try:
        lines = _lines_with_times(proc.stderr, 1, timeout=30)
    finally:
        proc.kill()
    assert lines and lines[0][0] - t0 >= 0.7 and lines[0][1].endswith("(retrying in 0.4 s)")


TWO_CAMS = ('[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\nresolution = [1920, 1080]\n\n'
            '[cameras.side]\ndevice = "/dev/v4l/by-id/side-video-index0"\nresolution = [3840, 2160]\n')


@pytest.fixture
def steered(root, monkeypatch):
    """Live view of a missing 'scope' (real reader, real delays) with every reader command recorded."""
    import shutil
    import subprocess
    import sys

    (root / "config.toml").write_text(TWO_CAMS)
    shutil.move(root / "mock" / "scope.png", root / "scope.png")  # the live camera is unplugged
    spawned = []
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        spawned.append(cmd)
        return real_popen(cmd, **kw)

    monkeypatch.setattr(live_mod.subprocess, "Popen", popen)
    stream = LiveStream([sys.executable, "-m", "bench_vision.livereader", "--root", str(root), "--cam", "scope",
                         "--mock"], CameraLock(root / ".bench-vision", mock=True), cam="scope")
    tokens = [stream.attach()]
    yield stream, spawned, BenchVision(root, mock=True), tokens
    for t in tokens:
        stream.detach(t)
    time.sleep(live_mod.IDLE_STOP + 0.5)


def _replug(root):
    import shutil

    shutil.move(root / "scope.png", root / "mock" / "scope.png")


def _args_of(cmd):
    return int(cmd[cmd.index("--attempt") + 1]), float(cmd[cmd.index("--wait") + 1]), "--wait-missing" in cmd


def _wait_for(cond, timeout=10):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert cond()


def test_capture_outcomes_steer_the_live_back_off(root, steered):
    """Missing live camera: a capture of another camera leaves the back-off alone, a capture that can't open
    the live camera counts as a failed try, and one that works resets it and the live view comes back."""
    from bench_vision.errors import CameraMissingError
    from bench_vision.livereader import retry_delay

    stream, spawned, bv, _ = steered
    _wait_for(lambda: stream._attempts == 1)  # first try failed: waiting 3 s
    assert len(spawned) == 1 and _args_of(spawned[0]) == (0, 0.0, False)

    bv.cameras.capture("side")  # works, but says nothing about the live camera
    _wait_for(lambda: len(spawned) == 2)
    attempt, wait, missing = _args_of(spawned[1])
    assert attempt == 1 and 1.5 < wait <= 3.0 and missing  # same step, rest of the current delay
    assert "unavailable" in stream.status and "retrying in" in stream.status

    with pytest.raises(CameraMissingError):
        bv.cameras.capture("scope")  # can't open the live camera: one more failed try
    _wait_for(lambda: len(spawned) == 3)
    attempt, wait, missing = _args_of(spawned[2])
    assert attempt == 2 and retry_delay(1) - 1 < wait <= retry_delay(1) and missing

    _replug(root)
    bv.cameras.capture("scope")  # works: back-off reset, retry at once
    _wait_for(lambda: len(spawned) == 4)
    assert _args_of(spawned[3]) == (0, 0.0, False)
    _wait_for(lambda: stream.status == "live")
    assert stream._attempts == 0


def test_replug_while_the_live_view_is_stopped_is_noticed_at_once(root, steered):
    """The camera comes back during a capture of the other camera, or while no page watches: the next reader
    must not sit out the rest of the delay."""
    stream, spawned, bv, tokens = steered
    _wait_for(lambda: stream._attempts == 2, timeout=15)  # waiting 6 s now
    with CameraLock(root / ".bench-vision").for_capture("side"):
        _replug(root)  # plugged back in while the live view has given up the camera
    t0 = time.monotonic()
    _wait_for(lambda: stream.status == "live", timeout=10)
    assert time.monotonic() - t0 < 2.5 and _args_of(spawned[-1])[2]


def test_long_device_path_keeps_the_back_off_across_other_captures(root, monkeypatch):
    """A realistic by-id path makes the status line long; the display must still know the remaining wait."""
    import subprocess
    import sys

    long_path = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Microscope_Camera_SN0001-video-index0"
    (root / "config.toml").write_text(TWO_CAMS.replace("/dev/v4l/by-id/scope-video-index0", long_path))
    spawned = []
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        spawned.append(cmd)
        return real_popen(cmd, **kw)

    monkeypatch.setattr(live_mod.subprocess, "Popen", popen)
    stream = LiveStream([sys.executable, "-m", "bench_vision.livereader", "--root", str(root), "--cam", "scope"],
                        CameraLock(root / ".bench-vision"), cam="scope")
    token = stream.attach()
    try:
        _wait_for(lambda: stream._attempts == 1)
        assert "(retrying in 3 s)" in stream.status  # visible on the page despite the long reason
        for n in range(2):
            with CameraLock(root / ".bench-vision").for_capture("side"):
                pass
            _wait_for(lambda: len(spawned) == n + 2)
            attempt, wait, missing = _args_of(spawned[-1])
            assert attempt == 1 and wait > 0.5 and missing  # no retry-at-once, no extra step
        assert stream._attempts == 1
    finally:
        stream.detach(token)
        time.sleep(live_mod.IDLE_STOP + 0.5)


def test_successful_capture_while_no_page_watches_resets_the_back_off(root, monkeypatch):
    """Camera present but busy (so a replug check can't help): back-off grows, the page stops watching
    (as for 20 s after a `show`), a capture of it works, the page comes back -> retry at once from 3 s."""
    import subprocess
    import sys

    (root / "config.toml").write_text(TWO_CAMS)
    busy = root / "busy"
    busy.touch()
    code = ("import os, sys; from pathlib import Path; from bench_vision import livereader, camera; "
            "from bench_vision.errors import CameraOpenError; real = camera.MockBackend.open_stream\n"
            "def open_stream(self, cam, controls=()):\n"
            f"    if Path({str(busy)!r}).exists(): raise CameraOpenError('busy')\n"
            "    return real(self, cam, controls)\n"
            "camera.MockBackend.open_stream = open_stream; c = livereader.main(); sys.stderr.flush(); os._exit(c)")
    spawned = []
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        spawned.append(cmd)
        return real_popen(cmd, **kw)

    monkeypatch.setattr(live_mod.subprocess, "Popen", popen)
    stream = LiveStream([sys.executable, "-c", code, "--root", str(root), "--cam", "scope", "--mock"],
                        CameraLock(root / ".bench-vision", mock=True), cam="scope")
    bv = BenchVision(root, mock=True)
    token = stream.attach()
    try:
        _wait_for(lambda: stream._attempts == 2, timeout=15)  # waiting 6 s
        stream.detach(token)
        _wait_for(lambda: stream.status.startswith("stopped"), timeout=5)
        busy.unlink()
        bv.cameras.capture("scope")  # works while nobody watches
        token = stream.attach()
        _wait_for(lambda: stream.status == "live", timeout=5)
        assert _args_of(spawned[-1]) == (0, 0.0, False)
    finally:
        stream.detach(token)
        time.sleep(live_mod.IDLE_STOP + 0.5)


def test_replug_while_no_page_watches_is_noticed_on_return(root, steered):
    stream, spawned, bv, tokens = steered
    _wait_for(lambda: stream._attempts == 2, timeout=15)  # waiting 6 s now
    stream.detach(tokens.pop())
    _wait_for(lambda: stream.status.startswith("stopped"), timeout=5)
    _replug(root)
    tokens.append(stream.attach())
    t0 = time.monotonic()
    _wait_for(lambda: stream.status == "live", timeout=10)
    assert time.monotonic() - t0 < 2.5


def test_reader_stderr_flood_does_not_stall_the_stream(tmp_path):
    import sys

    flood = [sys.executable, "-c",
             "import struct, sys, time\n"
             "for i in range(400):\n"
             "    sys.stdout.buffer.write(struct.pack('>I', 3) + b'abc'); sys.stdout.buffer.flush()\n"
             "    sys.stderr.write('Corrupt JPEG data: premature end of data segment ' * 40 + '\\n'); sys.stderr.flush()\n"
             "    time.sleep(0.002)\n"
             "time.sleep(30)\n"]
    stream = LiveStream(flood, CameraLock(tmp_path))
    token = stream.attach()
    try:
        deadline = time.monotonic() + 15
        while stream.seq < 400 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stream.seq >= 400  # ~800 KB of stderr went by without blocking the reader
    finally:
        stream.detach(token)
        time.sleep(live_mod.IDLE_STOP + 0.5)
