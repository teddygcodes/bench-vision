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


def test_live_reader_reports_a_missing_camera_in_one_clear_line(root):
    import subprocess
    import sys

    (root / "config.toml").write_text('[cameras.scope]\ndevice = "/dev/v4l/by-id/not-plugged-in"\n')
    proc = subprocess.Popen([sys.executable, "-m", "bench_vision.livereader", "--root", str(root), "--cam", "scope"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.wait(timeout=30)  # stdin stays open, as under the display (closing it means "exit now")
    finally:
        proc.kill()
    err = proc.stderr.read()
    lines = err.decode().strip().splitlines()
    assert proc.returncode == 3, err
    assert lines and "Camera 'scope' is not connected" in lines[-1] and "Fatal" not in err.decode()


def test_live_reader_gives_up_when_the_camera_stops_delivering(root):
    import subprocess
    import sys

    code = ("import os, sys; from bench_vision import livereader, camera; livereader.NO_FRAMES = 0.5; "
            "camera.MockStream.read = lambda self: None; c = livereader.main(); sys.stderr.flush(); os._exit(c)")
    proc = subprocess.Popen([sys.executable, "-c", code, "--root", str(root), "--cam", "scope", "--mock"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        rc = proc.wait(timeout=20)  # stdin stays open: it must exit on its own
    finally:
        proc.kill()
    err = proc.stderr.read().decode()
    assert rc == 3 and "stopped delivering frames" in err


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
