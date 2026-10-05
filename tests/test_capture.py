import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bench_vision.app import BenchVision
from bench_vision.camera import CameraManager, Frame, MockBackend, OpenCVBackend
from bench_vision.config import CameraConfig
from bench_vision.errors import BenchVisionError, CameraError

from conftest import call, images_of, jpeg_size, list_tool_names, text_of


def test_tools_registered(bv):
    names = list_tool_names(bv)
    assert {"list_cameras", "capture"} <= set(names)


def test_list_cameras_mock_without_config(bv):
    err, content = call(bv, "list_cameras")
    assert not err
    data = json.loads(text_of(content))
    assert data["mode"] == "mock"
    cams = {c["name"]: c for c in data["cameras"]}
    assert set(cams) == {"scope", "side"}
    assert cams["side"]["resolution"] == [3840, 2160]
    assert cams["scope"]["status"] == "closed"
    assert cams["scope"]["connected"] is True


def test_capture_returns_small_jpeg_and_saves(bv, root):
    err, content = call(bv, "capture", cam="side")
    assert not err
    [img] = images_of(content)
    assert jpeg_size(img) == (1024, 576)
    text = text_of(content)
    assert "3840x2160" in text and "image * 3.750" in text and "y = y * 3.750" in text
    saved = sorted((root / "captures").rglob("*.jpg"))
    assert [p.name for p in saved] == ["side_120001.jpg"]
    assert saved[0].parent.name == "2026-10-04"
    assert jpeg_size(saved[0].read_bytes()) == (3840, 2160)  # full resolution on disk
    meta = json.loads(saved[0].with_suffix(".json").read_text())
    assert meta["camera"] == "side"
    assert meta["crop_box"] is None
    assert meta["full_res_size"] == [3840, 2160]
    assert meta["max_edge"] == 1024
    assert meta["returned_size"] == [1024, 576]
    assert "controls" in meta


def test_capture_rotation(bv):
    err, content = call(bv, "capture", cam="scope", rotate=90)
    assert not err
    assert jpeg_size(images_of(content)[0]) == (576, 1024)
    err, content = call(bv, "capture", cam="scope", rotate=45)
    assert err and "multiple of 90" in text_of(content)


def test_capture_max_edge(bv):
    err, content = call(bv, "capture", cam="scope", max_edge=512)
    assert jpeg_size(images_of(content)[0]) == (512, 288)
    err, content = call(bv, "capture", cam="scope", max_edge=10)
    assert err and "max_edge" in text_of(content)


def test_unknown_camera_is_clear(bv):
    err, content = call(bv, "capture", cam="overhead")
    assert err
    msg = text_of(content)
    assert "Unknown camera 'overhead'" in msg and "scope" in msg and "Traceback" not in msg


def test_default_rotation_from_config(root, clock):
    (root / "config.toml").write_text(
        '[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\nresolution = [1920, 1080]\n'
        "default_rotation = 270\n"
    )
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "capture", cam="scope")
    assert not err
    assert jpeg_size(images_of(content)[0]) == (576, 1024)


def test_configured_camera_without_mock_image(root, clock):
    (root / "config.toml").write_text(
        '[cameras.overhead]\ndevice = "/dev/v4l/by-id/overhead-video-index0"\n'
    )
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "capture", cam="overhead")
    assert err and "Mock camera 'overhead' has no image" in text_of(content)
    err, content = call(bv, "list_cameras")
    assert json.loads(text_of(content))["cameras"][0]["connected"] is False


def test_config_typo_reported_by_every_tool(root, clock):
    (root / "config.toml").write_text('[cameras.scope]\ndevcie = "/dev/v4l/by-id/x"\n')
    bv = BenchVision(root, mock=True, clock=clock)
    for tool, args in [("list_cameras", {}), ("capture", {"cam": "scope"})]:
        err, content = call(bv, tool, **args)
        assert err
        msg = text_of(content)
        assert "unknown key 'devcie'" in msg and "did you mean 'device'" in msg
        assert "Traceback" not in msg


def test_live_mode_missing_config(tmp_path, clock):
    bv = BenchVision(tmp_path, mock=False, clock=clock)
    err, content = call(bv, "list_cameras")
    assert err and "No config file" in text_of(content)


def test_mock_mode_without_images(tmp_path, clock):
    bv = BenchVision(tmp_path, mock=True, clock=clock)
    err, content = call(bv, "list_cameras")
    assert err and "has no images" in text_of(content)


def test_missing_real_device_names_config_entry(tmp_path, clock):
    (tmp_path / "config.toml").write_text(
        '[cameras.side]\ndevice = "/dev/v4l/by-id/usb-Arducam_not_plugged_in-video-index0"\n'
    )
    bv = BenchVision(tmp_path, mock=False, clock=clock)
    err, content = call(bv, "capture", cam="side")
    msg = text_of(content)
    assert err and "Camera 'side' is not connected" in msg and "[cameras.side]" in msg
    err, content = call(bv, "list_cameras")
    assert not err and json.loads(text_of(content))["cameras"][0]["connected"] is False


def test_unexpected_exception_has_no_traceback(bv, monkeypatch):
    def boom(*a, **k):
        raise ZeroDivisionError("kaboom")

    monkeypatch.setattr(bv.cameras.backend, "grab", boom)
    err, content = call(bv, "capture", cam="scope")
    msg = text_of(content)
    assert err and "ZeroDivisionError" in msg and "kaboom" not in msg and "Traceback" not in msg


def test_bad_argument_type_is_clear(bv):
    err, content = call(bv, "capture", cam="scope", rotate="sideways")
    assert err and "rotate" in text_of(content)


class SlowCountingBackend:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def device_present(self, cam):
        return True

    def grab(self, cam, controls):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return Frame(cam.name, np.zeros((10, 10, 3), np.uint8), {})


def test_never_two_cameras_open_at_once():
    backend = SlowCountingBackend()
    cams = {n: CameraConfig(n, f"/dev/v4l/by-id/{n}") for n in ("scope", "side", "overhead")}
    mgr = CameraManager(cams, backend)
    threads = [threading.Thread(target=mgr.capture, args=(n,)) for n in list(cams) * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert backend.max_active == 1
    assert mgr.open_camera is None


def test_open_camera_reset_after_error():
    class Failing(SlowCountingBackend):
        def grab(self, cam, controls):
            raise CameraError("unplugged")

    mgr = CameraManager({"a": CameraConfig("a", "/dev/v4l/by-id/a")}, Failing())
    with pytest.raises(CameraError):
        mgr.capture("a")
    assert mgr.open_camera is None


def test_captures_in_same_second_do_not_overwrite(root):
    from datetime import datetime

    bv = BenchVision(root, mock=True, clock=lambda: datetime(2026, 1, 2, 3, 4, 5))
    bv.capture("scope")
    bv.capture("scope")
    names = sorted(p.name for p in (root / "captures" / "2026-01-02").glob("*.jpg"))
    assert names == ["scope_030405.jpg", "scope_030405_1.jpg"]


def test_capture_survives_unwritable_captures_dir(root, clock):
    (root / "captures").write_text("not a directory")
    bv = BenchVision(root, mock=True, clock=clock)
    result = bv.capture("scope")
    assert isinstance(result[0], bytes)
    assert "WARNING: this capture was not saved" in result[1]


def test_repo_mock_images_work(clock, tmp_path):
    repo = Path(__file__).resolve().parent.parent
    import shutil

    shutil.copytree(repo / "mock", tmp_path / "mock")
    bv = BenchVision(tmp_path, mock=True, clock=clock)
    for cam in ("scope", "side"):
        imgs = [x for x in bv.capture(cam) if isinstance(x, bytes)]
        assert max(jpeg_size(imgs[0])) == 1024


def test_float_default_rotation_rejected_at_load(root, clock):
    (root / "config.toml").write_text(
        '[cameras.scope]\ndevice = "/dev/v4l/by-id/scope-video-index0"\ndefault_rotation = 90.0\n'
    )
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "capture", cam="scope")
    assert err and "default_rotation" in text_of(content)


def test_mock_accepts_any_config_controls(root, clock):
    (root / "config.toml").write_text(
        '[cameras.side]\ndevice = "/dev/v4l/by-id/side-video-index0"\nresolution = [3840, 2160]\n'
        'v4l2_controls = ["white_balance_automatic=0", "focus_automatic_continuous=0", "focus_absolute=2000"]\n'
    )
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "capture", cam="side")
    assert not err, text_of(content)
    meta = json.loads(next((root / "captures").rglob("*.json")).read_text())
    assert meta["controls"] == {"white_balance_automatic": 0, "focus_automatic_continuous": 0, "focus_absolute": 2000}


def test_concurrent_saves_same_second_keep_every_file(root):
    from datetime import datetime

    from bench_vision.storage import CaptureStore

    store = CaptureStore(root / "captures", clock=lambda: datetime(2026, 1, 1, 1, 1, 1))
    paths = []
    threads = [threading.Thread(target=lambda: paths.append(store.save("scope", b"x" * 100, {}))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(paths)) == 8
    assert len(list((root / "captures" / "2026-01-01").glob("*.jpg"))) == 8


@pytest.mark.skipif(__import__("os").geteuid() == 0, reason="root ignores permissions")
def test_unreadable_mock_dir_is_clear_error(root, clock):
    import os

    os.chmod(root / "mock", 0)
    try:
        bv = BenchVision(root, mock=True, clock=clock)
        err, content = call(bv, "list_cameras")
    finally:
        os.chmod(root / "mock", 0o755)
    msg = text_of(content)
    assert err and "Could not read mock directory" in msg and "Traceback" not in msg


def test_argument_validation_errors_are_short(bv):
    err, content = call(bv, "capture", cam="scope", rotate="sideways")
    msg = text_of(content)
    assert err and msg.startswith("Invalid arguments for capture: 'rotate'")
    assert "pydantic" not in msg and "\n" not in msg
    err, content = call(bv, "capture")
    assert err and "'cam' is required" in text_of(content)


def test_mock_names_must_be_valid_camera_names(root, clock):
    import shutil

    shutil.copy(root / "mock" / "scope.png", root / "mock" / "My Cam.png")
    bv = BenchVision(root, mock=True, clock=clock)
    assert set(bv.cameras.cameras) == {"scope", "side"}


@pytest.mark.skipif(__import__("os").geteuid() == 0, reason="root ignores permissions")
def test_unsearchable_mock_dir_is_clear_error(root, clock):
    import os

    os.chmod(root / "mock", 0o444)
    try:
        bv = BenchVision(root, mock=True, clock=clock)
        err, content = call(bv, "list_cameras")
    finally:
        os.chmod(root / "mock", 0o755)
    msg = text_of(content)
    assert err and "Could not read mock directory" in msg and "Errno" not in msg


def test_errors_name_the_actual_config_file(tmp_path, clock):
    cfg = tmp_path / "bench.toml"
    cfg.write_text('[cameras.side]\ndevice = "/dev/v4l/by-id/usb-nothing-video-index0"\n')
    bv = BenchVision(tmp_path, config_path=cfg, clock=clock)
    err, content = call(bv, "capture", cam="side")
    assert err and "bench.toml [cameras.side]" in text_of(content)


def test_truncated_mock_jpeg_is_rejected(root, clock):
    import cv2

    ok, jpg = cv2.imencode(".jpg", cv2.imread(str(root / "mock" / "side.png")))
    (root / "mock" / "side.png").unlink()
    (root / "mock" / "side.jpg").write_bytes(jpg.tobytes()[: len(jpg) // 3])
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "capture", cam="side")
    assert err and "could not be decoded" in text_of(content)


@pytest.mark.skipif(__import__("os").geteuid() == 0, reason="root ignores permissions")
def test_unreadable_mock_image_listed_as_disconnected(root, clock):
    import os

    os.chmod(root / "mock" / "scope.png", 0)
    try:
        bv = BenchVision(root, mock=True, clock=clock)
        err, content = call(bv, "list_cameras")
    finally:
        os.chmod(root / "mock" / "scope.png", 0o644)
    cams = {c["name"]: c for c in json.loads(text_of(content))["cameras"]}
    assert cams["scope"]["connected"] is False and cams["side"]["connected"] is True


def test_failed_sidecar_removes_image(root, monkeypatch):
    from datetime import datetime

    from bench_vision.storage import CaptureStore

    store = CaptureStore(root / "captures", clock=lambda: datetime(2026, 1, 1, 1, 1, 1))
    real = Path.write_text

    def fail(self, *a, **k):
        if self.suffix == ".json":
            raise OSError(28, "No space left on device")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError):
        store.save("scope", b"x", {})
    assert not list((root / "captures").rglob("*.jpg"))


def test_empty_subfolder_next_to_image_and_uppercase_ext(root, clock):
    (root / "mock" / "scope").mkdir()
    (root / "mock" / "side.png").rename(root / "mock" / "side.PNG")
    bv = BenchVision(root, mock=True, clock=clock)
    assert set(bv.cameras.cameras) == {"scope", "side"}
    assert bv.config_error is None
    assert isinstance(bv.capture("scope")[0], bytes)
    assert isinstance(bv.capture("side")[0], bytes)


def test_mock_with_missing_explicit_config_is_an_error(root, clock):
    bv = BenchVision(root, mock=True, config_path=root / "nothere.toml", clock=clock)
    err, content = call(bv, "list_cameras")
    assert err and "nothere.toml" in text_of(content)


@pytest.mark.skipif(__import__("os").geteuid() == 0, reason="root ignores permissions")
def test_permission_problem_not_reported_as_corrupt(root, clock):
    import os

    bv = BenchVision(root, mock=True, clock=clock)
    os.chmod(root / "mock" / "scope.png", 0)
    try:
        err, content = call(bv, "capture", cam="scope")
    finally:
        os.chmod(root / "mock" / "scope.png", 0o644)
    msg = text_of(content)
    assert err and ("not readable" in msg or "no image" in msg) and "corrupt" not in msg


@pytest.mark.skipif(__import__("os").geteuid() == 0, reason="root ignores permissions")
def test_one_unreadable_mock_folder_does_not_disable_others(root, clock):
    import os

    (root / "mock" / "overhead").mkdir()
    os.chmod(root / "mock" / "overhead", 0o111)
    try:
        bv = BenchVision(root, mock=True, clock=clock)
    finally:
        os.chmod(root / "mock" / "overhead", 0o755)
    assert bv.config_error is None
    assert set(bv.cameras.cameras) == {"scope", "side"}


def test_unexpected_startup_error_keeps_server_up(root, clock, monkeypatch):
    import bench_vision.app as app

    monkeypatch.setattr(app, "_mock_config", lambda m: 1 / 0)
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "list_cameras")
    msg = text_of(content)
    assert err and "ZeroDivisionError" in msg and "Traceback" not in msg


def test_long_names_not_echoed_in_full(bv):
    err, content = call(bv, "capture", cam="x" * 5000)
    assert err and len(text_of(content)) < 300


def test_undecodable_mock_image_has_unknown_resolution(root, clock):
    (root / "mock" / "scope.png").write_bytes(b"garbage")
    bv = BenchVision(root, mock=True, clock=clock)
    cams = {c["name"]: c for c in json.loads(bv.list_cameras())["cameras"]}
    assert cams["scope"]["resolution"] is None
    assert cams["side"]["resolution_source"] == "mock image"


def test_opencv_errors_become_plain_sentences(tmp_path, clock, monkeypatch):
    import cv2

    import bench_vision.camera as camera

    dev = tmp_path / "fake-video"
    dev.write_text("")
    cam = CameraConfig("side", str(dev))

    class Cap:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return True

        def set(self, *a):
            return True

        def read(self):
            raise cv2.error("OpenCV(5.0) /src/cap_v4l.cpp:123: error: (-2) VIDIOC_DQBUF failed")

        def release(self):
            pass

    monkeypatch.setattr(camera.cv2, "VideoCapture", Cap)
    with pytest.raises(CameraError) as ei:
        OpenCVBackend().grab(cam, [])
    msg = str(ei.value)
    assert "stopped delivering frames" in msg and "cap_v4l" not in msg and "OpenCV(" not in msg


def test_mock_missing_explicit_config_hint(root, clock):
    bv = BenchVision(root, mock=True, config_path=root / "nope.toml", clock=clock)
    err, content = call(bv, "capture", cam="scope")
    msg = text_of(content)
    assert err and "Fix the --config path" in msg and "start the server with `--mock`" not in msg


def test_empty_frames_count_as_no_frames(tmp_path, monkeypatch):
    import bench_vision.camera as camera

    dev = tmp_path / "fake-video"
    dev.write_text("")

    class Cap:
        def __init__(self, *a): pass
        def isOpened(self): return True
        def set(self, *a): return True
        def read(self): return True, np.zeros((0, 0, 3), np.uint8)
        def release(self): pass

    monkeypatch.setattr(camera.cv2, "VideoCapture", Cap)
    with pytest.raises(CameraError, match="returned no frames"):
        OpenCVBackend().grab(CameraConfig("side", str(dev)), [])


def test_saved_capture_is_full_res_rotated_and_records_max_edge(bv, root):
    err, content = call(bv, "capture", cam="scope", rotate=90, max_edge=512)
    assert not err
    assert jpeg_size(images_of(content)[0]) == (288, 512)
    [saved] = list((root / "captures").rglob("*.jpg"))
    assert jpeg_size(saved.read_bytes()) == (1080, 1920)
    meta = json.loads(saved.with_suffix(".json").read_text())
    assert meta["full_res_size"] == [1080, 1920]
    assert meta["max_edge"] == 512 and meta["returned_size"] == [288, 512]
    assert meta["rotation"] == 90
