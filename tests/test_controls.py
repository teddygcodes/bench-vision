import io
import json
import sys
import threading
import time

import numpy as np
import pytest

from bench_vision import camera as camera_mod
from bench_vision.app import BenchVision
from bench_vision.camera import CameraManager, OpenCVBackend
from bench_vision.config import CameraConfig, load_config
from bench_vision.errors import CameraError
from bench_vision.setup import run_setup
from bench_vision.v4l2 import V4L2

from conftest import call, list_tool_names, text_of
from fake_v4l2 import SCOPE_ID, SIDE_ID, FakeV4L2Ctl, make_by_id

not_linux = pytest.mark.skipif(sys.platform.startswith("linux"), reason="checks the non-Linux message")


# ------------------------------------------------------------------ setup


def test_setup_report_and_config(tmp_path):
    by_id = make_by_id(tmp_path)
    out = io.StringIO()
    code = run_setup(tmp_path, v4l2=V4L2(FakeV4L2Ctl()), by_id_dir=by_id, out=out)
    report = out.getvalue()
    assert code == 0
    assert report.startswith("===== bench-vision setup report") and report.rstrip().endswith("===== end of report =====")
    assert "v4l2-ctl 1.26.1" in report and "TOMLOV PC Camera" in report  # --list-devices verbatim
    assert f"{by_id}/{SCOPE_ID} -> " in report and f"{by_id}/{SIDE_ID} -> " in report
    assert "MJPG (Motion-JPEG, compressed)" in report and "3840x2160  15 fps" in report
    assert "focus_automatic_continuous" in report and "1: Manual Mode" in report  # controls incl. menus
    assert "no capture formats (probably a metadata node" in report
    cfg = load_config(tmp_path / "config.toml")
    assert set(cfg.cameras) == {"scope", "side"}
    side, scope = cfg.cameras["side"], cfg.cameras["scope"]
    assert side.device == f"/dev/v4l/by-id/{SIDE_ID}" and side.resolution == (3840, 2160) and side.fourcc == "MJPG"
    assert side.v4l2_controls == (("focus_automatic_continuous", 0), ("focus_absolute", 312))
    assert scope.resolution == (1920, 1080) and scope.v4l2_controls == ()


def test_setup_keeps_existing_config(tmp_path):
    (tmp_path / "config.toml").write_text("# mine\n")
    out = io.StringIO()
    run_setup(tmp_path, v4l2=V4L2(FakeV4L2Ctl()), by_id_dir=make_by_id(tmp_path), out=out)
    assert (tmp_path / "config.toml").read_text() == "# mine\n"
    assert (tmp_path / "config.generated.toml").exists() and "config.generated.toml" in out.getvalue()
    run_setup(tmp_path, force=True, v4l2=V4L2(FakeV4L2Ctl()), by_id_dir=tmp_path / "by-id", out=io.StringIO())
    assert "[cameras.side]" in (tmp_path / "config.toml").read_text()


def test_setup_with_no_cameras(tmp_path):
    out = io.StringIO()
    code = run_setup(tmp_path, v4l2=V4L2(FakeV4L2Ctl()), by_id_dir=tmp_path / "missing", out=out)
    assert code == 1 and "no USB cameras" in out.getvalue() and not (tmp_path / "config.toml").exists()


def test_setup_reports_a_failing_device(tmp_path):
    fake = FakeV4L2Ctl()
    fake.fail_on = "Arducam"
    out = io.StringIO()
    run_setup(tmp_path, v4l2=V4L2(fake), by_id_dir=make_by_id(tmp_path), out=out)
    assert "error: Listing formats of" in out.getvalue()
    assert set(load_config(tmp_path / "config.toml").cameras) == {"scope"}


def test_unknown_cameras_get_numbered_names(tmp_path):
    by_id = make_by_id(tmp_path, names=("usb-Generic_Webcam-video-index0",))
    fake = FakeV4L2Ctl()
    orig = fake.__call__

    def runner(argv):
        if "--list-formats-ext" in argv:
            from fake_v4l2 import SCOPE_FORMATS

            fake.calls.append(argv)
            return 0, SCOPE_FORMATS, ""
        return orig(argv)

    run_setup(tmp_path, v4l2=V4L2(runner), by_id_dir=by_id, out=io.StringIO())
    assert set(load_config(tmp_path / "config.toml").cameras) == {"cam1"}


@not_linux
def test_setup_cli_on_mac_is_one_line(tmp_path, capsys):
    from bench_vision.cli import main

    assert main(["setup", "--root", str(tmp_path)]) == 2
    err = capsys.readouterr().err.strip()
    assert "\n" not in err and "needs Linux/v4l2" in err and not (tmp_path / "config.toml").exists()


# ------------------------------------------------------------ set_control


def test_set_control_tool_registered(bv):
    assert "set_control" in list_tool_names(bv)


def test_set_control_mock_flow(bv, root):
    err, content = call(bv, "set_control", cam="side", control="focus_absolute", value=300)
    assert err and "inactive" in text_of(content) and "focus_automatic_continuous=0 first" in text_of(content)
    err, content = call(bv, "set_control", cam="side", control="focus_automatic_continuous", value=0)
    assert not err and "set to 0 (was 1" in text_of(content)
    err, content = call(bv, "set_control", cam="side", control="focus_absolute", value=300)
    text = text_of(content)
    assert not err and "focus_absolute set to 300 (was 0; range 0..1023)" in text and '"focus_absolute=300"' in text
    call(bv, "capture", cam="side")  # overrides are re-applied on every open
    meta = json.loads(next((root / "captures").rglob("*.json")).read_text())
    assert meta["controls"] == {"focus_automatic_continuous": 0, "focus_absolute": 300}


def test_reenabling_auto_drops_manual_override(bv):
    bv.set_control("side", "focus_automatic_continuous", 0)
    bv.set_control("side", "focus_absolute", 300)
    text = bv.set_control("side", "focus_automatic_continuous", 1)
    assert "stopped re-applying focus_absolute" in text
    assert bv.cameras.controls_for(bv.cameras.get("side")) == [("focus_automatic_continuous", 1)]


@pytest.mark.parametrize(
    "args,msg",
    [
        (dict(cam="side", control="focus_absolut", value=3), "Did you mean: focus_absolute"),
        (dict(cam="side", control="gain", value=500), "out of range 0..100"),
        (dict(cam="side", control="focus_automatic_continuous", value=2), "bool control"),
        (dict(cam="side", control="Gain; rm -rf", value=1), "must be a v4l2 control name"),
        (dict(cam="nope", control="gain", value=1), "Unknown camera 'nope'"),
        (dict(cam="side", control="gain", value="lots"), "Invalid arguments for set_control"),
    ],
)
def test_set_control_errors(bv, args, msg):
    err, content = call(bv, "set_control", **args)
    assert err and msg in text_of(content) and "Traceback" not in text_of(content)


@not_linux
def test_set_control_live_on_mac_needs_linux(tmp_path, clock):
    (tmp_path / "config.toml").write_text(f'[cameras.side]\ndevice = "/dev/v4l/by-id/{SIDE_ID}"\n')
    bv = BenchVision(tmp_path, clock=clock)
    err, content = call(bv, "set_control", cam="side", control="gain", value=5)
    msg = text_of(content)
    assert err and "needs Linux/v4l2" in msg and "\n" not in msg


def _live(tmp_path, clock, fake=None):
    by_id = make_by_id(tmp_path)
    (tmp_path / "config.toml").write_text(
        f'[cameras.side]\ndevice = "/dev/v4l/by-id/{SIDE_ID}"\nresolution = [3840, 2160]\n'
    )
    fake = fake or FakeV4L2Ctl()
    bv = BenchVision(tmp_path, clock=clock, backend=OpenCVBackend(V4L2(fake)))
    # point the configured by-id path at the fake device
    cam = bv.cameras.cameras["side"]
    bv.cameras.cameras["side"] = CameraConfig(**{**cam.__dict__, "device": str(by_id / SIDE_ID)})
    return bv, fake


def test_set_control_live_with_fake_v4l2(tmp_path, clock):
    bv, fake = _live(tmp_path, clock)
    text = bv.set_control("side", "focus_automatic_continuous", 0)
    assert "set to 0 (was 1" in text
    text = bv.set_control("side", "focus_absolute", 450)
    assert "set to 450 (was 312" in text
    sets = [a for a in sum(fake.calls, []) if a.startswith("--set-ctrl=")]
    assert sets == ["--set-ctrl=focus_automatic_continuous=0", "--set-ctrl=focus_absolute=450"]


def test_set_control_live_missing_device_names_config(tmp_path, clock):
    bv, fake = _live(tmp_path, clock)
    (tmp_path / "by-id" / SIDE_ID).unlink()
    with pytest.raises(CameraError, match=r"Camera 'side' is not connected.*\[cameras.side\]"):
        bv.set_control("side", "gain", 5)


def test_inactive_config_control_is_skipped_not_fatal(tmp_path):
    """focus_absolute in config while autofocus is on must not make every capture fail."""
    fake = FakeV4L2Ctl()
    backend = OpenCVBackend(V4L2(fake))
    by_id = make_by_id(tmp_path)
    cam = CameraConfig("side", str(by_id / SIDE_ID), v4l2_controls=(("focus_absolute", 300),))
    assert backend._apply_controls(cam, list(cam.v4l2_controls)) == {}
    assert not any(a.startswith("--set-ctrl") for a in sum(fake.calls, []))


# --------------------------------------------------------- capture timeouts


class SlowCap:
    open_delay = 0.0
    read_delay = 0.0

    def __init__(self, *a):
        time.sleep(self.open_delay)

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        time.sleep(self.read_delay)
        return True, np.zeros((8, 8, 3), np.uint8)

    def release(self):
        pass


@pytest.fixture
def fast_timeouts(monkeypatch):
    monkeypatch.setattr(camera_mod, "OPEN_TIMEOUT", 0.3)
    monkeypatch.setattr(camera_mod, "FRAME_TIMEOUT", 0.3)
    monkeypatch.setattr(camera_mod, "RELEASE_TIMEOUT", 0.3)


def _manager(tmp_path, cap_cls, monkeypatch):
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", cap_cls)
    devs = {}
    for n in ("scope", "side"):
        p = tmp_path / f"dev-{n}"
        p.write_text("")
        devs[n] = CameraConfig(n, str(p), resolution=(8, 8))
    return CameraManager(devs, OpenCVBackend(V4L2(FakeV4L2Ctl())))


@pytest.mark.parametrize("stage,attr,what", [("open", "open_delay", "did not open within"),
                                             ("read", "read_delay", "did not deliver a frame within")])
def test_capture_timeouts_release_lock_and_name_camera(tmp_path, monkeypatch, fast_timeouts, stage, attr, what):
    cap = type("Cap", (SlowCap,), {attr: 1.0})
    mgr = _manager(tmp_path, cap, monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(CameraError, match=f"Camera 'side' .*{what} 0.3 s"):
        mgr.capture("side")
    assert time.monotonic() - t0 < 1.5
    assert mgr.open_camera is None and not mgr._lock.locked()
    # while the hung device is still busy, no other camera may open (never two at once)...
    with pytest.raises(CameraError, match="Camera 'side' is still stuck"):
        mgr.capture("scope")
    # ...and once it lets go, captures work again
    time.sleep(1.2)
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", SlowCap)
    assert mgr.capture("scope").size == (8, 8)


def test_normal_capture_is_unaffected_by_timeouts(tmp_path, monkeypatch, fast_timeouts):
    mgr = _manager(tmp_path, type("Cap", (SlowCap,), {"read_delay": 0.05}), monkeypatch)
    assert mgr.capture("side").size == (8, 8)
    assert mgr.capture("scope").size == (8, 8)


def test_non_utf8_v4l2_output_does_not_crash(tmp_path, monkeypatch):
    """uvcvideo can truncate a product name mid-character; the report must still come out."""
    import subprocess as sp

    from bench_vision import v4l2 as v4l2_mod

    def fake_run(argv, **kw):
        assert kw.get("errors") == "replace"
        raw = b"\xe6\x91 Camera (usb-1):\n\t/dev/video0\n"
        return sp.CompletedProcess(argv, 0, raw.decode("utf-8", errors=kw["errors"]), "")

    monkeypatch.setattr(v4l2_mod, "require_v4l2", lambda: None)
    monkeypatch.setattr(v4l2_mod.subprocess, "run", fake_run)
    code, out, _ = v4l2_mod.subprocess_runner(["v4l2-ctl", "--list-devices"])
    assert code == 0 and "Camera" in out and "\ufffd" in out


def test_flags_with_several_values_parse():
    from bench_vision.v4l2 import parse_list_ctrls

    c = parse_list_ctrls("   focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=0 value=5 flags=update, inactive\n")
    assert c["focus_absolute"].inactive


def test_list_devices_with_no_devices_still_reports(tmp_path):
    def runner(argv):
        if argv[1:] == ["--list-devices"]:
            return 1, "", "Cannot open device /dev/video0, exiting."
        return FakeV4L2Ctl()(argv)

    out = io.StringIO()
    code = run_setup(tmp_path, v4l2=V4L2(runner), by_id_dir=tmp_path / "none", out=out)
    assert code == 1 and "found nothing: Cannot open device" in out.getvalue()


def test_timeout_reply_is_not_delayed_by_release_wait(tmp_path, monkeypatch, fast_timeouts):
    mgr = _manager(tmp_path, type("Cap", (SlowCap,), {"read_delay": 2.0}), monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(CameraError):
        mgr.capture("side")
    assert time.monotonic() - t0 < 0.55  # ~one 0.3 s timeout, not timeout + release wait


def test_controls_for_snapshot_under_concurrent_edits():
    mgr = CameraManager({"side": CameraConfig("side", "/dev/v4l/by-id/x")}, None)
    stop = threading.Event()

    def edit():
        i = 0
        while not stop.is_set():
            d = mgr.overrides.setdefault("side", {})
            d[f"c{i % 50}"] = i
            d.pop(f"c{(i + 25) % 50}", None)
            i += 1

    t = threading.Thread(target=edit)
    t.start()
    try:
        for _ in range(20000):
            mgr.controls_for(mgr.cameras["side"])
    finally:
        stop.set()
        t.join()


@pytest.mark.parametrize(
    "config,overrides,expect",
    [
        (("exposure_time_absolute=200",), [("auto_exposure", 1), ("exposure_time_absolute", 100)],
         [("auto_exposure", 1), ("exposure_time_absolute", 100)]),
        (("focus_absolute=300",), [("focus_automatic_continuous", 0), ("focus_absolute", 450)],
         [("focus_automatic_continuous", 0), ("focus_absolute", 450)]),
        (("focus_automatic_continuous=0", "focus_absolute=300", "gain=4"), [("focus_absolute", 10)],
         [("focus_automatic_continuous", 0), ("gain", 4), ("focus_absolute", 10)]),
    ],
)
def test_auto_switches_always_applied_first(config, overrides, expect):
    from bench_vision.config import parse_control

    cam = CameraConfig("side", "/dev/v4l/by-id/x", v4l2_controls=tuple(parse_control(c, "t") for c in config))
    mgr = CameraManager({"side": cam}, None)
    mgr.overrides["side"] = dict(overrides)
    assert mgr.controls_for(cam) == expect


def test_override_reapplied_after_replug(tmp_path, monkeypatch):
    """A replug resets the camera to auto; the next open must still apply the manual value."""
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", SlowCap)
    fake = FakeV4L2Ctl()
    by_id = make_by_id(tmp_path)
    cam = CameraConfig("side", str(by_id / SIDE_ID), resolution=(8, 8), v4l2_controls=(("focus_absolute", 300),))
    mgr = CameraManager({"side": cam}, OpenCVBackend(V4L2(fake)))
    mgr.overrides["side"] = {"focus_automatic_continuous": 0, "focus_absolute": 450}
    fake.values.clear()  # replug
    assert mgr.capture("side").controls == {"focus_automatic_continuous": 0, "focus_absolute": 450}


def test_shutter_priority_keeps_exposure_override(bv):
    bv.set_control("scope", "auto_exposure", 1)
    bv.set_control("scope", "exposure_time_absolute", 100)
    text = bv.set_control("scope", "auto_exposure", 2)
    assert "stopped re-applying" not in text
    assert ("exposure_time_absolute", 100) in bv.cameras.controls_for(bv.cameras.get("scope"))


def test_setup_never_picks_undecodable_formats():
    from bench_vision.setup import best_mode
    from bench_vision.v4l2 import PixelFormat

    fmts = [PixelFormat("H264", "H.264", {(3840, 2160): []}), PixelFormat("NV12", "NV12", {(1280, 720): []})]
    assert best_mode(fmts) == ("NV12", (1280, 720))


def test_setup_locks_focus_with_old_kernel_names(tmp_path):
    from fake_v4l2 import SIDE_CTRLS

    old = SIDE_CTRLS.replace("focus_automatic_continuous", "focus_auto")
    fake = FakeV4L2Ctl()
    base = fake.__call__

    def runner(argv):
        if ("--list-ctrls-menus" in argv or "--list-ctrls" in argv) and SIDE_ID in " ".join(argv):
            return 0, old, ""
        return base(argv)

    run_setup(tmp_path, v4l2=V4L2(runner), by_id_dir=make_by_id(tmp_path), out=io.StringIO())
    side = load_config(tmp_path / "config.toml").cameras["side"]
    assert side.v4l2_controls == (("focus_auto", 0), ("focus_absolute", 312))


def test_old_kernel_auto_switch_ordering_and_dropping():
    cam = CameraConfig("side", "/dev/v4l/by-id/x", v4l2_controls=(("focus_absolute", 300), ("focus_auto", 0)))
    mgr = CameraManager({"side": cam}, None)
    assert mgr.controls_for(cam) == [("focus_auto", 0), ("focus_absolute", 300)]
