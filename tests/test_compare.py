import json

import cv2
from PIL import Image
import numpy as np
import pytest

from bench_vision.app import BenchVision

from conftest import call, images_of, jpeg_size, list_tool_names, make_image, text_of


@pytest.fixture
def changing(root, clock):
    """scope cycles: capture 1 = base, capture 2 = base with a bright patch at x=1200..1400, y=600..700."""
    base = make_image(root / "mock" / "scope" / "1.png", 1920, 1080, seed=1)
    changed = base.copy()
    changed[600:700, 1200:1400] = (255, 255, 255)
    cv2.imwrite(str(root / "mock" / "scope" / "2.png"), changed)
    return BenchVision(root, mock=True, clock=clock)


def test_tools_registered(bv):
    assert {"save_reference", "compare"} <= set(list_tool_names(bv))


def test_save_reference_stores_lossless_full_res(bv, root):
    err, content = call(bv, "save_reference", cam="side", name="u1-before")
    assert not err, text_of(content)
    assert jpeg_size(images_of(content)[0]) == (1024, 576)
    png = root / "references-mock" / "side" / "u1-before.png"
    meta = json.loads(png.with_suffix(".json").read_text())
    assert cv2.imread(str(png)).shape == (2160, 3840, 3)
    assert meta["full_res_size"] == [3840, 2160] and meta["rotation"] == 0 and meta["camera"] == "side"
    assert "saved reference 'u1-before'" in text_of(content)
    [cap] = list((root / "captures").rglob("*.json"))
    assert json.loads(cap.read_text())["tool"] == "save_reference"


def test_compare_identical_frames_shows_no_change(bv):
    call(bv, "save_reference", cam="scope", name="ref")
    err, content = call(bv, "compare", cam="scope", name="ref")
    assert not err, text_of(content)
    pair, heat = images_of(content)
    assert max(jpeg_size(pair)) == 1024 and jpeg_size(heat) == (1024, 576)
    text = text_of(content)
    assert "0 px (0%) changed" in text and "No changed regions" in text


def test_compare_finds_the_changed_region(changing, root):
    call(changing, "save_reference", cam="scope", name="before")
    err, content = call(changing, "compare", cam="scope", name="before")
    assert not err, text_of(content)
    text = text_of(content)
    meta = [json.loads(p.read_text()) for p in sorted((root / "captures").rglob("*.json"))][-1]
    assert meta["tool"] == "compare" and meta["reference"] == "before"
    x, y, w, h = meta["diff"]["regions"][0]
    # padded with context (25% + 16 px a side) around the 200x100 patch at (1200, 600)
    assert abs(x - 1134) <= 6 and abs(y - 559) <= 6 and abs(w - 332) <= 12 and abs(h - 182) <= 12
    assert x <= 1200 and y <= 600 and x + w >= 1400 and y + h >= 700
    assert f"x={x}, y={y}, w={w}, h={h}" in text
    assert meta["diff"]["changed_pct"] > 0.5
    # heatmap is hot where the patch is and cool elsewhere
    import io

    from PIL import Image

    heat = np.asarray(Image.open(io.BytesIO(images_of(content)[1])).convert("RGB")).astype(int)
    patch = heat[600 * 576 // 1080 + 5 : 700 * 576 // 1080 - 5, 1200 * 1024 // 1920 + 5 : 1400 * 1024 // 1920 - 5]
    corner = heat[20:60, 20:60]
    assert patch[..., 0].mean() > corner[..., 0].mean() + 60  # red channel


def test_compare_uses_reference_rotation(bv, root):
    call(bv, "save_reference", cam="scope", name="rot", rotate=90)
    err, content = call(bv, "compare", cam="scope", name="rot")
    assert not err and "rotation 90°" in text_of(content) and "No changed regions" in text_of(content)


def test_compare_missing_reference_lists_available(bv):
    err, content = call(bv, "compare", cam="scope", name="nope")
    assert err and "No references saved for 'scope' yet" in text_of(content)
    call(bv, "save_reference", cam="scope", name="a")
    call(bv, "save_reference", cam="scope", name="b")
    err, content = call(bv, "compare", cam="scope", name="nope")
    assert err and "Saved references for 'scope': a, b." in text_of(content)
    err, content = call(bv, "compare", cam="side", name="a")  # references are per camera
    assert err and "No reference named 'a' for camera 'side'" in text_of(content)


@pytest.mark.parametrize("name", ["", "../escape", "a/b", ".hidden", "x" * 65, "a b", "..", "naïve", "abc\n"])
def test_bad_reference_names(bv, root, name):
    for tool in ("save_reference", "compare"):
        err, content = call(bv, tool, cam="scope", name=name)
        assert err and "Reference name must be" in text_of(content), (tool, name)
    assert not (root / "references-mock").exists() or not any((root / "references-mock").rglob("*.png"))


def test_replacing_a_reference_says_so(bv):
    call(bv, "save_reference", cam="scope", name="r")
    text = text_of(call(bv, "save_reference", cam="scope", name="r")[1])
    assert "Replaced the reference saved 2026-10-04T12:00:" in text


def test_resolution_change_is_clear(root, clock):
    bv = BenchVision(root, mock=True, clock=clock)
    bv.save_reference("scope", "r")
    (root / "mock" / "scope.png").unlink()
    make_image(root / "mock" / "scope.png", 960, 540, seed=1)
    err, content = call(bv, "compare", cam="scope", name="r")
    assert err and "is 1920x1080 but 'scope' now delivers 960x540" in text_of(content)


def test_corrupt_reference_files(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    (root / "references-mock" / "scope" / "r.png").write_bytes(b"junk")
    err, content = call(bv, "compare", cam="scope", name="r")
    assert err and "don't match" in text_of(content)
    call(bv, "save_reference", cam="scope", name="s")
    (root / "references-mock" / "scope" / "s.json").write_text("{bad")
    err, content = call(bv, "compare", cam="scope", name="s")
    assert err and "corrupt .json sidecar" in text_of(content)
    call(bv, "save_reference", cam="scope", name="s")
    js = root / "references-mock" / "scope" / "s.json"
    js.write_text(json.dumps({**json.loads(js.read_text()), "rotation": 45}))
    err, content = call(bv, "compare", cam="scope", name="s")
    assert err and "don't match" in text_of(content)  # edited metadata breaks the pair checksum


def test_unwritable_references_dir_is_clear(root, clock):
    (root / "references-mock").write_text("not a dir")
    bv = BenchVision(root, mock=True, clock=clock)
    err, content = call(bv, "save_reference", cam="scope", name="r")
    msg = text_of(content)
    assert err and "Could not save reference 'r'" in msg and "Traceback" not in msg


def test_errors_before_camera_opens(bv):
    opened = []
    real = bv.cameras.capture
    bv.cameras.capture = lambda n: opened.append(n) or real(n)
    for tool, args in [("save_reference", dict(cam="scope", name="../x")),
                       ("compare", dict(cam="scope", name="missing")),
                       ("compare", dict(cam="nope", name="r")),
                       ("save_reference", dict(cam="scope", name="r", max_edge=1))]:
        err, _ = call(bv, tool, **args)
        assert err
    assert opened == []


def test_broken_config_first(root, clock):
    (root / "config.toml").write_text("[cameras.scope\n")
    bv = BenchVision(root, mock=True, clock=clock)
    for tool in ("save_reference", "compare"):
        err, content = call(bv, tool, cam="nope", name="../bad")
        assert err and "not valid TOML" in text_of(content)


def test_region_hint_carries_rotation(changing):
    call(changing, "save_reference", cam="scope", name="r", rotate=90)
    text = text_of(call(changing, "compare", cam="scope", name="r")[1])
    assert "rotate=90)" in text and "1080x1920 frame" in text


def test_reference_sidecar_mismatch_is_detected(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    js = root / "references-mock" / "scope" / "r.json"
    meta = json.loads(js.read_text())
    meta["full_res_size"] = [1080, 1920]  # e.g. a half-finished concurrent replace
    js.write_text(json.dumps(meta))
    err, content = call(bv, "compare", cam="scope", name="r")
    assert err and "don't match" in text_of(content)


def test_concurrent_save_and_compare_same_name(bv):
    import threading

    call(bv, "save_reference", cam="scope", name="r")
    errors = []

    def saver(rot):
        try:
            bv.save_reference("scope", "r", rotate=rot)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def comparer():
        try:
            bv.compare("scope", "r")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=saver, args=((i % 2) * 180,)) for i in range(6)]
    threads += [threading.Thread(target=comparer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_mock_and_live_references_are_separate(root, clock):
    BenchVision(root, mock=True, clock=clock).save_reference("scope", "r")
    assert (root / "references-mock" / "scope" / "r.png").exists()
    assert not (root / "references").exists()


def test_reference_from_other_device_warns(root, clock):
    cfg = '[cameras.scope]\ndevice = "/dev/v4l/by-id/%s-video-index0"\nresolution = [1920, 1080]\n'
    (root / "config.toml").write_text(cfg % "a")
    BenchVision(root, mock=True, clock=clock).save_reference("scope", "r")
    (root / "config.toml").write_text(cfg % "b")
    text = BenchVision(root, mock=True, clock=clock).compare("scope", "r")[2]
    assert "WARNING: this reference was saved from device" in text


def test_reference_files_are_world_readable(bv, root):
    import stat

    call(bv, "save_reference", cam="scope", name="r")
    mode = (root / "references-mock" / "scope" / "r.png").stat().st_mode
    assert stat.S_IMODE(mode) == 0o644


def test_thin_change_hint_is_usable_by_capture_region(root, clock):
    base = make_image(root / "mock" / "scope" / "1.png", 1920, 1080, seed=1)
    changed = base.copy()
    changed[500:540, 900:904] = (255, 255, 255)  # a 4 px wide bridge
    cv2.imwrite(str(root / "mock" / "scope" / "2.png"), changed)
    bv = BenchVision(root, mock=True, clock=clock)
    bv.save_reference("scope", "r")
    bv.compare("scope", "r")
    meta = [json.loads(p.read_text()) for p in sorted((root / "captures").rglob("*.json"))][-1]
    x, y, w, h = meta["diff"]["regions"][0]
    assert w >= 64 and h >= 64 and x <= 900 and x + w >= 904
    err, content = call(bv, "capture_region", cam="scope", x=x, y=y, w=w, h=h)
    assert not err, text_of(content)


def test_side_by_side_labels_stay_in_their_half():
    from PIL import Image as PILImage

    from bench_vision import imaging

    img = PILImage.new("RGB", (300, 500), (0, 0, 0))
    out = np.asarray(imaging.side_by_side(img, img, ("REFERENCE '" + "x" * 64 + "'", ""), 4096).convert("RGB"))
    strip = out[:30]
    yellow = (strip[..., 0] > 200) & (strip[..., 1] > 180) & (strip[..., 2] < 80)
    assert yellow[:, :300].any() and not yellow[:, 300:].any()


def test_hostile_sidecar_strings_are_bounded(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    js = root / "references-mock" / "scope" / "r.json"
    meta = json.loads(js.read_text())
    meta["saved_at"] = "Z" * 5000
    js.write_text(json.dumps(meta))
    assert len(text_of(call(bv, "compare", cam="scope", name="r")[1])) < 1500


def _diff(ref, cur):
    from bench_vision import imaging

    return imaging.diff_heatmap(ref, cur, 1024)[1]


def _board(w, h):
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), (40, 95, 30), np.uint8)
    return img, rng


def test_thin_bridge_at_4k_is_found():
    ref, _ = _board(3840, 2160)
    cur = ref.copy()
    cur[1000:1002, 1500:1700] = 255  # 2 px wide bridge
    stats = _diff(ref, cur)
    assert stats["region_count"] == 1
    x, y, w, h = stats["regions"][0]
    assert x <= 1500 and x + w >= 1700 and y <= 1000 and y + h >= 1002


def test_one_px_line_is_one_region():
    ref, _ = _board(1920, 1080)
    cur = ref.copy()
    cur[500, 500:800] = 255
    stats = _diff(ref, cur)
    assert stats["region_count"] == 1
    x, y, w, h = stats["regions"][0]
    assert x <= 500 and x + w >= 800


def test_noise_and_small_exposure_shift_stay_quiet():
    ref, rng = _board(1920, 1080)
    noisy = np.clip(ref.astype(int) + rng.normal(0, 8, ref.shape) + 10, 0, 255).astype(np.uint8)
    stats = _diff(ref, noisy)
    assert stats["region_count"] == 0 and stats["changed_pct"] == 0.0


@pytest.mark.skipif(__import__("os").geteuid() == 0, reason="root ignores permissions")
def test_inaccessible_camera_reference_dir_is_clear(bv, root):
    import os

    call(bv, "save_reference", cam="scope", name="r")
    d = root / "references-mock" / "scope"
    os.chmod(d, 0)
    try:
        for tool in ("save_reference", "compare"):
            err, content = call(bv, tool, cam="scope", name="r")
            msg = text_of(content)
            assert err and "internal error" not in msg and ("not accessible" in msg or "Permission denied" in msg), msg
    finally:
        os.chmod(d, 0o755)


def test_names_are_case_insensitive(bv, root):
    text = text_of(call(bv, "save_reference", cam="scope", name="U1-Before")[1])
    assert "'u1-before'" in text
    err, content = call(bv, "compare", cam="scope", name="u1-BEFORE")
    assert not err, text_of(content)
    assert (root / "references-mock" / "scope" / "u1-before.png").exists()


def test_missing_name_listing_is_capped(bv, root):
    import shutil

    call(bv, "save_reference", cam="scope", name="r0")
    d = root / "references-mock" / "scope"
    for i in range(1, 60):
        shutil.copy(d / "r0.png", d / f"r{i}.png")
        shutil.copy(d / "r0.json", d / f"r{i}.json")
    msg = text_of(call(bv, "compare", cam="scope", name="nope")[1])
    assert "and 40 more" in msg and len(msg) < 600


def test_bad_full_res_size_message_is_bounded(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    js = root / "references-mock" / "scope" / "r.json"
    meta = json.loads(js.read_text())
    meta["full_res_size"] = "1920x1080" + "!" * 5000
    js.write_text(json.dumps(meta))
    msg = text_of(call(bv, "compare", cam="scope", name="r")[1])
    assert "don't match" in msg and len(msg) < 400


def test_two_processes_never_see_a_torn_reference(root):
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    script = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from bench_vision.app import BenchVision
        bv = BenchVision(Path({str(root)!r}), mock=True)
        role = sys.argv[1]
        bad = 0
        for i in range(15):
            if role == "save":
                bv.save_reference("scope", "r", rotate=(i % 2) * 180)
            else:
                text = bv.compare("scope", "r")[2]
                bad += "No changed regions" not in text
        print("bad", bad)
    """)
    BenchVisionSetup = __import__("bench_vision.app", fromlist=["BenchVision"]).BenchVision
    BenchVisionSetup(root, mock=True).save_reference("scope", "r")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
    procs = [subprocess.Popen([sys.executable, "-c", script, role], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for role in ("save", "compare", "compare")]
    outs = [p.communicate(timeout=180) for p in procs]
    for out, err in outs:
        assert out.strip() in ("bad 0",), err[-2000:]


def test_heatmap_shows_thin_change_hot_and_outlined():
    import io

    from bench_vision import imaging

    ref, _ = _board(3840, 2160)
    cur = ref.copy()
    cur[1000:1003, 1500:1700] = 255  # 3 px bridge at 4K, under 1 px after downscaling
    heat, stats = imaging.diff_heatmap(ref, cur, 1024)
    px = np.asarray(Image.open(io.BytesIO(imaging.encode_jpeg(heat, 85, full_chroma=True))).convert("RGB")).astype(int)
    line = px[1001 * 576 // 2160 - 1 : 1001 * 576 // 2160 + 2, 1550 * 1024 // 3840 : 1650 * 1024 // 3840]
    assert line[..., 0].max() > 200  # strongly red/yellow, not a faint smear
    assert stats["regions"] and stats["region_count"] == 1


def test_dashed_line_and_close_blobs_give_one_hint():
    ref, _ = _board(1920, 1080)
    cur = ref.copy()
    for x in range(900, 1100, 20):
        cur[500:502, x : x + 10] = 255  # dashes with 10 px gaps
    cur[700:705, 300:305] = 255
    cur[700:705, 320:325] = 255  # two blobs 15 px apart
    stats = _diff(ref, cur)
    boxes = stats["regions"]
    assert len(boxes) == 2  # one hint for the dashed line, one for the two blobs
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or a[1] + a[3] < b[1] or b[1] + b[3] < a[1]


def test_brightness_shift_is_reported_as_a_fact():
    ref, rng = _board(1920, 1080)
    ref = np.clip(ref.astype(int) + rng.normal(0, 4, ref.shape), 0, 255).astype(np.uint8)
    cur = np.clip(ref.astype(int) + 60 + rng.normal(0, 4, ref.shape), 0, 255).astype(np.uint8)
    stats = _diff(ref, cur)
    assert stats["brightness_shift"] >= 55 and stats["changed_pct"] > 90


def test_mild_drift_does_not_hide_rework():
    ref, rng = _board(1920, 1080)
    ref = np.clip(ref.astype(int) + rng.normal(0, 4, ref.shape), 0, 255).astype(np.uint8)
    cur = np.clip(ref.astype(int) + 8 + rng.normal(0, 4, ref.shape), 0, 255).astype(np.uint8)
    for i in range(5):  # five reworked joints
        cur[300:330, 200 + i * 300 : 230 + i * 300] = 230
    stats = _diff(ref, cur)
    assert stats["region_count"] == 5 and len(stats["regions"]) == 3


def test_brightness_note_even_without_regions(changing, root):
    bright = cv2.imread(str(root / "mock" / "scope" / "1.png")).astype(int) + 15
    cv2.imwrite(str(root / "mock" / "scope" / "2.png"), np.clip(bright, 0, 255).astype(np.uint8))
    call(changing, "save_reference", cam="scope", name="r")
    text = text_of(call(changing, "compare", cam="scope", name="r")[1])
    assert "Median brightness is 15 levels higher than the reference" in text


def test_compare_missing_reference_creates_nothing(bv, root):
    call(bv, "compare", cam="scope", name="nope")
    assert not (root / "references-mock").exists()


def test_compare_text_always_carries_the_caveat(changing):
    call(changing, "save_reference", cam="scope", name="r")
    text = text_of(call(changing, "compare", cam="scope", name="r")[1])
    assert "Caveat: absdiff can't tell rework from a moved board" in text


def test_reference_without_checksum_is_rejected(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    js = root / "references-mock" / "scope" / "r.json"
    meta = json.loads(js.read_text())
    del meta["pair_sha256"]
    js.write_text(json.dumps(meta))
    err, content = call(bv, "compare", cam="scope", name="r")
    assert err and "no image checksum" in text_of(content)


def test_busy_wait_is_bounded_per_call(bv, root):
    import fcntl
    import os
    import threading
    import time

    call(bv, "save_reference", cam="scope", name="r")
    fd = os.open(root / "references-mock" / "scope" / ".lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        times, errs = [], []

        def one():
            t = time.monotonic()
            try:
                bv.references.load("scope", "r")
            except Exception as e:  # noqa: BLE001
                errs.append(str(e))
            times.append(time.monotonic() - t)

        threads = [threading.Thread(target=one) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(errs) == 4 and all("busy" in e for e in errs) and max(times) < 7
    finally:
        os.close(fd)


def test_torn_pair_from_interrupted_save_is_detected(bv, root):
    import shutil

    d = root / "references-mock" / "scope"
    call(bv, "save_reference", cam="scope", name="r", rotate=180)
    shutil.copy(d / "r.png", root / "png180")
    call(bv, "save_reference", cam="scope", name="r", rotate=0)
    shutil.copy(root / "png180", d / "r.png")  # new PNG landed, old sidecar kept (same size!)
    err, content = call(bv, "compare", cam="scope", name="r")
    assert err and "don't match" in text_of(content)


def test_tiny_change_is_not_reported_as_zero_percent():
    ref, _ = _board(3840, 2160)
    cur = ref.copy()
    cur[1000:1004, 2000:2004] = 230
    stats = _diff(ref, cur)
    assert stats["region_count"] == 1 and stats["changed_px"] > 0 and stats["changed_pct"] > 0


def test_number_formatting_never_scientific():
    from bench_vision.app import _num

    assert _num(0) == "0" and _num(4.8e-05) == "0.00005" and _num(0.28) == "0.3" and _num(12.345) == "12.3"


def test_tiny_change_text_is_readable(root, clock):
    base = make_image(root / "mock" / "side" / "1.png", 3840, 2160, seed=2)
    changed = base.copy()
    changed[1000:1003, 2000:2003] = 255
    cv2.imwrite(str(root / "mock" / "side" / "2.png"), changed)
    bv = BenchVision(root, mock=True, clock=clock)
    bv.save_reference("side", "r")
    text = bv.compare("side", "r")[2]
    assert "e-0" not in text and "#1 x=" in text and "(0%)" not in text


def test_white_balance_change_is_not_called_darker():
    ref = np.full((540, 960, 3), 100, np.uint8)
    cur = ref.astype(int)
    cur[..., 2] += 20  # R up
    cur[..., 0] -= 20  # B down
    stats = _diff(ref, np.clip(cur, 0, 255).astype(np.uint8))
    assert abs(stats["brightness_shift"]) < 8


def test_num_edges():
    from bench_vision.app import _num

    assert _num(99.96, pct=True) == ">99.9" and _num(99.96) == "100.0" and _num(1.2e-7) == "<0.000001"


def test_legend_counts_boxes(root, clock):
    base = make_image(root / "mock" / "scope" / "1.png", 1920, 1080, seed=1)
    changed = base.copy()
    for i in range(6):
        changed[200:230, 100 + i * 300 : 130 + i * 300] = 255
    cv2.imwrite(str(root / "mock" / "scope" / "2.png"), changed)
    bv = BenchVision(root, mock=True, clock=clock)
    bv.save_reference("scope", "r")
    assert "boxes = the 3 largest of 6 zoom regions, covering 3 of 6 changed areas" in bv.compare("scope", "r")[2]


def test_replacing_unreadable_reference_says_so(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    (root / "references-mock" / "scope" / "r.json").write_text("{bad")
    text = text_of(call(bv, "save_reference", cam="scope", name="r")[1])
    assert "old sidecar was missing or unreadable" in text


def test_identical_frames_legend_says_no_boxes(bv):
    call(bv, "save_reference", cam="scope", name="r")
    text = text_of(call(bv, "compare", cam="scope", name="r")[1])
    assert "no boxes" in text and "0 largest" not in text


def test_large_and_separate_changes_both_get_boxes():
    ref, _ = _board(1920, 1080)
    cur = ref.copy()
    cur[50:900, 50:1400] = 255  # big change
    cur[1000:1004, 1800:1806] = 255  # unrelated small bridge far away
    stats = _diff(ref, cur)
    assert len(stats["regions"]) == 2 and stats["region_count"] == 2


def test_isolated_speck_is_not_drawn_warm():
    from bench_vision import imaging

    ref = np.zeros((540, 960, 3), np.uint8)
    cur = ref.copy()
    cur[270, 480] = 255
    heat, stats = imaging.diff_heatmap(ref, cur, 960)
    px = np.asarray(heat.convert("RGB")).astype(int)[265:276, 475:486]
    assert stats["changed_px"] == 0 and px[..., 0].max() < 100


def test_edited_rotation_in_sidecar_is_detected(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    js = root / "references-mock" / "scope" / "r.json"
    js.write_text(json.dumps({**json.loads(js.read_text()), "rotation": 180}))
    err, content = call(bv, "compare", cam="scope", name="r")
    assert err and "don't match" in text_of(content)


def test_outline_does_not_cover_edge_change():
    from bench_vision import imaging

    ref, _ = _board(1920, 1080)
    cur = ref.copy()
    cur[1070:1080, 1910:1920] = 255  # corner
    heat, stats = imaging.diff_heatmap(ref, cur, 1024)
    px = np.asarray(heat.convert("RGB")).astype(int)
    corner = px[-5:, -5:]
    assert corner[..., 0].mean() > 150  # still hot, not painted over by the box outline


def test_merged_boxes_wording_counts_areas(root, clock):
    base = make_image(root / "mock" / "scope" / "1.png", 1920, 1080, seed=1)
    changed = base.copy()
    for x in range(900, 1100, 20):
        changed[500:502, x : x + 10] = 255
    cv2.imwrite(str(root / "mock" / "scope" / "2.png"), changed)
    bv = BenchVision(root, mock=True, clock=clock)
    bv.save_reference("scope", "r")
    text = bv.compare("scope", "r")[2]
    assert "1 box covering all 10 changed areas" in text and "largest" not in text


def test_single_region_wording(root, clock):
    base = make_image(root / "mock" / "scope" / "1.png", 1920, 1080, seed=1)
    changed = base.copy()
    changed[500:520, 900:920] = 255
    cv2.imwrite(str(root / "mock" / "scope" / "2.png"), changed)
    bv = BenchVision(root, mock=True, clock=clock)
    bv.save_reference("scope", "r")
    text = bv.compare("scope", "r")[2]
    assert "1 box covering the changed area" in text and "Changed region in this" in text


def test_subthreshold_difference_is_not_warm():
    from bench_vision import imaging

    ref, _ = _board(1920, 1080)
    cur = ref.copy()
    cur[400:500, 800:1100] = np.clip(cur[400:500, 800:1100].astype(int) + 25, 0, 255)
    heat, stats = imaging.diff_heatmap(ref, cur, 1024)
    px = np.asarray(heat.convert("RGB")).astype(int)[230:250, 450:560]
    assert stats["changed_px"] == 0
    assert px[..., 0].mean() < 100  # red stays low: blue/cyan, not yellow


def test_heatmap_encode_survives_noise():
    from PIL import Image as PILImage

    from bench_vision import imaging

    rng = np.random.default_rng(0)
    img = PILImage.fromarray(rng.integers(0, 255, (576, 1024, 3), dtype=np.uint8))
    for q in (85, 95, 100):
        assert imaging.encode_jpeg(img, q, full_chroma=True)[:2] == b"\xff\xd8"


def test_label_not_drawn_on_top_left_change():
    from bench_vision import imaging

    ref, _ = _board(1920, 1080)
    cur = ref.copy()
    cur[0:20, 0:20] = 255
    heat, _ = imaging.diff_heatmap(ref, cur, 1024)
    px = np.asarray(heat.convert("RGB")).astype(int)[0:10, 0:10]
    white = (px.min(axis=2) > 200).mean()
    assert white < 0.2  # the change stays visible, not hidden under a white "1"


def test_stale_temp_files_are_swept(bv, root):
    import os
    import time

    call(bv, "save_reference", cam="scope", name="r")
    d = root / "references-mock" / "scope"
    stale = d / ".r.abc.tmp"
    stale.write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    call(bv, "save_reference", cam="scope", name="r")
    assert not stale.exists()


def test_counted_colour_starts_at_yellow():
    import cv2 as _cv2

    lut = _cv2.applyColorMap(np.array([[160]], np.uint8), _cv2.COLORMAP_JET)[0, 0]  # BGR
    assert lut[2] > 200 and lut[1] > 200 and lut[0] < 60  # yellow


def test_full_frame_box_outline_is_on_image():
    from bench_vision import imaging

    rng = np.random.default_rng(3)
    ref = cv2.GaussianBlur(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8), (3, 3), 0)
    heat, stats = imaging.diff_heatmap(ref, np.roll(ref, 3, axis=1), 960)
    assert stats["regions"] == [[0, 0, 960, 540]]
    px = np.asarray(heat.convert("RGB")).astype(int)
    assert (px[:, 0].min(axis=1) > 200).mean() > 0.8  # white outline along the left edge


def test_smallest_counted_change_stays_warm_after_jpeg(root, clock):
    import io

    from bench_vision import imaging

    ref, _ = _board(3840, 2160)
    cur = ref.copy()
    cur[1500:1502, 400:402] = 255  # 4 px group at 4K: well under one pixel at 1024 wide
    heat, stats = imaging.diff_heatmap(ref, cur, 1024)
    assert stats["changed_px"] >= 4
    px = np.asarray(Image.open(io.BytesIO(imaging.encode_jpeg(heat, 75, full_chroma=True))).convert("RGB")).astype(int)
    y, x = 1501 * 576 // 2160, 401 * 1024 // 3840
    patch = px[y - 3 : y + 4, x - 3 : x + 4]
    assert ((patch[..., 0] > 180) & (patch[..., 2] < 120)).sum() >= 4


def test_compare_sidecar_identifies_reference_version(bv, root):
    call(bv, "save_reference", cam="scope", name="r")
    call(bv, "compare", cam="scope", name="r")
    meta = [json.loads(p.read_text()) for p in sorted((root / "captures").rglob("*.json"))][-1]
    assert meta["reference_saved_at"].startswith("2026-10-04T12:00") and len(meta["reference_sha256"]) == 64
