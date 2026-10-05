from pathlib import Path

import pytest

from bench_vision.config import load_config
from bench_vision.errors import ConfigError

GOOD = """
[server]
jpeg_quality = 80

[cameras.scope]
device = "/dev/v4l/by-id/usb-TOMLOV_PC_Camera-video-index0"
resolution = [1920, 1080]
default_rotation = 180

[cameras.side]
device = "/dev/v4l/by-id/usb-Arducam_8MP-video-index0"
resolution = [3840, 2160]
v4l2_controls = ["focus_automatic_continuous=0", "focus_absolute=300"]
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text)
    return p


def test_good_config(tmp_path):
    cfg = load_config(write(tmp_path, GOOD))
    assert list(cfg.cameras) == ["scope", "side"]
    assert cfg.jpeg_quality == 80
    assert cfg.cameras["scope"].default_rotation == 180
    assert cfg.cameras["side"].resolution == (3840, 2160)
    assert cfg.cameras["side"].v4l2_controls == (("focus_automatic_continuous", 0), ("focus_absolute", 300))


def test_missing_file_mentions_setup_and_mock(tmp_path):
    with pytest.raises(ConfigError, match="bench-vision setup.*--mock"):
        load_config(tmp_path / "config.toml")


def test_typo_in_camera_key_suggests_fix(tmp_path):
    with pytest.raises(ConfigError, match=r"\[cameras.side\]: unknown key 'resoluton' \(did you mean 'resolution'\?\)"):
        load_config(write(tmp_path, GOOD.replace("resolution = [3840", "resoluton = [3840")))


def test_typo_in_top_level_table(tmp_path):
    with pytest.raises(ConfigError, match="unknown key 'camera'.*did you mean 'cameras'"):
        load_config(write(tmp_path, GOOD.replace("[cameras.scope]", "[camera.scope]")))


def test_bare_dev_video_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"\[cameras.scope\].*/dev/v4l/by-id/"):
        load_config(write(tmp_path, GOOD.replace("/dev/v4l/by-id/usb-TOMLOV_PC_Camera-video-index0", "/dev/video0")))


def test_invalid_toml_is_clear(tmp_path):
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write(tmp_path, GOOD + "\n[cameras.side\n"))


@pytest.mark.parametrize(
    "old,new,msg",
    [
        ('"focus_absolute=300"', '"focus_absolute 300"', "name=integer"),
        ("default_rotation = 180", "default_rotation = 45", "default_rotation"),
        ("resolution = [1920, 1080]", "resolution = [1920]", "resolution"),
        ("jpeg_quality = 80", "jpeg_quality = 5", "jpeg_quality"),
        ("[cameras.scope]", "[cameras.Scope]", "lowercase"),
        ('resolution = [3840, 2160]', 'resolution = [3840, 2160]\nfourcc = "MJ\u20acG"', "fourcc"),
    ],
)
def test_bad_values(tmp_path, old, new, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(write(tmp_path, GOOD.replace(old, new)))


def test_duplicate_device(tmp_path):
    text = GOOD.replace("usb-Arducam_8MP-video-index0", "usb-TOMLOV_PC_Camera-video-index0")
    with pytest.raises(ConfigError, match="same device"):
        load_config(write(tmp_path, text))


def test_no_cameras(tmp_path):
    with pytest.raises(ConfigError, match="no cameras"):
        load_config(write(tmp_path, "[server]\njpeg_quality = 80\n"))
