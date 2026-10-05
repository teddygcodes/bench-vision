"""Load and validate config.toml."""

from __future__ import annotations

import difflib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

BY_ID_PREFIX = "/dev/v4l/by-id/"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
CONTROL_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*=\s*(-?\d+)\s*$")

TOP_LEVEL_KEYS = {"cameras", "server"}
SERVER_KEYS = {"jpeg_quality"}
CAMERA_KEYS = {"device", "resolution", "fourcc", "default_rotation", "warmup_frames", "v4l2_controls"}


@dataclass(frozen=True)
class CameraConfig:
    name: str
    device: str
    resolution: tuple[int, int] = (1920, 1080)
    fourcc: str = "MJPG"
    default_rotation: int = 0
    warmup_frames: int = 5
    v4l2_controls: tuple[tuple[str, int], ...] = ()
    source: str = "config.toml"  # file it came from, for error messages


@dataclass(frozen=True)
class Config:
    cameras: dict[str, CameraConfig]
    jpeg_quality: int = 85
    path: Path | None = None


def _did_you_mean(word: str, options: set[str]) -> str:
    match = difflib.get_close_matches(word, sorted(options), n=1)
    return f" (did you mean '{match[0]}'?)" if match else ""


def _check_keys(where: str, table: dict, allowed: set[str]) -> None:
    for key in table:
        if key not in allowed:
            raise ConfigError(
                f"{where}: unknown key '{key}'{_did_you_mean(key, allowed)}. "
                f"Allowed keys: {', '.join(sorted(allowed))}."
            )


def parse_control(text: str, where: str) -> tuple[str, int]:
    m = CONTROL_RE.match(text) if isinstance(text, str) else None
    if not m:
        raise ConfigError(
            f'{where}: v4l2_controls entry {text!r} is not of the form "name=integer", '
            'e.g. "focus_absolute=300".'
        )
    return m.group(1), int(m.group(2))


def _parse_camera(name: str, raw: object, where_file: str) -> CameraConfig:
    where = f"{where_file} [cameras.{name}]"
    if not NAME_RE.match(name):
        raise ConfigError(
            f"{where}: camera name '{name}' must be lowercase letters, digits, '-' or '_' "
            "(it is used in file names)."
        )
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a table, got {type(raw).__name__}.")
    _check_keys(where, raw, CAMERA_KEYS)

    device = raw.get("device")
    if not isinstance(device, str) or not device:
        raise ConfigError(f"{where}: 'device' is required (a path under {BY_ID_PREFIX}).")
    if not device.startswith(BY_ID_PREFIX) or len(device) == len(BY_ID_PREFIX) or ".." in device.split("/"):
        raise ConfigError(
            f"{where}: device '{device}' must be a stable path under {BY_ID_PREFIX} "
            "(bare /dev/videoN numbers reorder between boots). Run `uv run bench-vision setup` "
            "to list them."
        )

    res = raw.get("resolution", [1920, 1080])
    if (
        not isinstance(res, list)
        or len(res) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 16384 for v in res)
    ):
        raise ConfigError(f"{where}: 'resolution' must be [width, height] in pixels, got {res!r}.")

    fourcc = raw.get("fourcc", "MJPG")
    if not isinstance(fourcc, str) or not re.fullmatch(r"[!-~][ -~]{3}", fourcc):
        raise ConfigError(f"{where}: 'fourcc' must be a 4-character ASCII code like \"MJPG\" (case-sensitive), got {fourcc!r}.")

    rot = raw.get("default_rotation", 0)
    if not isinstance(rot, int) or isinstance(rot, bool) or rot not in (0, 90, 180, 270):
        raise ConfigError(f"{where}: 'default_rotation' must be 0, 90, 180 or 270, got {rot!r}.")

    warm = raw.get("warmup_frames", 5)
    if not isinstance(warm, int) or isinstance(warm, bool) or not 1 <= warm <= 60:
        raise ConfigError(f"{where}: 'warmup_frames' must be an integer 1-60, got {warm!r}.")

    ctrls_raw = raw.get("v4l2_controls", [])
    if not isinstance(ctrls_raw, list):
        raise ConfigError(f'{where}: \'v4l2_controls\' must be a list of "name=value" strings.')
    ctrls = tuple(parse_control(c, where) for c in ctrls_raw)

    return CameraConfig(
        name=name,
        device=device,
        resolution=(res[0], res[1]),
        fourcc=fourcc,
        default_rotation=rot,
        warmup_frames=warm,
        v4l2_controls=ctrls,
        source=where_file,
    )


def parse_config(data: dict, where: str = "config.toml", path: Path | None = None) -> Config:
    _check_keys(where, data, TOP_LEVEL_KEYS)

    server = data.get("server", {})
    if not isinstance(server, dict):
        raise ConfigError(f"{where} [server]: expected a table.")
    _check_keys(f"{where} [server]", server, SERVER_KEYS)
    quality = server.get("jpeg_quality", 85)
    if not isinstance(quality, int) or isinstance(quality, bool) or not 30 <= quality <= 100:
        raise ConfigError(f"{where} [server]: 'jpeg_quality' must be an integer 30-100, got {quality!r}.")

    cams_raw = data.get("cameras")
    if not isinstance(cams_raw, dict) or not cams_raw:
        raise ConfigError(
            f"{where}: no cameras defined. Add at least one [cameras.<name>] table, "
            "or run `uv run bench-vision setup` to generate one."
        )
    cameras = {name: _parse_camera(name, raw, where) for name, raw in cams_raw.items()}

    seen: dict[str, str] = {}
    for cam in cameras.values():
        if cam.device in seen:
            raise ConfigError(
                f"{where}: cameras '{seen[cam.device]}' and '{cam.name}' use the same device {cam.device}."
            )
        seen[cam.device] = cam.name

    return Config(cameras=cameras, jpeg_quality=quality, path=path)


def load_config(path: Path, mock: bool = False) -> Config:
    try:
        exists = path.is_file() or path.exists()
    except OSError as e:
        raise ConfigError(f"Could not access {path}: {e.strerror or e}.") from None
    if not exists:
        if mock:
            raise ConfigError(
                f"No config file at {path}. Fix the --config path, or drop --config to let --mock "
                "use the images in ./mock/ directly."
            )
        raise ConfigError(
            f"No config file at {path}. Run `uv run bench-vision setup` with the cameras plugged in "
            "to create one, or start the server with `--mock` to use the images in ./mock/."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Could not read {path}: {e.strerror or e}.") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path} is not valid UTF-8 text.") from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}.") from None
    return parse_config(data, where=path.name, path=path)
