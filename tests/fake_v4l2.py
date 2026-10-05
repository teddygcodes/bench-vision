"""A fake `v4l2-ctl` that replays realistic output for the TOMLOV microscope and the Arducam."""

from __future__ import annotations

import re
from pathlib import Path

SCOPE_ID = "usb-TOMLOV_PC_Camera_TM3K-AF_Max-video-index0"
SCOPE_META = "usb-TOMLOV_PC_Camera_TM3K-AF_Max-video-index1"
SIDE_ID = "usb-Arducam_Technology_Co.__Ltd._Arducam_8MP_USB_Camera-video-index0"

LIST_DEVICES = """TOMLOV PC Camera: TOMLOV PC Camer (usb-0000:04:00.3-1):
\t/dev/video0
\t/dev/video1
\t/dev/media0

Arducam 8MP USB Camera: Arducam (usb-0000:04:00.3-2):
\t/dev/video2
\t/dev/video3
\t/dev/media1
"""

SCOPE_FORMATS = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
"""

SIDE_FORMATS = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 3840x2160
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 3840x2160
\t\t\tInterval: Discrete 1.000s (1.000 fps)
"""

META_FORMATS = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture
"""

SCOPE_CTRLS = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                       contrast 0x00980901 (int)    : min=0 max=64 step=1 default=32 value=32
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
\t\t\t\t1: Manual Mode
\t\t\t\t3: Aperture Priority Mode
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
"""

SIDE_CTRLS = SCOPE_CTRLS + """     focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=0 value=312 flags=inactive
"""


class FakeV4L2Ctl:
    """Callable runner: (argv) -> (code, stdout, stderr). Records calls; keeps control values."""

    def __init__(self, by_id_dir: Path | None = None):
        self.calls: list[list[str]] = []
        self.by_id_dir = by_id_dir
        self.values: dict[str, dict[str, int]] = {}
        self.fail_on: str | None = None

    def _dev(self, argv: list[str]) -> str:
        return Path(argv[argv.index("-d") + 1]).name if "-d" in argv else ""

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        self.calls.append(argv)
        dev = self._dev(argv)
        if self.fail_on and self.fail_on in dev:
            return 1, "", f"Failed to open {dev}: No such device"
        if argv[1:] == ["--version"]:
            return 0, "v4l2-ctl 1.26.1\n", ""
        if argv[1:] == ["--list-devices"]:
            return 0, LIST_DEVICES, ""
        if "--list-formats-ext" in argv:
            return 0, {SCOPE_ID: SCOPE_FORMATS, SIDE_ID: SIDE_FORMATS}.get(dev, META_FORMATS), ""
        if "--list-ctrls" in argv or "--list-ctrls-menus" in argv:
            vals = self.values.get(dev, {})
            lines = []
            for line in {SCOPE_ID: SCOPE_CTRLS, SIDE_ID: SIDE_CTRLS}.get(dev, "").splitlines():
                name = line.split()[0] if line.strip() else ""
                if name in vals:
                    line = re.sub(r"value=-?\d+", f"value={vals[name]}", line)
                if name == "focus_absolute" and vals.get("focus_automatic_continuous") == 0:
                    line = line.replace(" flags=inactive", "")
                if name == "exposure_time_absolute" and vals.get("auto_exposure") == 1:
                    line = line.replace(" flags=inactive", "")
                if "--list-ctrls" in argv and line.startswith("\t\t"):
                    continue  # menu entries only appear with --list-ctrls-menus
                lines.append(line)
            return 0, "\n".join(lines) + "\n", ""
        for a in argv:
            if a.startswith("--set-ctrl="):
                name, value = a.split("=", 1)[1].split("=")
                self.values.setdefault(dev, {})[name] = int(value)
                return 0, "", ""
            if a.startswith("--get-ctrl="):
                name = a.split("=", 1)[1]
                return 0, f"{name}: {self.values.get(dev, {}).get(name, 0)}\n", ""
        return 1, "", f"unexpected argv {argv}"


def make_by_id(tmp: Path, names=(SCOPE_ID, SCOPE_META, SIDE_ID)) -> Path:
    """A fake /dev/v4l/by-id directory of symlinks to fake /dev/videoN files."""
    d = tmp / "by-id"
    d.mkdir()
    for i, n in enumerate(names):
        target = tmp / f"video{i}"
        target.write_text("")
        (d / n).symlink_to(target)
    return d
