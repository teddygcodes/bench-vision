# bench-vision

An MCP server that gives Claude Code eyes on a soldering bench: a top-down USB microscope
(`scope`) and a low oblique camera (`side`). Claude captures stills, zooms into regions at full
resolution, overlays a labelled grid you can point at ("look at B3"), diffs before/after rework,
and adjusts focus/exposure. Images go to Claude; nothing runs a local vision model.

`CLAUDE.md` tells Claude how to judge hand-soldered joints from these two views.

## Requirements

- `git`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.11 for you)
- Claude Code with its `claude` command on your PATH (check: `claude --version`)
- For real cameras: Linux with V4L2 and `v4l2-ctl` (package `v4l-utils`). Mock mode works anywhere.

## Quick start in mock mode (no cameras, any OS)

Mock mode serves the synthetic board images in `mock/` (`scope.jpg`, `side.jpg`) instead of
cameras, so you can try everything first.

```bash
git clone https://github.com/teddygcodes/bench-vision.git
cd bench-vision
uv sync
uv run pytest -q
```

Check that a capture works (this runs the same `capture` tool Claude will use, without Claude):

```bash
uv run bench-vision capture scope --mock
```

It prints something like
`scope: full frame 1920x1080 (rotation 0°), returned 1024x576. ... Saved captures/2026-10-05/scope_142233.jpg`
and the full-resolution still is now in `captures/`.

Register the server with Claude Code. Run this **inside the `bench-vision` directory**: the
default scope (`local`) ties the server to this directory, and Claude Code starts it from here.

```bash
claude mcp add bench-vision -- uv run bench-vision serve --mock
claude mcp list
```

`claude mcp list` should show `bench-vision: uv run bench-vision serve --mock - ✔ Connected`.

### First inspection

Start Claude Code in the same directory and ask:

```bash
claude
```

> Use bench-vision: list the cameras, draw a 4x6 grid on scope, then inspect the joints J1-J8
> and give me a per-joint verdict.

Claude will call `list_cameras`, `grid`, then `capture_cell` / `capture_region` on the joints
(Claude Code asks you to approve each bench-vision tool the first time it is used).
The mock board has planted defects: J3-J4 bridged, J5 cold, J6 unsoldered, J7 insufficient,
and a solder ball between J7 and J8.

## First run on the Linux box (real cameras)

On the mini PC, with both cameras plugged in (the TOMLOV in "PC Camera" mode):

```bash
sudo apt install v4l-utils
sudo usermod -aG video "$USER"
```

Log out and back in once so the `video` group applies. Then (if you already cloned the repo for the
mock quick start, just `cd` into it and skip the clone):

```bash
git clone https://github.com/teddygcodes/bench-vision.git
cd bench-vision
uv sync
uv run bench-vision setup
```

`setup` prints a report between `=====` lines: `v4l2-ctl --list-devices`, every
`/dev/v4l/by-id/` path, each camera's formats/resolutions/frame rates and its full control
list. **Paste the whole report to Claude** so it can check the camera names, resolutions and
controls. It also writes a starter `config.toml` (or `config.generated.toml` if a
`config.toml` already exists; `--force` overwrites). Camera names are guessed from the by-id
names (`arducam` -> `side`, `tomlov` -> `scope`, anything else `cam1`, `cam2`, ...); rename
the `[cameras.*]` tables if a guess is wrong.

Check each camera without Claude, then register the live server (inside `bench-vision`). If you
registered the mock server earlier, run `claude mcp remove bench-vision` first.

```bash
uv run bench-vision capture scope
uv run bench-vision capture side
claude mcp add bench-vision -- uv run bench-vision serve
claude mcp list
```

**Control names on older kernels.** Newer kernels call the autofocus switch
`focus_automatic_continuous` and the manual exposure `exposure_time_absolute` (with
`auto_exposure`). Older kernels call them `focus_auto`,
`exposure_absolute` and `exposure_auto`. Use whichever names `setup` lists for your camera,
both in `config.toml` and when asking Claude to `set_control`. For example, to lock the
Arducam's focus on an older kernel:
`v4l2_controls = ["focus_auto=0", "focus_absolute=300"]`. `setup` writes the right names
for the focus lock automatically.

## Configuration (`config.toml`)

```toml
[server]
jpeg_quality = 85                 # JPEGs returned to Claude

[cameras.side]
device = "/dev/v4l/by-id/usb-Arducam_..._video-index0"   # never bare /dev/videoN
resolution = [3840, 2160]
fourcc = "MJPG"
default_rotation = 0              # clockwise: 0, 90, 180 or 270
warmup_frames = 5                 # frames grabbed so exposure settles; the last is kept
v4l2_controls = ["focus_automatic_continuous=0", "focus_absolute=300"]   # applied in order at every open
```

Add more `[cameras.<name>]` tables for more cameras (e.g. an `overhead`). See
`config.example.toml`. Errors in the file (typos, wrong types, a missing device) come back from
every tool as one clear line naming the entry.

## Tools

| Tool | What it does |
|---|---|
| `list_cameras()` | Names, device paths, connected, open/closed, resolution |
| `capture(cam, rotate=0, max_edge=1024)` | One still (long edge <= 1024 px) |
| `capture_region(cam, x, y, w, h, rotate=0, max_edge=768)` | Crop of a fresh full-res frame, scaled to a 768 px long edge |
| `grid(cam, rows, cols, rotate=0, max_edge=1024)` | Full frame with a labelled grid (A1 = top-left); remembered per camera |
| `capture_cell(cam, cell, max_edge=768, margin=0.0)` | `capture_region` for a grid cell such as `"B3"` |
| `save_reference(cam, name)` | Keep the current frame as a named reference |
| `compare(cam, name)` | Reference and now side by side, plus an absdiff heatmap with numbered changed regions |
| `set_control(cam, control, value)` | Set a v4l2 control (focus, exposure, gain); checked against the camera's control list |

Coordinates are full-resolution pixels of the (rotated) frame; every reply states the frame
size and the scale of the returned image. Only one camera is ever open at a time, opened
lazily per call; opening times out after 5 s and each frame after 5 s.

## Files it writes

- `captures/YYYY-MM-DD/<cam>_<HHMMSS>.jpg` + `.json`: the full-resolution frame behind every
  tool call, with a sidecar (camera, controls, crop box, sizes). This history survives `/clear`.
- `references/<cam>/<name>.png` + `.json`: saved references (`references-mock/` in mock mode).
- `.bench-vision/`: the last grid per camera, so `capture_cell` survives a server restart.

All of these are git-ignored.

## Troubleshooting

- **`claude mcp list` shows bench-vision as failed:** the server didn't start at all (config mistakes
  don't cause this; they come back from each tool instead). Make sure you registered it from inside the
  `bench-vision` directory, that `uv` is on the PATH Claude Code uses, and that
  `uv run bench-vision serve --mock` starts there without an error (it then waits silently; stop it
  with Ctrl-C). `uv run bench-vision capture scope --mock` checks the cameras/config side.
- **Using it from another project directory:** register it for your user with absolute paths:
  `claude mcp add -s user bench-vision -- uv run --directory /path/to/bench-vision bench-vision serve --root /path/to/bench-vision`
- **"This needs Linux/v4l2":** `setup`, `set_control` and real cameras need Linux with
  `v4l2-ctl`. On a Mac, use `--mock`.
- **A camera "did not open/deliver a frame within 5 s":** unplug and replug it. Until the hung
  call returns, other captures are refused so two cameras are never open at once.
- **macOS: `ModuleNotFoundError: No module named 'bench_vision'`:** a clone inside an iCloud-synced
  folder (such as Desktop) can get its `.venv` files hidden or evicted. Clone somewhere else, or run
  `rm -rf .venv && uv sync`.
