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
`scope: full frame 1920x1080 (rotation 0°), returned 1024x576. ... image_path: captures/2026-10-05/scope_142233.jpg (full-resolution frame)`
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

## Wall display

Trying the display needs only `uv sync` (no cameras, no Claude Code). `bench-vision display` serves a page at <http://127.0.0.1:8765> for a monitor on the wall: the
left two thirds show the image Claude is talking about (or two side by side) with a caption bar,
the right third shows the current step. Claude drives it with `show`, `show_compare`,
`show_step` and `show_clear`; updates appear within a second, no reload. If the display isn't
running, those tools say so in one line and everything else keeps working.

Try it in mock mode. In one terminal, inside `bench-vision`, start the display server and leave it
running:

```bash
uv run bench-vision display
```

Open <http://127.0.0.1:8765> in a browser. In a second terminal, inside `bench-vision`, push a
step (`bench-vision call` runs any tool once without Claude):

```bash
uv run bench-vision call show_step '{"title": "Hello from bench-vision", "body": "If you can read this,\nthe wall display works.", "progress": "step 1 of 1"}' --mock
```

The step panel on the right shows it immediately. To push an image with a circled spot, take a
capture, then paste the path printed after `image_path:` (just the path, e.g.
`captures/2026-10-05/scope_142233.jpg`; paths relative to `bench-vision` are fine) in place of
`PASTE_IMAGE_PATH_HERE`:

```bash
uv run bench-vision call capture '{"cam": "scope"}' --mock
uv run bench-vision call show '{"image_path": "PASTE_IMAGE_PATH_HERE", "caption": "J5 cold joint: reflow with fresh flux", "marks": [{"kind": "circle", "x": 1000, "y": 480, "w": 120, "h": 120, "text": "J5"}]}' --mock
```

(On the Linux box with real cameras, drop `--mock`.) Stop the display server with Ctrl-C.

### Start on login (Ubuntu)

On the mini PC, with the repo at `~/bench-vision` and `uv` at `~/.local/bin/uv` (check with
`command -v uv`; if either differs, edit the two paths in `deploy/bench-vision-display.service`):

```bash
sudo snap install chromium
mkdir -p ~/.config/systemd/user ~/.config/autostart
cp deploy/bench-vision-display.service ~/.config/systemd/user/
cp deploy/bench-vision-kiosk.desktop ~/.config/autostart/
systemctl --user daemon-reload
systemctl --user enable --now bench-vision-display.service
```

The service starts the display server when you log in; the autostart entry waits until the server
answers, then opens Chromium fullscreen (`--kiosk`) on the page. Log out and back in to check.
Leave kiosk mode with Alt+F4. (The autostart entry uses `wget`, which stock Ubuntu desktop
includes; if not, `sudo apt install wget`.) To keep the screen from blanking, locking or
suspending:

```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
```

If the wall monitor is a second screen, add `--ozone-platform=x11 --window-position=X,0` to the
`chromium` command in `~/.config/autostart/bench-vision-kiosk.desktop`, where X is the width of the
first screen in pixels (window placement is ignored under Wayland without the first flag).
Check the server with `systemctl --user status bench-vision-display`. To use another port, change
it in three places: add `--port PORT` to `ExecStart` in
`~/.config/systemd/user/bench-vision-display.service` (then `systemctl --user daemon-reload` and
`systemctl --user restart bench-vision-display`), replace both `8765`s in
`~/.config/autostart/bench-vision-kiosk.desktop`, and set
`[display] url = "http://127.0.0.1:PORT"` in `config.toml`.

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

[display]
url = "http://127.0.0.1:8765"     # where the show tools push to; match `bench-vision display --port` (default)

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
| `show(image_path, caption, marks=[])` | Put an image on the wall display, with circles/boxes/arrows/labels drawn at full-res coordinates |
| `show_compare(left_path, right_path, caption, left_label, right_label)` | Two images side by side on the wall display |
| `show_step(title, body, image_path=None, marks=[], progress=None)` | Fill the display's step panel (stays until replaced) |
| `show_clear(area="all")` | Clear `all`, `image` or `step` |

Coordinates are full-resolution pixels of the (rotated) frame; every reply states the frame
size and the scale of the returned image, and ends with `image_path:` (the saved full-res frame,
which `show` takes directly); `save_reference` and `compare` also give `reference_path:`. Only one camera is ever open at a time, opened
lazily per call; opening times out after 5 s and each frame after 5 s.

## Files it writes

- `captures/YYYY-MM-DD/<cam>_<HHMMSS>.jpg` + `.json`: the full-resolution frame behind every
  tool call, with a sidecar (camera, controls, crop box, sizes). This history survives `/clear`.
- `references/<cam>/<name>.png` + `.json`: saved references (`references-mock/` in mock mode).
- `.bench-vision/`: the last grid per camera, so `capture_cell` survives a server restart.
- `<image>.marked.jpg`: the annotated copy `show`/`show_step` draw marks on, next to the capture
  (images from elsewhere, e.g. `mock/`, get theirs in today's `captures/` folder).

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
