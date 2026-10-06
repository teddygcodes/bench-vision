# bench-vision

An MCP server that gives Claude Code eyes on a soldering bench: a top-down USB microscope
(`scope`) and a low oblique camera (`side`). Claude captures stills, zooms into regions at full
resolution, overlays a labelled grid you can point at ("look at B3"), diffs before/after rework,
and adjusts focus/exposure. Images go to Claude; nothing runs a local vision model.

`CLAUDE.md` tells Claude how to judge hand-soldered joints from these two views.

## Requirements

- `git`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.11 for you)
- Claude Code with its `claude` command on your PATH (check: `claude --version`); needed to register
  the MCP server, not for mock captures or the wall display
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

Trying the display needs only `uv sync` (no cameras, no Claude Code). `bench-vision display`
serves a page at <http://127.0.0.1:8765> for a monitor on the wall: the left two thirds show the
image Claude is talking about (or two side by side) with a caption bar, and otherwise a live view
of the `scope` camera; the right third shows the board map and the current step. Claude drives it
with `show`, `show_compare`, `show_step` and `show_clear`; updates appear within a second, no reload. If the display isn't
running, those tools say so in one line and everything else keeps working.

Try it in mock mode. In one terminal, inside `bench-vision`, start the display server and leave it
running (`--mock` makes its live view play the mock board; on the Linux box leave it out):

```bash
uv run bench-vision display --mock
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

(On the Linux box with real cameras, drop `--mock`.) Keep the display running for the next section.

### Live view, board map and verdicts

When nothing has been pushed with `show` for 20 seconds (`[display] live_idle_seconds`), or after
`show_clear` with `{"area": "image"}`, the image area shows a live view of the `scope` camera
(`[display] live_camera`) at up to 15 fps. It only runs while the page is open, stays on this
machine (nothing from it goes to Claude), and hands the camera to Claude's capture tools whenever
they need it, within half a second. If the camera is missing or unplugged, the image area says so and
the live view tries again after 3 s, 6 s, 12 s and so on, up to every 30 s. A capture of that camera that works
brings it back at once. `set_target` draws a box on it. Run the display and the MCP
server from the same `bench-vision` folder: they share the camera hand-over and the board files.

A board is a reference image plus the joints on it. With the display from above still running
(`--mock`), try it on the mock board. Take a capture and note its `image_path`:

```bash
uv run bench-vision call capture '{"cam": "scope"}' --mock
```

Record the board, pasting that path in place of `PASTE_IMAGE_PATH_HERE` (the joint boxes are the
mock board's J1-J8 in full-resolution pixels):

```bash
uv run bench-vision call board_init '{"name": "mock-board", "image_path": "PASTE_IMAGE_PATH_HERE", "joints": [{"id": "J1", "x": 165, "y": 475, "w": 130, "h": 130}, {"id": "J2", "x": 373, "y": 475, "w": 130, "h": 130}, {"id": "J3", "x": 580, "y": 475, "w": 130, "h": 130}, {"id": "J4", "x": 787, "y": 475, "w": 130, "h": 130}, {"id": "J5", "x": 995, "y": 475, "w": 130, "h": 130}, {"id": "J6", "x": 1202, "y": 475, "w": 130, "h": 130}, {"id": "J7", "x": 1410, "y": 475, "w": 130, "h": 130}, {"id": "J8", "x": 1617, "y": 475, "w": 130, "h": 130}]}' --mock
```

The board-map tile appears at the top of the step panel with every joint grey (todo). Mark two
joints verified, one flagged, and make J5 the active one:

```bash
uv run bench-vision call board_set '{"name": "mock-board", "joint_id": "J1", "state": "verified"}' --mock
uv run bench-vision call board_set '{"name": "mock-board", "joint_id": "J2", "state": "verified"}' --mock
uv run bench-vision call board_set '{"name": "mock-board", "joint_id": "J3", "state": "flagged"}' --mock
uv run bench-vision call board_set '{"name": "mock-board", "joint_id": "J5", "state": "active"}' --mock
```

The map now shows J1 and J2 green, J3 red, and J5 yellow and pulsing. The last command prints the
`set_target` call for J5 (in the form Claude uses); as a command it is:

```bash
uv run bench-vision call set_target '{"cam": "scope", "x": 995, "y": 475, "w": 130, "h": 130, "label": "J5"}' --mock
```

A yellow box labelled J5 appears around J5 on the live view (once the live view is showing again,
i.e. up to 20 s after the last `show`). Verdicts go in the strip along the bottom of the image area
as a thumbnail with the joint id under it (green border = good, red = anything else). Use the same
capture path as for `board_init`:

```bash
uv run bench-vision call record_verdict '{"joint_id": "J3", "verdict": "bridge", "image_path": "PASTE_IMAGE_PATH_HERE", "note": "bridged to J4"}' --mock
```

Boards are saved in `boards/`. When the display restarts it reloads the current board, its last
verdicts, the current step and the targets. Stop the display server with Ctrl-C when you're done.

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
live_camera = "scope"             # camera for the wall's live view (default: scope)
live_idle_seconds = 20            # seconds a pushed image stays before the live view returns

[cameras.side]
device = "/dev/v4l/by-id/usb-Arducam_..._video-index0"   # never bare /dev/videoN
resolution = [3840, 2160]
fourcc = "MJPG"
default_rotation = 0              # clockwise: 0, 90, 180 or 270
warmup_frames = 5                 # frames grabbed at least (the last is kept); with auto exposure a capture
                                  # keeps reading ~1.5-2.5 s until brightness settles (skipped when exposure
                                  # is manual: auto_exposure=1 / exposure_auto=1, in config or set_control)
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
| `set_target(cam, x, y, w, h, label)` / `clear_target(cam)` | Box (or unbox) a spot on the wall's live view, in that camera's full-res pixels |
| `board_init(name, image_path, joints)` | Record a board: reference image + `[{id, x, y, w, h}]` joints; becomes the current board |
| `board_set(name, joint_id, state)` | `todo`, `active`, `verified` or `flagged` (one active joint at a time) |
| `record_verdict(joint_id, verdict, image_path, note="")` | Log a verdict for a joint of the current board; shown on the verdict strip |

Coordinates are full-resolution pixels of the (rotated) frame; every reply states the frame
size and the scale of the returned image, and includes `image_path:` (the saved full-res frame,
which `show` takes directly); `save_reference` and `compare` also give `reference_path:`. Only one camera is ever open at a time, opened
lazily per call; opening times out after 5 s and each frame after 5 s.

## Files it writes

- `captures/YYYY-MM-DD/<cam>_<HHMMSS>.jpg` + `.json`: the full-resolution frame behind every
  tool call, with a sidecar (camera, controls, crop box, sizes). This history survives `/clear`.
- `references/<cam>/<name>.png` + `.json`: saved references (`references-mock/` in mock mode).
- `.bench-vision/`: the last grid per camera, so `capture_cell` survives a server restart.
- `boards/<name>.json`, `boards/<name>.jpg`, `boards/<name>/`: boards, their reference images, verdict
  log and thumbnails; `boards/.current.json` names the board the display shows.
- `<image>.marked.jpg`: the annotated copy `show`/`show_step` draw marks on, next to the capture
  (images from elsewhere, e.g. `mock/`, get theirs in today's `captures/` folder).

All of these are git-ignored.

## Troubleshooting

- **`Corrupt JPEG data: ... extraneous bytes before marker 0xd9` on every capture:** harmless. Some
  cameras (the TOMLOV, which shows up as "RaySmartTech VMS700B") pad each MJPEG frame and libjpeg
  warns about it on stderr; the frames decode fine and nothing reaches the MCP connection.
- **A control in `v4l2_controls` is read-only on the camera** (`setup` lists it with `flags=read-only`,
  e.g. the TOMLOV's focus): it is skipped with a warning on the server's stderr (the `capture` CLI
  shows it; for the MCP server it's in Claude Code's MCP log); remove it from the list.
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
