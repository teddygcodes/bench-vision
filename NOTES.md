# Review notes

Cosmetic findings from the per-deliverable reviews. Recorded here instead of
triggering another review round; none affects results, data, or error handling.

## Deliverable 1
- `cameras = [1, 2]` in config.toml says "no cameras defined" rather than "must be a table".
- A by-id device path with a trailing slash (`/dev/v4l/by-id/x/`) is accepted.
- Argument-validation errors lack the SDK's "Error executing tool" prefix that other errors have.

## Deliverable 2
- In the 26x50 grid only the top row and left column are labelled, at ~9 px; readable but you have to count.
- `rows=True` / `margin=True` are coerced to 1 by the MCP layer's lax validation before the tools' own checks run.
- A process killed mid-write can leave `.bench-vision/.grids*.tmp` files behind.

## Deliverable 3
- A hand-edited reference sidecar with `"saved_at": null` makes replies read "saved None".
- A non-string `device` in a hand-edited sidecar is quoted twice in the device-mismatch warning.
- Counted changes are drawn at least 3x3 px on the heatmap, so very noisy frames look warmer than the changed-% suggests.
- A box spanning the full frame height puts its number label inside the box.
- If the second file swap of a reference save fails, the old reference is lost (the torn pair is detected on load).

## Deliverable 4
- `setup` writes by-id names into TOML unescaped; a name containing `"` or `\` would break the generated file (udev sanitises by-id names, so practically unreachable).
- Menu controls are range-checked but not checked against their actual menu entries (e.g. `auto_exposure=2`); real v4l2-ctl rejects such a value with a clear error, the mock accepts it.
- For Stepwise/Continuous frame sizes the report keeps only the maximum size, not the minimum, step or frame rates.
- If `--list-ctrls-menus` fails on a device whose formats listed fine, that device is left out of the generated config.
- "No capture devices found ... plug the cameras in" is printed even when devices exist but expose only metadata nodes.
- While a capture is hung, `list_cameras` shows that camera as "closed".
- Before a camera's first open, set_control can call `focus_absolute` inactive even though config.toml will turn autofocus off at open time.
- In mock mode the capture sidecar lists inactive config controls (e.g. `focus_absolute` with autofocus on) as applied.
- A set_control racing the controls phase of a capture of the same camera can leave the old config value for that one capture.
- If setup fails mid-run, the report lacks its "===== end of report =====" line.
- If a device's `release()` overruns its 5 s wait without any timeout being reported, the next capture calls it "stuck in an earlier capture that timed out".
- If `--get-ctrl` fails after a successful `--set-ctrl` (e.g. a write-only control), set_control reports a failure and doesn't record the override.
- An implausible probed size (over 16384 px) makes setup exit with a config error after a partial report, writing nothing.
- The report's controls section strips indentation, so menu entries ("1: Manual Mode") line up with control lines.
- An override that fails validation at open (e.g. the device changed) is reported as a "config.toml [cameras.x] v4l2_controls" error.
- `value=True` / `value="5"` are coerced to integers by the MCP layer (same class as the deliverable 2 note).
- The set_control docstring says an unknown name lists available controls "with their ranges"; the list has names only (ranges appear in the out-of-range error).
- Older kernels name controls `focus_auto` / `exposure_auto` / `exposure_absolute`. Handled since deliverable 5 (setup's focus lock, set_control hints and override dropping, apply order); see the README's Linux section.
- `subprocess.run` waits for a killed v4l2-ctl after its timeout; one stuck in an uninterruptible ioctl on a dead USB device could block set_control or setup (captures are protected by their worker timeout).
- If a device reports no `focus_absolute` value, the starter config writes `focus_absolute=0`, which a camera with min > 0 would reject at open.
- Opening a real non-by-id device on a Mac reports "could not be opened ... video group" rather than "needs Linux" (by-id configs can't reach this).
- The 90 s capture lock wait is shorter than the worst-case capture with 6+ config controls (each v4l2-ctl call may take up to 5 s), so a queued capture could give up early.

## Deliverable 5
- The README tool table omits `rotate`/`max_edge` on `save_reference` and `max_edge` on `compare` (the tools' own descriptions list them).
- The README's "use it from another project directory" tip doesn't mention that this repo's `CLAUDE.md` inspection guidance won't load in that other project.
- README "First inspection": "Start Claude Code in the same directory and ask:" sits before the `claude` command rather than the prompt; reads slightly out of order.
- The `capture` CLI takes no cross-process camera lock, so running it while a live server is mid-capture could open two cameras at once (two Claude sessions on one repo have the same gap).
- `--out ""` is silently ignored.
- With stdout piped, the `--out` error on stderr can appear before the capture text it says is "above".
- No unit test covers set_control's old-kernel hints and override dropping (verified by hand in review).
- A Ctrl-C in the first few milliseconds, before `bench-vision` has even imported its CLI module, still gets Python's default traceback.
- README: `uv run pytest -q` doesn't say what success looks like ("N passed", about a minute).
- README: `claude mcp list` also prints a health-check line and every other configured server; only the bench-vision line matters.
- README "First inspection" doesn't mention signing in to Claude Code or its first-run folder-trust prompt.
- A Ctrl-C that lands during interpreter shutdown after a successful capture exits with status -2 and no message (output and saved file are complete).
- A second Ctrl-C in the microseconds before the handler ignores SIGINT could still escape with a traceback.
- The `capture` CLI's missing-config message suggests "start the server with `--mock`"; for that subcommand it should say "add `--mock`".
- `bench-vision capture` silences the server's own log, so an internal error's "details are in the server's stderr log" shows nothing there, and live-camera warnings (resolution mismatch, skipped inactive controls) are hidden.
- `test_old_kernel_auto_switch_ordering_and_dropping` only checks ordering, despite its name.
- `bench-vision capture` exits 0 when the capture couldn't be saved to captures/ (the WARNING text says so).

## Deliverable 6
- NaN/infinite mark values arrive as null from the MCP layer, so the error says "'x' is required".
- The step body limit counts lines, not rendered lines: five long lines can wrap to ~15 and, at 720p, clip the body and hide the step image while the tool still reports "Step shown".
- Mark text that doesn't fit beside its mark goes to the legend below the image, where an entry wider than the image is shortened with "…".
- For image-sized pushes, an HTTP error from the display server is reported as "isn't running" (the server answers while the client is still sending the body).
- `bench-vision call` silently ignores unknown argument keys (same class as the MCP-layer coercion notes).
- The page skips re-rendering when the state JSON is unchanged; after a display-server restart, a push landing before the page reconnects with an identical id/label/caption could leave the old image showing.
- The display server keeps its state only in memory, so a restart (e.g. by systemd) blanks the step panel.
- The push endpoint accepts any data starting with FF D8 as a JPEG (only the bench-vision tools push).
- Chromium could also get `--no-default-browser-check` and something for the "restore pages" bubble after a power cut (`--disable-session-crashed-bubble` or editing exit_type in its Preferences).
- `Restart=on-failure` with `RestartSec=2` hits systemd's default start limit (5 in 10 s) if the port stays busy, leaving the unit failed (`systemctl --user reset-failed` clears it).
- In mock mode, a config.toml containing only `[display]` fails with "no cameras defined".
- The display server doesn't check the Host header (DNS rebinding); low risk since it binds 127.0.0.1 and requires a JSON Content-Type.
- The red "reconnecting" bar covers the caption bar while the display server is down.
- A chunked POST to the display gets the 413 "body must be 1..N bytes" message.
- `show` replies show an absolute path when the project root is reached through a symlink (only when driving the code directly; the CLI resolves the root).
- A mark label can sit over the board's own silkscreen text next to the mark (it avoids marks, not board features).
- README: the "Wall display" section says it needs only `uv sync`, but the quick start above it also runs the (optional) test suite.
- In a very dense cluster, a numbered badge can end up a little away from its mark (it never covers a mark; the legend below the image lists the text).
- With ~20 marks in one tight cluster, a fallback badge can sit on another badge or label (the least-covering fallback); the legend still lists every text.
- Legend rows are aligned by the top of each row's ink, so baselines shift slightly between rows (e.g. a row with accented capitals sits lower).
- The legend font follows the image's long edge, so on a very narrow tall image (e.g. 200x1920) every entry is shortened to "N lo…".
- Whitespace-only mark text passes validation and draws an empty label box.
- Tabs, other control characters and glyphs the font lacks (emoji, CJK) draw as missing-glyph boxes.

## Deliverable 7
- When a second display on the same root holds the camera lock, the live-view status says "paused for a capture".
- The LIVE badge (top-left) can cover a reticle placed in the top-left corner.
- A missing verdict thumbnail shows as a broken image with alt text.
- set_target uses the configured resolution if this server hasn't captured from that camera yet; a camera delivering a different aspect ratio would misplace the reticle.
- A capture that waits more than 1.5 s for the camera lock fails rather than queueing, even when the holder is another capture (two servers, or the `capture` CLI during a server capture).
- The live reader opens the camera at its full configured resolution and decodes every frame before scaling to 960 px; a lower stream resolution would save CPU and USB bandwidth (needs per-camera mode knowledge to keep the same field of view).
- Mock-mode boards and display state share the live project's `boards/` and `.bench-vision/display.json`, so the README's mock walkthrough leaves "mock-board" as the current board.
- record_verdict crops the joint whenever the image has the board reference's size, even if it came from another camera.
- Joint ids up to 24 characters wrap to several lines in the verdict strip and shrink the live view at 1280x720.
- While the display server is down, the page keeps the LIVE badge on the frozen frame (the offline banner explains it).
- A reader that prints garbage instead of frames leaves the status at "starting".
- A reader stuck in the kernel past the 0.9 s kill wait has the lock released anyway; the capture may then fail with "device busy" (never blocked), and only the display's log says why.
- A hand-edited board whose joint lacks x/y/w/h makes board_set(active) report "internal error (KeyError)" rather than "corrupt board".
- Targets are not tied to a board: a reticle stays up after board_init switches to another board.
- Raw OpenCV error text can reach the wall's live-view status when a read fails with cv2.error; an open failure that isn't a bench-vision error shows only the exception type.
- Two MCP servers on one root overwrite each other's set_control values in controls.json; the live reader uses the last writer's.
- After a reader dies from a signal, the status may show its last stderr line (a warning) rather than "camera reader exited (-9)".
- The live-view open error lacks the capture error's "another program / video group?" hint.
- Run by hand with stdin closed, the reader exits 0 silently; with stdin = /dev/zero its watcher spins. Under the display stdin is always a pipe.
- Each capture during a live-view back-off both retries the camera at once and moves the back-off one step on, so a few captures take the wait to 30 s (a replug is still noticed within ~1 s).
