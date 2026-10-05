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
- Older kernels name controls `focus_auto` / `exposure_auto` / `exposure_absolute`; with those, setup doesn't generate the autofocus lock and the inactive hints / override dropping don't apply (opening still works: inactive controls are skipped). Check `setup` output on the mini PC.
- `subprocess.run` waits for a killed v4l2-ctl after its timeout; one stuck in an uninterruptible ioctl on a dead USB device could block set_control or setup (captures are protected by their worker timeout).
- If a device reports no `focus_absolute` value, the starter config writes `focus_absolute=0`, which a camera with min > 0 would reject at open.
- Opening a real non-by-id device on a Mac reports "could not be opened ... video group" rather than "needs Linux" (by-id configs can't reach this).
- The 90 s capture lock wait is shorter than the worst-case capture with 6+ config controls (each v4l2-ctl call may take up to 5 s), so a queued capture could give up early.
