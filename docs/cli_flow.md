# CLI Flow

The CLI is in `src/imu_video_sync/cli.py`.

## Auto-detecting files
- If you do not pass `--video` and `--log`, the tool looks for exactly one `.mp4` and one `.csv` in the current directory.
- If there are multiple files, you must pass explicit paths.

## Signal selection
- `--signal gyroMag` uses a single signal.
- `--signal auto` tries multiple signals and picks the best.
- `--signals` lets you set the exact list to compare.

## Windowing behavior
- By default, the tool scans multiple windows and builds a consensus lag.
- Use `--no-auto-window` to analyze a single window.
- `--window` sets the window length (default 360 seconds).
- `--window-step` controls how far each window shifts (default 20 seconds).
- If the requested window is longer than the available data, it is automatically shrunk.
- If the window is effectively the full video length, auto-windowing is disabled and the tool falls back to single-window mode.

## Window count estimate
For a session of length `T`, window length `W`, and step `S`, window count is roughly:
`floor((T - W) / S) + 1`

## Outputs
The CLI prints a summary and a RaceRender-friendly offset line. Optional files are created only if you pass their flags: `--write-video-imu-csv`, `--write-shifted-log`, or `--plot`.
