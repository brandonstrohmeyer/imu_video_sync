# imu-video-sync

Command-line tool that time-syncs a telemetry log (CSV) to a camera video (MP4) using IMU cross-correlation. It accurately estimates the time offset by comparing motion patterns and prints an offset instruction that can be used by your video editor of choice including RaceRender.


## Supported Devices

**Cameras**
- GoPro HERO5+
- DJI Osmo Action 4+

**Loggers**
- AiM CSV (e.g., XLog, Solo 2)
- RaceChrono CSV (phone IMU)

## Installation
1. Go to the GitHub [Releases](https://github.com/brandonstrohmeyer/imu_video_sync/releases) page.
2. Download the binary for your OS:
   - Windows: `IMUVideoSync-windows-x64.exe`
   - macOS: `IMUVideoSync-macos-x64`
   - Linux: `IMUVideoSync-linux-x64`
3. Run the binary from a terminal.

## Usage

Basic example:
```
IMUVideoSync --video session.mp4 --log aim.csv

...

Sync Summary
─────────────────────────────
Correlation peak            0.631
Peak-to-sidelobe ratio      3.190
Stability (stddev s)        0.000
Confidence                  High (88/100)

Offset Summary
─────────────────────────────
Lag (seconds)                   +24.540
Lag (frames)                    +1471
Timecode offset                 +00:00:24;32
Video offset within project     00:00:24.540
```

## Outputs
- **Signal Candidates** table (includes the selected signal).
- **Sync Summary** with:
  - `Correlation peak`: max normalized correlation (higher is better).
  - `Peak-to-sidelobe ratio (PSR)`: peak vs. second-strongest local peak (higher is better).
  - `Stability (stddev s)`: lag variability across subwindows (lower is better).
  - `Confidence`: High/Medium/Low with a 0-100 score.
  - `Drift (s/s)` when measurable: trend of lag over time (near zero is better).
- **Offset Summary** with:
  - `Lag (seconds)`: estimated offset in seconds.
  - `Lag (frames)`: offset in frames (when FPS is known).
  - `Timecode offset`: SMPTE timecode offset (when FPS is known).
  - `Video offset within project: HH:MM:SS.mmm` when video starts later than data.
  - `Data offset within project: HH:MM:SS.mmm` when data starts later than video.

**Outputs (Files)**
- `video_imu.csv` if `--write-video-imu-csv`
  - Columns: `time_s, gx, gy, gz, ax, ay, az` (whichever are available).
  - Useful for debugging or external analysis.
- `log_shifted.csv` if `--write-shifted-log`
  - Log CSV with timestamps shifted by the computed lag.
- `sync_plot.png` if `--plot`
  - Visual diagnostics: signals, correlation curve, and difference signal.

## How It Works
1. Extract video IMU (gyro/accel) from MP4 using `telemetry-parser` and normalize timestamps to start at 0.0 seconds.
2. Parse a log CSV (AiM-style or RaceChrono), detect delimiter and column names, and convert time to seconds-from-start.
3. Choose a correlation signal (`gyroMag` by default; can compare multiple signals).
4. Resample both signals to a uniform rate (default 50 Hz).
5. Band-pass filter (default 0.2-8 Hz) and z-score normalize.
6. Cross-correlate within the lag range (default +/-600 s) to estimate offset.
7. Compute confidence metrics + rating and print a summary + RaceRender offset.

**Signals**

Supported signals:
- `gyroMag` (magnitude of available gyro axes)
- `accMag` (magnitude of available accel axes)
- `gyroZ` (Z gyro axis if available)
- `yawRate` (yaw rate channel if detected)
- `latAcc` (lateral acceleration if detected)

Use `--signal auto` or `--signals` to compare multiple signals and pick the best-scoring lag.

**Auto Window Selection**

By default the tool scans many windows across the session and uses a robust consensus of lags (weighted median). This is more reliable than a single window when there is idling, waiting in grid, or long dead time. If the requested window is longer than the available data, it is automatically shrunk. If the window is effectively the full video length, auto-windowing is disabled and the tool falls back to a single-window estimate.

Use `--no-auto-window` to use a single window, or tune the scan granularity with `--window-step`.

**Interpreting Results**
Good sync values usually look like this:
- `Correlation peak` >= 0.4
- `PSR` >= 3.0
- `Stability (stddev s)` <= 0.2
- `Drift (s/s)` close to 0
- `Confidence` >= 60/100 (rough heuristic)

Bad sync values usually look like this:
- `Correlation peak` < 0.2
- `PSR` < 1.5
- `Stability (stddev s)` > 0.5
- `Drift (s/s)` noticeably non-zero

Examples:
- Good: `Lag +376.280 s`, `peak 0.56`, `PSR 5.66`, `stability 0.03`, `confidence 88/100`
- Bad: `Lag +12.000 s`, `peak 0.09`, `PSR 1.1`, `stability 0.80`, `confidence 22/100`

If values are "bad," try:
- Using `--signals` to compare multiple signals.
- Increasing `--window` to include more driving.
- Reducing `--max-lag` if you know the offset range (the expected time difference between when the data log starts and when the video starts).
- If `stability` prints `nan`, reduce the window length so the stability check has room to run.

**Design Principles**
- Source modularity: GoPro GPMF, DJI Osmo Action metadata, AiM CSV, and RaceChrono CSV are built-in.
- Robust to axis mismatch: defaults to `gyroMag` (vector magnitude) instead of raw axis alignment.
- High signal-to-noise: band-pass filtering and z-score normalization before correlation.
- Stable results: multi-window consensus across the session for better confidence.
- Non-interactive: auto-detect inputs with explicit flags for overrides.

## Notes
- If the requested signal is missing in either file, the tool auto-falls back to the best available option and prints a warning.
- The shifted log CSV overwrites the time column with seconds-from-start values plus the computed lag.
- Use `--video-source` or `--log-source` to force a specific backend by name.
- RaceRender can only apply positive offsets. The tool prints which input to offset accordingly.
- Camera metadata extraction uses `telemetry-parser`.
