# imu-video-sync

Command-line tool that time-syncs a telemetry log (CSV) to a camera video (MP4) using IMU cross-correlation. Built-in sources include GoPro MP4 (GPMF) and AiM-style CSV logs. The architecture is modular so additional in-repo sources can be added cleanly. It estimates the time offset by comparing motion patterns and prints a RaceRender-friendly offset instruction.

**How It Works**
1. Extract video IMU (gyro/accel) from MP4 using a source backend (GoPro GPMF via `pygpmf-oz`) and normalize timestamps to start at 0.0 seconds.
2. Parse a log CSV (AiM-style by default), detect delimiter and column names, and convert time to seconds-from-start.
3. Choose a correlation signal (`gyroMag` by default; can compare multiple signals).
4. Resample both signals to a uniform rate (default 50 Hz).
5. Band-pass filter (default 0.2-8 Hz) and z-score normalize.
6. Cross-correlate within the lag range (default +/-600 s) to estimate offset.
7. Compute confidence metrics and print a summary + RaceRender offset.

**Quickstart**
1. Install dependencies: `pip install -r requirements.txt`
2. Install FFmpeg and ensure `ffmpeg` is on your PATH (required by `pygpmf-oz`).
3. Optional for plotting: `pip install matplotlib`
4. Run in a directory with exactly one `.mp4` and one `.csv`: `imu_video_sync` (otherwise pass `--video` and `--log`).

**Usage**
```
imu_video_sync
imu_video_sync --video session.mp4 --log aim.csv
imu_video_sync --signal gyroMag --max-lag 600 --window 360
imu_video_sync --signal auto
imu_video_sync --signals gyroMag,yawRate,latAcc
imu_video_sync --start 60 --window 240
imu_video_sync --no-auto-window
imu_video_sync --window-step 10
imu_video_sync --log-time-col "Time" --log-gyro-cols "GyroX,GyroY,GyroZ"
imu_video_sync --write-video-imu-csv
imu_video_sync --write-shifted-log
imu_video_sync --plot
```

**Outputs (Printed)**
- Selected video signal and inferred sample rate.
- Selected log signal and inferred sample rate.
- **Sync Summary** with:
  - `Lag (seconds)`: estimated offset in seconds.
  - `Correlation peak`: max normalized correlation (higher is better).
  - `Peak-to-sidelobe ratio (PSR)`: peak vs. second-strongest local peak (higher is better).
  - `Stability (stddev s)`: lag variability across subwindows (lower is better).
  - `Drift (s/s)` when measurable: trend of lag over time (near zero is better).
- **RaceRender Offset**:
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

**Signals**

Supported signals:
- `gyroMag` (magnitude of available gyro axes)
- `accMag` (magnitude of available accel axes)
- `gyroZ` (Z gyro axis if available)
- `yawRate` (yaw rate channel if detected)
- `latAcc` (lateral acceleration if detected)

Use `--signal auto` or `--signals` to compare multiple signals and pick the best-scoring lag.

**Auto Window Selection**

By default the tool scans many windows across the session and uses a robust consensus of lags (weighted median). This is more reliable than a single window when there is idling, waiting in grid, or long dead time.

Use `--no-auto-window` to use a single window, or tune the scan granularity with `--window-step`.

**Interpreting Results**
Good sync values usually look like this:
- `Correlation peak` >= 0.4
- `PSR` >= 3.0
- `Stability (stddev s)` <= 0.2
- `Drift (s/s)` close to 0

Bad sync values usually look like this:
- `Correlation peak` < 0.2
- `PSR` < 1.5
- `Stability (stddev s)` > 0.5
- `Drift (s/s)` noticeably non-zero

Examples:
- Good: `Lag +376.280 s`, `peak 0.56`, `PSR 5.66`, `stability 0.03`
- Bad: `Lag +12.000 s`, `peak 0.09`, `PSR 1.1`, `stability 0.80`

If values are "bad," try:
- Using `--signals` to compare multiple signals.
- Increasing `--window` to include more driving.
- Reducing `--max-lag` if you know the offset range (the expected time difference between when the data log starts and when the video starts).

**Design Principles**
- One reliable path: uses `pygpmf-oz` for GoPro GPMF telemetry (current video backend).
- Robust to axis mismatch: defaults to `gyroMag` (vector magnitude) instead of raw axis alignment.
- High signal-to-noise: band-pass filtering and z-score normalization before correlation.
- Stable results: multi-window consensus across the session for better confidence.
- Non-interactive: auto-detect inputs with explicit flags for overrides.

**Notes**
- If the requested signal is missing in either file, the tool auto-falls back to the best available option and prints a warning.
- The shifted log CSV overwrites the time column with seconds-from-start values plus the computed lag.
- Use `--video-source` or `--log-source` to force a specific backend by name.
- RaceRender can only apply positive offsets. The tool prints which input to offset accordingly.
