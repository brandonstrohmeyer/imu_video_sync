# Data Sources

This tool uses two data sources.

## GoPro MP4 Telemetry
- Parsed via `pygpmf-oz`.
- The MP4 contains GPMF (GoPro Metadata Format) packets with sensor data.
- The code extracts gyro and accelerometer samples and their timestamps.
- Timestamps are normalized so the first sample is at `time_s = 0.0`.

## What we keep
- `gyro`: 3-axis rotation (gx, gy, gz).
- `accel`: 3-axis acceleration (ax, ay, az).
- `time_s`: seconds from clip start.

## Missing data
Some GoPro clips may lack gyro or accel. The tool falls back to whatever is available.

## AiM CSV
- The CSV is parsed with pandas after detecting the delimiter and header row.
- Common time column names are auto-detected (Time, Timestamp, SessionTime, etc.).
- IMU column names are detected for gyro (GyroX, YawRate, etc.) and accel (AccX, LatAcc, etc.).
- If you know your column names, you can pass them with `--aim-time-col`, `--aim-gyro-cols`, and `--aim-acc-cols`.

## Time normalization
AiM timestamps are converted to seconds from the start of the log, so both sources share a comparable time base.
