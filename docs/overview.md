# Overview

`imu-video-sync` aligns a camera video with a telemetry log by comparing motion patterns from each device's IMU (gyro/accelerometer).

## High-level Flow
1. Read video telemetry from the MP4 and extract gyro/accel samples with timestamps.
2. Read the log CSV and detect time plus IMU columns.
3. Pick a signal (default `gyroMag`, or try multiple signals).
4. Resample both signals to a common sample rate.
5. Filter and normalize so shapes are comparable.
6. Cross-correlate to find the time lag that best aligns the signals.
7. Report the lag, confidence metrics, and a confidence rating.

## What is being compared
The tool compares one 1-D trace from each source. By default this is `gyroMag`, which is the magnitude of the gyro vector:
`gyroMag = sqrt(gx^2 + gy^2 + gz^2)`

This is orientation-agnostic, so it still matches well even if axes do not line up exactly between devices.

## Why this works
Both the video and the data log capture the same physical motion. The IMU signals have similar patterns when the car turns, brakes, and accelerates. Cross-correlation finds the time shift that best matches those patterns.

## Window scanning (not random)
The tool does not rely on one window or a lap length. It scans many fixed-length windows across the log, scores each alignment, and then builds a consensus from the best windows. By default it auto-selects a window length based on the clip duration; use `--window` to force a specific length or `--no-auto-window-size` to disable auto-selection. If the requested window is too long for the data, it is auto-shrunk; if the window is effectively the full clip, the tool falls back to a single-window estimate.

## Auto-tuned defaults
When you use the defaults, the CLI also auto-tunes `--max-lag`, `--fs`, and filter cutoffs to better fit the clip length and sample rates.

## Outputs
The CLI prints a sync summary and a RaceRender-friendly offset line. Optional outputs include a video IMU CSV, a shifted log CSV, and a diagnostic plot.
