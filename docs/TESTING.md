# Testing Standards

This document defines the minimum, standardized tests required for every source plugin. The goal is to keep test coverage consistent as new log and video sources are added.

## Scope
- Applies to all sources registered under `imu_video_sync.sources`.
- Covers smoke, parsing, and behavioral correctness for log and video plugins.
- Allows opt-in heavy fixtures via environment variables.

## Required Tests: Log Sources
- `sniff` positive: the plugin should score highly on a known-good log file for that source.
- `sniff` negative: the plugin should score low on an unrelated CSV (or a file of the wrong type).
- `load` smoke: loading a known-good file returns a `LogData` with a non-empty `ImuBundle`.
- `time` monotonic: `time_s` must be strictly increasing or non-decreasing after cleaning.
- `time` normalization: `time_s[0]` must be `0.0` (within a small tolerance).
- `channels` presence: at least one of `gyro` or `accel` exists, or a documented fallback channel (e.g., `lat_acc`) exists.
- `nan` handling: rows with NaNs in required channels are dropped (test with a tiny synthetic CSV).
- `override` behavior: optional column overrides (e.g., `--log-gyro-cols`) are respected and validated.

## Required Tests: Video Sources
- `sniff` positive: the plugin should score highly on a known-good video file for that source.
- `sniff` negative: the plugin should score low on an unrelated MP4/MOV.
- `load` smoke: loading a known-good file returns an `ImuBundle` with at least one IMU series.
- `fps` detection: detect a non-zero frame rate from the video fixture (telemetry-parser FrameInfo only).
- `time` monotonic: returned `time_s` must be strictly increasing or non-decreasing.
- `time` normalization: `time_s[0]` must be `0.0` (within a small tolerance).
- `channels` presence: at least one of `gyro` or `accel` exists; otherwise, the plugin must raise a clear error.
 - For telemetry-parser, use the GoPro and DJI fixtures to validate both brands.

## Required Tests: Cross-Source Integration
- `signal` availability: `available_signals(log) & available_signals(video)` must be non-empty for at least one supported log/video pairing.
- `correlation` smoke: run `_compute_metrics` on a known-good pair and assert a finite lag.
- `correlation` regression: when fixtures are stable, pin an expected lag with a tolerance.

## Fixtures and Environment Variables
- Small fixtures live in `tests/fixtures/` and are used by default.
- Keep video fixtures small (target ~20-25 MB) to avoid bloating the repo; re-encode if needed.
- Large or proprietary fixtures are opt-in via environment variables.
- Each heavy test must skip gracefully if the fixture is missing.
- Suggested env vars:
- `IMU_SYNC_<SOURCE>_CSV` for log files.
- `IMU_SYNC_<SOURCE>_MP4` for video files.

## Fixture Inventory
- `tests/fixtures/aim.csv`
- `tests/fixtures/gopro.mp4`
- `tests/fixtures/gopro_imu_full.npz`
- `tests/fixtures/racechrono.csv`
- `tests/fixtures/dji.mp4`

## Performance Expectations
- `sniff` should be fast and avoid full-file parsing.
- `load` should be deterministic and not write outside temporary directories.
- Heavy decoding (e.g., video metadata extraction) should be isolated to smoke tests.
