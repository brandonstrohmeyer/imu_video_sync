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
- Integration fixtures are enumerated in `tests/fixtures/manifest.json` and run via a single parameterized test.
- Manifest entries should include `expected_lag_s` and a `tolerance_s` (default 0.1s).
- The manifest-driven integration test also covers log/video `sniff`, `load`, `time` checks, and FPS detection for all manifest videos.

## Fixtures and Environment Variables
- Small fixtures live in `tests/fixtures/` and are used by default.
- Keep video fixtures as small as possible while preserving telemetry; re-encode if needed.
- Large or proprietary fixtures are opt-in via environment variables.
- New fixtures should follow the naming convention: `<short_id>-<device_type>.<ext>` so video/log pairs are easy to match.
  - Example: `5f2167b0-gopro-hero8-black.mp4` and `5f2167b0-racebox.csv`.
- Every integration fixture pair must have a manifest entry.
- If a full-size copy is kept for reference, suffix it with `.full.mp4` and never reference it from the manifest.
- Each heavy test must skip gracefully if the fixture is missing.
- Suggested env vars:
- `IMU_SYNC_<SOURCE>_CSV` for log files.
- `IMU_SYNC_<SOURCE>_MP4` for video files.

## Fixture Video Compression (Telemetry-Preserving)
We keep video fixtures minimal while retaining the telemetry data stream. Audio is always removed.
- Inspect streams to confirm a telemetry data stream exists (GoPro uses `gpmd`).
  - `ffprobe -hide_banner -i input.mp4`
- Re-encode video to H.264, downscale, remove audio, and copy the data stream.
  - `ffmpeg -i input.mp4 -map 0 -map -0:a -c:v libx264 -crf 36 -preset veryslow -vf "scale=640:-2" -c:d copy -movflags +faststart output.mp4`
- DJI telemetry tracks (`djmd`/`dbgi`) cannot be re-muxed by ffmpeg; use MP4Box to reattach them after re-encode.
  - `ffmpeg -i input.mp4 -map 0:0 -c:v libx264 -crf 36 -preset veryslow -vf "scale=640:-2" -movflags +faststart video_only.mp4`
  - `MP4Box -add video_only.mp4 -add input.mp4#trackID=3 -add input.mp4#trackID=4 -new output.mp4`
- Verify telemetry extraction from the re-encoded file before committing the fixture.
  - `python -m telemetry_parser output.mp4`
- If telemetry is missing, ensure the data stream is mapped and copied (`-map 0` and `-c:d copy`), then retry with a higher resolution or a lower CRF.
- Target size: as small as possible while telemetry remains intact; typically a few MB for a 2–3 minute clip.

## Fixture Inventory
- `tests/fixtures/aim.csv`
- `tests/fixtures/5f2167b0-gopro-hero8-black.mp4`
- `tests/fixtures/5f2167b0-racebox.csv`
- `tests/fixtures/c1cba136-dji-osmo-action5-pro.mp4`
- `tests/fixtures/c1cba136-racechrono.csv`
- `tests/fixtures/7e4858b7-gopro-hero12-black.mp4`
- `tests/fixtures/7e4858b7-aim.csv`

## Performance Expectations
- `sniff` should be fast and avoid full-file parsing.
- `load` should be deterministic and not write outside temporary directories.
- Heavy decoding (e.g., video metadata extraction) should be isolated to smoke tests.
