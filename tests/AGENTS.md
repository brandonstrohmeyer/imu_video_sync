# Agent Testing Blueprint

This file is the source of truth for how tests are organized and extended in this repo. Treat it as doc-driven development: update this file first, then refactor tests to match it.

## Goals
- Standardize tests across all log and video sources.
- Keep integration coverage consistent via a single, manifest-driven suite.
- Keep fixtures minimal and deterministic.

## Test Suite Structure
- Use parameterized tests for everything that can be expressed as data.
- Keep a small number of device-specific tests only when a device requires unique parsing logic or has a documented regression.

## Manifest-Driven Integration Tests
- A single manifest defines every fixture pair used in integration tests.
- The integration test iterates the manifest and runs the same test logic for every pair.
- The manifest lives at `tests/fixtures/manifest.json`.
- The manifest suite also covers log/video `sniff`, `load`, `time` checks, and FPS detection for all manifest videos.

### Required Manifest Fields
- `id`
- `video`
- `log`
- `expected_lag_s`
- `tolerance_s` (default 0.1s if omitted)
- `video_device`
- `log_device`

### Optional Manifest Fields
- `notes`
- `min_confidence`
- `max_abs_lag_s`
- `override_window_s`
- `override_window_step`

### Integration Test Requirements
- Auto-detect the log and video sources.
- Run correlation on the default settings (auto window sizing enabled).
- Assert lag is finite.
- If `expected_lag_s` is provided, assert `abs(lag_s - expected_lag_s) <= tolerance_s`.
- If `min_confidence` is provided, assert confidence is at least that value.
- If `max_abs_lag_s` is provided, assert `abs(lag_s)` is within range.
- Emit a clear skip reason when a fixture is missing.

## Required Tests For Log Sources
- `sniff` positive and negative.
- `load` smoke returns a non-empty `ImuBundle`.
- `time_s` monotonic and normalized (first value near zero).
- `channels` presence: at least one of `gyro` or `accel`, or a documented fallback channel.
- `nan` handling: required-channel NaNs are dropped.
- `override` behavior for configurable columns.

## Required Tests For Video Sources
- `sniff` positive and negative.
- `load` smoke returns a non-empty `ImuBundle`.
- `fps` detection non-zero.
- `time_s` monotonic and normalized (first value near zero).
- `channels` presence: at least one of `gyro` or `accel`, or a clear error is raised.

## Fixtures
- Small fixtures live in `tests/fixtures/` and are used by default.
- Large or proprietary fixtures are opt-in via environment variables.
- New fixture pairs must follow: `<short_id>-<device_type>.<ext>`.
- Every fixture pair must have a manifest entry.
- Integration tests must use the smallest telemetry-preserving MP4 available.
- If a full-size copy is kept for reference, suffix it with `.full.mp4` and never list it in the manifest.

## Fixture Video Compression
- Use telemetry-preserving H.264 re-encode, remove audio, and keep the data stream.
- Verify telemetry extraction after re-encode.
- Keep size as small as possible while telemetry remains intact.
- For DJI fixtures, reattach `djmd`/`dbgi` data tracks using MP4Box after re-encoding video-only.

## Environment Variables
- `IMU_SYNC_<SOURCE>_CSV` for log files.
- `IMU_SYNC_<SOURCE>_MP4` for video files.

## When To Add Device-Specific Tests
- Only if the device has unique parsing logic or a documented regression.
- The test must explain why it cannot be covered by the manifest-driven suite.
