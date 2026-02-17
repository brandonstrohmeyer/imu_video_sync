# Data Sources

This tool uses modular, in-repo sources for both video and log inputs. Each source is responsible for producing a common IMU bundle so the correlation pipeline stays DRY.

## Video Sources

### `gopro_gpmf` (built-in)
- Parsed via `pygpmf-oz`.
- The MP4 contains GPMF (GoPro Metadata Format) packets with sensor data.
- The source extracts gyro and accelerometer samples and their timestamps.
- Timestamps are normalized so the first sample is at `time_s = 0.0`.

**What we keep**
- `gyro`: 3-axis rotation (gx, gy, gz).
- `accel`: 3-axis acceleration (ax, ay, az).
- `time_s`: seconds from clip start.

**Missing data**
Some GoPro clips may lack gyro or accel. The tool falls back to whatever is available.

## Planned Video Sources

### `dji_osmo_action` (research)
- GPS metadata extraction may be possible via `pyosmogps` (pure Python).
- `telemetry-parser` is not suitable due to its Rust dependency.
- IMU availability in metadata is still unverified.

## Log Sources

### `aim_csv` (built-in)
- The CSV is parsed with pandas after detecting the delimiter and header row.
- Common time column names are auto-detected (Time, Timestamp, SessionTime, etc.).
- IMU column names are detected for gyro (GyroX, YawRate, etc.) and accel (AccX, LatAcc, etc.).
- If you know your column names, you can pass them with `--log-time-col`, `--log-gyro-cols`, and `--log-acc-cols`.

## Time normalization
Log timestamps are converted to seconds from the start of the log, so both sources share a comparable time base.

## Forcing a source
Use `--video-source` or `--log-source` to force a specific backend by name when more than one source could match.
