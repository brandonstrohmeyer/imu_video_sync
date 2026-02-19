# Data Sources

This tool uses modular, in-repo sources for both video and log inputs. Each source is responsible for producing a common IMU bundle so the correlation pipeline stays DRY.

## Auto-Detection (`sniff`)
Each source implements a `sniff(path)` method that returns a confidence score between `0.0` and `1.0`. When you do not force a source, the loader calls `sniff()` on every registered source and picks the one with the highest score. Use `--video-source` or `--log-source` to override this selection if needed.

## Video Sources

### `telemetry_parser` (built-in)
- Parsed via `telemetry-parser`.
- Extracts gyro and accelerometer data from supported camera files.
- Timestamps are normalized so the first sample is at `time_s = 0.0`.

**What we keep**
- `gyro`: 3-axis rotation (gx, gy, gz).
- `accel`: 3-axis acceleration (ax, ay, az).
- `time_s`: seconds from clip start.

**Missing data**
Some clips may lack gyro or accel. The tool falls back to whatever is available.

## Log Sources

### `aim_csv` (built-in)
- The CSV is parsed with pandas after detecting the delimiter and header row.
- Common time column names are auto-detected (Time, Timestamp, SessionTime, etc.).
- IMU column names are detected for gyro (GyroX, YawRate, etc.) and accel (AccX, LatAcc, etc.).
- If you know your column names, you can pass them with `--log-time-col`, `--log-gyro-cols`, and `--log-acc-cols`.

### `racechrono_csv` (built-in)
- Parses RaceChrono CSV exports with dynamic headers.
- Detects `Time (s)` (or `Elapsed time (s)`) and IMU columns labeled `*acc` and `*gyro`.
- Normalizes timestamps to seconds from the start of the log.

### `racebox_csv` (built-in)
- Parses RaceBox CSV exports.
- Detects `Time` plus `GyroX/GyroY/GyroZ` and `X/Y/Z` accel axes.
- Normalizes timestamps to seconds from the start of the log.

## Time normalization
Log timestamps are converted to seconds from the start of the log, so both sources share a comparable time base.

## Forcing a source
Use `--video-source` or `--log-source` to force a specific backend by name when more than one source could match.
