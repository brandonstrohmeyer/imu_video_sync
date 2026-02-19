# CHANGELOG.md

## v0.2.0
**Added**
- Minimal GUI with Windows/macOS launch support.
- RaceBox CSV log source.

**Changed**
- SciPy removed from runtime; correlation + filtering now use NumPy-only implementations.
- CLI auto-tunes defaults for window size, max-lag, sample rate, and filter cutoffs.
- Window-scanning logic updated for short clips and improved candidate selection.

**Removed**
- Legacy fixtures/tests replaced by manifest-based integration coverage.

## v0.1.0
- Initial Release with support for:
  - Cameras
    - GoPro HERO5+
    - DJI Osmo Action 4+
  - Loggers
    - AiM CSV (e.g., XLog, Solo 2)
    - RaceChrono CSV (phone IMU)
