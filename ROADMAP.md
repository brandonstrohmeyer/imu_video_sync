# Roadmap

## Vision
This project aims to support any camera or data logger that provides time-stamped IMU data (gyro and/or accelerometer), whether embedded in video metadata or exported as telemetry. The compatibility matrix below is a working plan and will evolve as devices are validated. Our long-term goal is to support all such devices, not just a single brand or ecosystem.

## Compatibility Matrix (Working Plan)
Legend:
`Yes` = confirmed in principle.
`Model-dependent` = varies by model, firmware, or settings.
`Expected` = likely available but not yet verified in this repo.

**Cameras**
| Device / Example | Gyro Present | Gyro Accessible | Status | Notes |
| --- | --- | --- | --- | --- |
| GoPro HERO5+ | Yes | Yes (GPMF in MP4) | ✅ Supported | Current primary path via `pygpmf-oz`. |
| DJI Osmo Action (varies) | Model-dependent | Model-dependent | ⚠️ Investigate | Validate by model/firmware. Prefer pure-Python options (e.g., `pyosmogps` for GPS-only metadata). Rust deps (e.g., `telemetry-parser`) are out of scope. |
| Other action cams (Insta360, etc.) | Model-dependent | Model-dependent | 🛠️ Planned | Depends on metadata path or SDK. |

**Loggers**
| Device / Example | Gyro Present | Gyro Accessible | Status | Notes |
| --- | --- | --- | --- | --- |
| AiM CSV (e.g., XLog, Solo 2) | Model-dependent | Yes (exported channels) | ✅ Supported | Requires detectable gyro/accel columns. |
| RaceBox Mini / Mini S | Yes | Expected | 🛠️ Planned | Need export format and timebase validation. |
| RaceBox (full-size) | Unknown | Unknown | ⚠️ Investigate | Specs often mention accel only; verify gyro presence. |
| Dragy Pro | Yes | Expected | 🛠️ Planned | Confirm export format and sampling rate. |

**Apps**
| Device / Example | Gyro Present | Gyro Accessible | Status | Notes |
| --- | --- | --- | --- | --- |
| RaceChrono (phone IMU) | Phone-dependent | Yes (app export) | 🛠️ Planned | Requires phone sensor logging enabled. |

## Compatibility Criteria
To be supported, a device must provide:
- Time-stamped IMU samples with a stable timebase.
- A usable export format (CSV, JSON, or embedded metadata).
- Sufficient sample rate for correlation (ideally 50 Hz or higher).
