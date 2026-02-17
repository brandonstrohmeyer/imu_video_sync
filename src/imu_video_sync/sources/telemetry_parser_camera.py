from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..core.models import ImuBundle, SourceMeta, TimeSeries
from .base import VideoSource
from .registry import register_source

try:
    import telemetry_parser  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    telemetry_parser = None


_TIME_KEYS = ["timestamp_ms", "timestamp", "time_s", "time", "t", "ts"]
_GYRO_KEYS = ["gyro", "gyroscope", "gyro_deg_s", "gyro_dps"]
_ACCEL_KEYS = ["accel", "accelerometer", "accel_m_s2", "accel_ms2", "accl"]


@dataclass(frozen=True)
class _SeriesData:
    time_s: np.ndarray
    values: np.ndarray


def _ensure_nx3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.shape[0] == 3 and values.shape[1] != 3:
        return values.T
    return values


def _axis_names(values: np.ndarray) -> List[str]:
    values = np.asarray(values)
    if values.ndim == 1:
        return ["x"]
    n = values.shape[1]
    if n == 3:
        return ["x", "y", "z"]
    if n == 2:
        return ["x", "y"]
    if n == 1:
        return ["x"]
    return [f"a{i}" for i in range(n)]


def _to_seconds(times: Iterable[float]) -> np.ndarray:
    time_s = np.asarray(times, dtype=float)
    if time_s.size == 0:
        return time_s
    finite = np.isfinite(time_s)
    if not finite.any():
        return time_s
    max_t = float(np.nanmax(time_s[finite]))
    # Heuristic: telemetry-parser often returns milliseconds. Detect ms by
    # either a very large max timestamp or large typical step size.
    diffs = np.diff(time_s[finite])
    median_diff = float(np.nanmedian(diffs)) if diffs.size else 0.0
    if max_t > 1e5 or median_diff > 10.0:
        time_s = time_s / 1000.0
    time_s = time_s - float(time_s[0])
    return time_s


def _pick_key(data: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    return None


def _extract_from_mapping(data: Dict[str, Any]) -> Tuple[Optional[_SeriesData], Optional[_SeriesData]]:
    time_key = _pick_key(data, _TIME_KEYS)
    gyro_key = _pick_key(data, _GYRO_KEYS)
    accel_key = _pick_key(data, _ACCEL_KEYS)

    gyro_series = accel_series = None
    if time_key and gyro_key in data:
        times = _to_seconds(data[time_key])
        values = _ensure_nx3(np.asarray(data[gyro_key], dtype=float))
        gyro_series = _SeriesData(time_s=times, values=values)
    if time_key and accel_key in data:
        times = _to_seconds(data[time_key])
        values = _ensure_nx3(np.asarray(data[accel_key], dtype=float))
        accel_series = _SeriesData(time_s=times, values=values)
    return gyro_series, accel_series


def _extract_from_samples(samples: Iterable[Any]) -> Tuple[Optional[_SeriesData], Optional[_SeriesData]]:
    gyro_times: List[float] = []
    gyro_vals: List[np.ndarray] = []
    accel_times: List[float] = []
    accel_vals: List[np.ndarray] = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        time_key = _pick_key(sample, _TIME_KEYS)
        if time_key is None:
            continue
        t = sample.get(time_key)
        if t is None:
            continue

        gyro_key = _pick_key(sample, _GYRO_KEYS)
        accel_key = _pick_key(sample, _ACCEL_KEYS)

        if gyro_key and sample.get(gyro_key) is not None:
            gyro_times.append(float(t))
            gyro_vals.append(np.asarray(sample[gyro_key], dtype=float))
        if accel_key and sample.get(accel_key) is not None:
            accel_times.append(float(t))
            accel_vals.append(np.asarray(sample[accel_key], dtype=float))

    gyro_series = accel_series = None
    if gyro_times and gyro_vals:
        times = _to_seconds(gyro_times)
        values = _ensure_nx3(np.vstack(gyro_vals))
        gyro_series = _SeriesData(time_s=times, values=values)
    if accel_times and accel_vals:
        times = _to_seconds(accel_times)
        values = _ensure_nx3(np.vstack(accel_vals))
        accel_series = _SeriesData(time_s=times, values=values)
    return gyro_series, accel_series


def _extract_imu(imu_data: Any) -> Tuple[Optional[_SeriesData], Optional[_SeriesData]]:
    if isinstance(imu_data, dict):
        if "samples" in imu_data and isinstance(imu_data["samples"], list):
            return _extract_from_samples(imu_data["samples"])
        return _extract_from_mapping(imu_data)
    if isinstance(imu_data, (list, tuple)):
        return _extract_from_samples(imu_data)
    return None, None


def extract_telemetry_imu(path: Path) -> ImuBundle:
    if telemetry_parser is None:
        raise ImportError("telemetry-parser is not installed. Install it to use this source.")

    parser = telemetry_parser.Parser(str(path))
    imu_data = parser.normalized_imu()
    gyro_series, accel_series = _extract_imu(imu_data)

    if gyro_series is None and accel_series is None:
        raise ValueError("No IMU data found in telemetry-parser output.")

    gyro_ts = None
    accel_ts = None
    if gyro_series is not None:
        gyro_ts = TimeSeries(
            time_s=gyro_series.time_s,
            values=gyro_series.values,
            axes=_axis_names(gyro_series.values),
            name="gyro",
            units="deg/s",
        )
    if accel_series is not None:
        accel_ts = TimeSeries(
            time_s=accel_series.time_s,
            values=accel_series.values,
            axes=_axis_names(accel_series.values),
            name="accel",
            units="m/s^2",
        )

    notes: List[str] = []
    camera = getattr(parser, "camera", None)
    model = getattr(parser, "model", None)
    if camera:
        notes.append(f"camera={camera}")
    if model:
        notes.append(f"model={model}")

    meta = SourceMeta(
        name="telemetry_parser",
        kind="video",
        path=Path(path),
        notes=notes,
    )

    return ImuBundle(
        gyro=gyro_ts,
        accel=accel_ts,
        channels={},
        meta=meta,
    )


@register_source("video")
class TelemetryParserVideoSource(VideoSource):
    name = "telemetry_parser"

    @classmethod
    def sniff(cls, path: Path) -> float:
        if path.suffix.lower() in {".mp4", ".mov", ".mxf"}:
            return 0.6
        return 0.0

    @classmethod
    def load(cls, path: Path, **opts) -> ImuBundle:
        return extract_telemetry_imu(Path(path))
