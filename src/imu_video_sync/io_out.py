from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .core.models import ImuBundle, TimeSeries


def _clean_time_values(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    # Drop NaNs and sort by time to ensure a clean timebase.
    mask = np.isfinite(time_s)
    if values.ndim == 1:
        mask &= np.isfinite(values)
    else:
        mask &= np.all(np.isfinite(values), axis=1)
    time_s = time_s[mask]
    values = values[mask]
    if time_s.size == 0:
        raise ValueError("No valid samples after cleaning GoPro IMU data.")
    order = np.argsort(time_s)
    time_s = time_s[order]
    values = values[order]
    # Remove duplicate timestamps to keep interpolation stable.
    uniq, idx = np.unique(time_s, return_index=True)
    return uniq, values[idx]


def _interp_axes(time_src: np.ndarray, values: np.ndarray, time_dst: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.interp(time_dst, time_src, values)
    # Interpolate each axis independently.
    out = []
    for axis in range(values.shape[1]):
        out.append(np.interp(time_dst, time_src, values[:, axis]))
    return np.column_stack(out)


def _axis_columns(prefix: str, axes: list[str]) -> list[str]:
    if axes == ["x", "y", "z"]:
        return [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
    cleaned = []
    for idx, axis in enumerate(axes):
        name = "".join(ch for ch in axis.lower() if ch.isalnum())
        if not name:
            name = str(idx)
        cleaned.append(f"{prefix}_{name}")
    return cleaned


def _series_to_columns(series: TimeSeries, prefix: str) -> dict[str, np.ndarray]:
    values = np.asarray(series.values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    axes = series.axes if series.axes else [str(i) for i in range(values.shape[1])]
    cols = _axis_columns(prefix, axes)
    return {cols[i]: values[:, i] for i in range(values.shape[1])}


def write_imu_csv(path: Path, imu: ImuBundle) -> None:
    # Write a unified IMU CSV, interpolating to a shared time base if needed.
    if imu.gyro is None and imu.accel is None:
        raise ValueError("No IMU data to write.")

    base_time = None
    gyro = accel = None
    if imu.gyro is not None:
        gyro_time_s, gyro = _clean_time_values(imu.gyro.time_s, imu.gyro.values)
        base_time = gyro_time_s
    if base_time is None and imu.accel is not None:
        accel_time_s, accel = _clean_time_values(imu.accel.time_s, imu.accel.values)
        base_time = accel_time_s

    if base_time is None:
        raise ValueError("IMU timebase missing.")

    data = {"time_s": base_time}

    if imu.gyro is not None:
        gyro_time_s = imu.gyro.time_s
        if not np.array_equal(gyro_time_s, base_time):
            gyro = _interp_axes(gyro_time_s, imu.gyro.values, base_time)
        else:
            gyro = imu.gyro.values
        gyro = TimeSeries(
            time_s=base_time,
            values=np.asarray(gyro, dtype=float),
            axes=imu.gyro.axes,
            name=imu.gyro.name,
        )
        data.update(_series_to_columns(gyro, "g"))

    if imu.accel is not None:
        accel_time_s = imu.accel.time_s
        if not np.array_equal(accel_time_s, base_time):
            accel = _interp_axes(accel_time_s, imu.accel.values, base_time)
        else:
            accel = imu.accel.values
        accel = TimeSeries(
            time_s=base_time,
            values=np.asarray(accel, dtype=float),
            axes=imu.accel.axes,
            name=imu.accel.name,
        )
        data.update(_series_to_columns(accel, "a"))

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def write_shifted_log(
    path: Path, df: pd.DataFrame, time_col: str, time_s: np.ndarray, lag_seconds: float
) -> None:
    out = df.copy()
    # Overwrite the time column with seconds-from-start plus the estimated lag.
    if len(time_s) != len(out):
        raise ValueError("Time array length does not match AiM dataframe length.")
    out[time_col] = time_s + float(lag_seconds)
    out.to_csv(path, index=False)
