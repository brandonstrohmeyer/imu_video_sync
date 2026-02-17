from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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


def write_gopro_csv(
    path: Path,
    gyro_time_s: Optional[np.ndarray],
    gyro: Optional[np.ndarray],
    accel_time_s: Optional[np.ndarray],
    accel: Optional[np.ndarray],
) -> None:
    # Write a unified GoPro IMU CSV, interpolating to a shared time base if needed.
    if gyro is None and accel is None:
        raise ValueError("No GoPro IMU data to write.")

    base_time = None
    if gyro is not None and gyro_time_s is not None:
        gyro_time_s, gyro = _clean_time_values(gyro_time_s, gyro)
        base_time = gyro_time_s
    if base_time is None and accel is not None and accel_time_s is not None:
        accel_time_s, accel = _clean_time_values(accel_time_s, accel)
        base_time = accel_time_s

    if base_time is None:
        raise ValueError("GoPro IMU timebase missing.")

    data = {"time_s": base_time}

    if gyro is not None:
        if gyro_time_s is None:
            raise ValueError("Gyro data missing timebase.")
        if not np.array_equal(gyro_time_s, base_time):
            gyro = _interp_axes(gyro_time_s, gyro, base_time)
        gyro = np.asarray(gyro, dtype=float)
        if gyro.ndim == 1:
            data["gx"] = gyro
        else:
            if gyro.shape[1] >= 1:
                data["gx"] = gyro[:, 0]
            if gyro.shape[1] >= 2:
                data["gy"] = gyro[:, 1]
            if gyro.shape[1] >= 3:
                data["gz"] = gyro[:, 2]

    if accel is not None:
        if accel_time_s is None:
            raise ValueError("Accel data missing timebase.")
        if not np.array_equal(accel_time_s, base_time):
            accel = _interp_axes(accel_time_s, accel, base_time)
        accel = np.asarray(accel, dtype=float)
        if accel.ndim == 1:
            data["ax"] = accel
        else:
            if accel.shape[1] >= 1:
                data["ax"] = accel[:, 0]
            if accel.shape[1] >= 2:
                data["ay"] = accel[:, 1]
            if accel.shape[1] >= 3:
                data["az"] = accel[:, 2]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def write_shifted_csv(
    path: Path, df: pd.DataFrame, time_col: str, time_s: np.ndarray, lag_seconds: float
) -> None:
    out = df.copy()
    # Overwrite the time column with seconds-from-start plus the estimated lag.
    if len(time_s) != len(out):
        raise ValueError("Time array length does not match AiM dataframe length.")
    out[time_col] = time_s + float(lag_seconds)
    out.to_csv(path, index=False)
