from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np

from .models import ImuBundle, TimeSeries


SIGNAL_PRIORITY = ["gyroMag", "yawRate", "latAcc", "accMag", "gyroZ"]


@dataclass(frozen=True)
class DerivedSignal:
    name: str
    time_s: np.ndarray
    values: np.ndarray
    derived_from: str
    axes: list[str]


def _series_axes(series: TimeSeries) -> list[str]:
    if series.axes:
        return list(series.axes)
    values = np.asarray(series.values)
    if values.ndim == 1:
        return ["x"]
    return [f"a{i}" for i in range(values.shape[1])]


def _magnitude(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.abs(values)
    if values.shape[1] == 1:
        return np.abs(values[:, 0])
    return np.linalg.norm(values, axis=1)


def _last_axis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    return values[:, -1]


def available_signals(imu: ImuBundle) -> Set[str]:
    available: Set[str] = set()
    if imu.gyro is not None:
        available.update(["gyroMag", "gyroZ", "yawRate"])
    if imu.accel is not None:
        available.add("accMag")
    if "yaw_rate" in imu.channels:
        available.add("yawRate")
    if "lat_acc" in imu.channels:
        available.add("latAcc")
    return available


def choose_signal(
    requested: str, imu_a: ImuBundle, imu_b: ImuBundle
) -> Tuple[str, Optional[str]]:
    available_a = available_signals(imu_a)
    available_b = available_signals(imu_b)

    if requested in available_a and requested in available_b:
        return requested, None

    for candidate in SIGNAL_PRIORITY:
        if candidate in available_a and candidate in available_b:
            return candidate, f"Requested {requested} not available in both files. Using {candidate} instead."

    raise ValueError("No compatible signal found between log and video data.")


def derive_signal(imu: ImuBundle, name: str) -> DerivedSignal:
    if name == "gyroMag":
        if imu.gyro is None:
            raise ValueError("Gyro channels not available.")
        axes = _series_axes(imu.gyro)
        return DerivedSignal(
            name=name,
            time_s=imu.gyro.time_s,
            values=_magnitude(imu.gyro.values),
            derived_from="gyro",
            axes=axes,
        )
    if name == "accMag":
        if imu.accel is None:
            raise ValueError("Accel channels not available.")
        axes = _series_axes(imu.accel)
        return DerivedSignal(
            name=name,
            time_s=imu.accel.time_s,
            values=_magnitude(imu.accel.values),
            derived_from="accel",
            axes=axes,
        )
    if name == "gyroZ":
        if imu.gyro is None:
            raise ValueError("Gyro channels not available.")
        axes = _series_axes(imu.gyro)
        axis = axes[-1] if axes else "z"
        return DerivedSignal(
            name=name,
            time_s=imu.gyro.time_s,
            values=_last_axis(imu.gyro.values),
            derived_from="gyro_z",
            axes=[axis],
        )
    if name == "yawRate":
        if "yaw_rate" in imu.channels:
            series = imu.channels["yaw_rate"]
            axes = _series_axes(series)
            return DerivedSignal(
                name=name,
                time_s=series.time_s,
                values=np.asarray(series.values, dtype=float).reshape(-1),
                derived_from="yaw_rate",
                axes=axes,
            )
        if imu.gyro is None:
            raise ValueError("Yaw rate not available.")
        axes = _series_axes(imu.gyro)
        axis = axes[-1] if axes else "z"
        return DerivedSignal(
            name=name,
            time_s=imu.gyro.time_s,
            values=_last_axis(imu.gyro.values),
            derived_from="gyro_z",
            axes=[axis],
        )
    if name == "latAcc":
        if "lat_acc" in imu.channels:
            series = imu.channels["lat_acc"]
            axes = _series_axes(series)
            return DerivedSignal(
                name=name,
                time_s=series.time_s,
                values=np.asarray(series.values, dtype=float).reshape(-1),
                derived_from="lat_acc",
                axes=axes,
            )
        if imu.accel is None:
            raise ValueError("Lateral accel not available.")
        axes = _series_axes(imu.accel)
        axis = axes[-1] if axes else "y"
        return DerivedSignal(
            name=name,
            time_s=imu.accel.time_s,
            values=_last_axis(imu.accel.values),
            derived_from="accel_axis",
            axes=[axis],
        )

    raise ValueError(f"Unsupported signal type: {name}")
