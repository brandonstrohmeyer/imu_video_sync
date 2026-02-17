from pathlib import Path

import numpy as np
import pytest

from imu_video_sync.core.models import ImuBundle, SourceMeta, TimeSeries
from imu_video_sync.core.signals import SIGNAL_PRIORITY, available_signals, choose_signal, derive_signal


def _build_log_imu() -> ImuBundle:
    time_s = np.array([0.0, 1.0, 2.0])
    gyro = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    accel = np.array([[0.0, 0.1, 0.2], [0.0, 0.2, 0.4], [0.0, 0.3, 0.6]])
    lat_acc = np.array([0.1, 0.2, 0.3])
    return ImuBundle(
        gyro=TimeSeries(time_s=time_s, values=gyro, axes=["GyroX", "GyroY", "GyroZ"], name="gyro"),
        accel=TimeSeries(time_s=time_s, values=accel, axes=["AccX", "AccY", "AccZ"], name="accel"),
        channels={
            "lat_acc": TimeSeries(time_s=time_s, values=lat_acc, axes=["LatAcc"], name="lat_acc"),
        },
        meta=SourceMeta(name="test", kind="log", path=Path("log.csv")),
    )


def _build_video_imu(with_accel: bool = True) -> ImuBundle:
    time_s = np.array([0.0, 1.0, 2.0])
    gyro = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    accel = np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]) if with_accel else None
    return ImuBundle(
        gyro=TimeSeries(time_s=time_s, values=gyro, axes=["x", "y", "z"], name="gyro"),
        accel=TimeSeries(time_s=time_s, values=accel, axes=["x", "y", "z"], name="accel") if accel is not None else None,
        channels={},
        meta=SourceMeta(name="test", kind="video", path=Path("video.mp4")),
    )


def test_available_signals_sets():
    log_imu = _build_log_imu()
    video_imu = _build_video_imu()
    log_set = available_signals(log_imu)
    video_set = available_signals(video_imu)
    assert {"gyroMag", "gyroZ", "yawRate", "accMag", "latAcc"} <= log_set
    assert {"gyroMag", "gyroZ", "yawRate", "accMag"} <= video_set


def test_choose_signal_fallback_warning():
    log_imu = _build_log_imu()
    video_imu = _build_video_imu(with_accel=False)
    signal, warning = choose_signal("accMag", log_imu, video_imu)
    assert signal == SIGNAL_PRIORITY[0]
    assert warning is not None


def test_choose_signal_no_overlap():
    log_imu = _build_log_imu()
    video_imu = _build_video_imu(with_accel=False)
    video_imu = ImuBundle(gyro=None, accel=None, channels={}, meta=video_imu.meta)
    with pytest.raises(ValueError):
        choose_signal("gyroMag", log_imu, video_imu)


def test_derive_signal_outputs():
    log_imu = _build_log_imu()
    video_imu = _build_video_imu()
    log_mag = derive_signal(log_imu, "gyroMag")
    assert log_mag.values.shape == (3,)
    video_z = derive_signal(video_imu, "gyroZ")
    assert video_z.time_s.shape == (3,)
    assert video_z.values.shape == (3,)
