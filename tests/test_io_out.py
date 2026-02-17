from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imu_video_sync.core.models import ImuBundle, SourceMeta, TimeSeries
from imu_video_sync.io_out import write_imu_csv, write_shifted_log


def _build_imu(gyro_time, gyro, accel_time, accel) -> ImuBundle:
    gyro_series = TimeSeries(
        time_s=gyro_time,
        values=gyro,
        axes=["x", "y", "z"],
        name="gyro",
    )
    accel_series = TimeSeries(
        time_s=accel_time,
        values=accel,
        axes=["x", "y", "z"],
        name="accel",
    )
    return ImuBundle(
        gyro=gyro_series,
        accel=accel_series,
        channels={},
        meta=SourceMeta(name="test", kind="video", path=Path("video.mp4")),
    )


def test_write_imu_csv_interpolates_axes(tmp_path):
    gyro_time = np.array([0.0, 1.0, 2.0])
    gyro = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    accel_time = np.array([0.0, 2.0])
    accel = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    out = tmp_path / "video.csv"

    imu = _build_imu(gyro_time, gyro, accel_time, accel)
    write_imu_csv(out, imu)
    df = pd.read_csv(out)

    assert list(df.columns) == ["time_s", "gx", "gy", "gz", "ax", "ay", "az"]
    assert df["time_s"].iloc[1] == pytest.approx(1.0)
    assert df["ax"].iloc[1] == pytest.approx(11.5)
    assert df["ay"].iloc[1] == pytest.approx(12.5)
    assert df["az"].iloc[1] == pytest.approx(13.5)


def test_write_imu_csv_requires_data(tmp_path):
    out = tmp_path / "video.csv"
    empty = ImuBundle(gyro=None, accel=None, channels={}, meta=None)
    with pytest.raises(ValueError):
        write_imu_csv(out, empty)


def test_write_shifted_log(tmp_path):
    df = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "Val": [10, 11, 12]})
    time_s = np.array([0.0, 1.0, 2.0])
    out = tmp_path / "shifted.csv"
    write_shifted_log(out, df, "Time", time_s, 1.5)
    out_df = pd.read_csv(out)
    assert out_df["Time"].tolist() == [1.5, 2.5, 3.5]


def test_write_shifted_log_length_mismatch(tmp_path):
    df = pd.DataFrame({"Time": [0.0, 1.0], "Val": [10, 11]})
    time_s = np.array([0.0, 1.0, 2.0])
    out = tmp_path / "shifted.csv"
    with pytest.raises(ValueError):
        write_shifted_log(out, df, "Time", time_s, 1.0)
