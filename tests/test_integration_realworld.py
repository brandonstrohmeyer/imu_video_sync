import os
from pathlib import Path

import numpy as np
import pytest

from imu_video_sync.aim_csv import load_aim_csv
from imu_video_sync.cli import _compute_metrics
from imu_video_sync.gopro_extract import GoproIMU, extract_gopro_imu


def _env_path(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default


def test_gopro_metadata_extraction_smoke():
    mp4_path = _env_path(
        "IMU_SYNC_MP4_SMOKE",
        Path("tests/fixtures/gopro.mp4"),
    )
    if not mp4_path.exists():
        pytest.skip(f"Missing MP4 smoke fixture: {mp4_path}")

    imu = extract_gopro_imu(str(mp4_path))
    has_gyro = imu.gyro is not None and imu.gyro.size > 0
    has_accel = imu.accel is not None and imu.accel.size > 0
    assert has_gyro or has_accel


def test_realworld_correlation_fixture():
    npz_path = _env_path(
        "IMU_SYNC_GOPRO_NPZ",
        Path("tests/fixtures/gopro_imu_full.npz"),
    )
    csv_path = _env_path(
        "IMU_SYNC_AIM_CSV",
        Path("tests/fixtures/aim.csv"),
    )
    if not npz_path.exists():
        pytest.skip(f"Missing GoPro IMU fixture: {npz_path}")
    if not csv_path.exists():
        pytest.skip(f"Missing AiM CSV fixture: {csv_path}")

    expected_lag = float(os.getenv("IMU_SYNC_EXPECTED_LAG", "376.240"))
    tolerance = float(os.getenv("IMU_SYNC_LAG_TOLERANCE", "0.1"))

    aim = load_aim_csv(csv_path)
    data = np.load(npz_path)

    gopro = GoproIMU(
        gyro_time_s=data.get("gyro_time_s"),
        accel_time_s=data.get("accel_time_s"),
        gyro=data.get("gyro"),
        accel=data.get("accel"),
        backend=str(data.get("backend", "gpmf")),
    )

    metrics = _compute_metrics(
        aim=aim,
        gopro=gopro,
        signal="gyroMag",
        fs=50.0,
        window_s=360.0,
        lowpass_hz=8.0,
        highpass_hz=0.2,
        max_lag_s=600.0,
        start_override=None,
        auto_window=True,
        window_step_s=20.0,
    )

    lag = metrics["lag_seconds"]
    assert abs(lag - expected_lag) <= tolerance
