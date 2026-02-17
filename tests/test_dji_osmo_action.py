import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from imu_video_sync.sources.dji_osmo_action import DjiOsmoActionSource, extract_dji_accel


def _env_path(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default


def _has_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def test_dji_sniff_smoke():
    mp4_path = _env_path(
        "IMU_SYNC_DJI_MP4",
        Path("tests/fixtures/dji.mov"),
    )
    if not mp4_path.exists() or not _has_ffmpeg():
        pytest.skip("Missing DJI MP4 fixture or ffmpeg/ffprobe not available.")

    score = DjiOsmoActionSource.sniff(mp4_path)
    assert score > 0.0


def test_dji_extract_accel_smoke():
    mp4_path = _env_path(
        "IMU_SYNC_DJI_MP4",
        Path("tests/fixtures/dji.mov"),
    )
    if not mp4_path.exists() or not _has_ffmpeg():
        pytest.skip("Missing DJI MP4 fixture or ffmpeg/ffprobe not available.")

    imu = extract_dji_accel(mp4_path)
    assert imu.accel is not None
    assert imu.accel.values.shape[1] == 3
    assert imu.accel.time_s[0] == pytest.approx(0.0)
    assert np.all(np.diff(imu.accel.time_s) >= 0)
