import os
from pathlib import Path

import numpy as np
import pytest

from imu_video_sync.sources.telemetry_parser_camera import (
    TelemetryParserVideoSource,
    extract_telemetry_imu,
)


def _env_path(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default


def test_telemetry_parser_sniff_smoke():
    mp4_path = _env_path(
        "IMU_SYNC_GOPRO_MP4",
        Path("tests/fixtures/gopro.mp4"),
    )
    if not mp4_path.exists():
        pytest.skip(f"Missing GoPro fixture: {mp4_path}")

    score = TelemetryParserVideoSource.sniff(mp4_path)
    assert score > 0.0


@pytest.mark.parametrize(
    "fixture_path",
    [
        Path("tests/fixtures/gopro.mp4"),
        Path("tests/fixtures/dji.mp4"),
    ],
)
def test_telemetry_parser_load_smoke(fixture_path: Path):
    if not fixture_path.exists():
        pytest.skip(f"Missing camera fixture: {fixture_path}")

    imu = extract_telemetry_imu(fixture_path)
    has_gyro = imu.gyro is not None and imu.gyro.values.size > 0
    has_accel = imu.accel is not None and imu.accel.values.size > 0
    assert has_gyro or has_accel

    if imu.gyro is not None:
        assert imu.gyro.time_s[0] == pytest.approx(0.0)
        assert np.all(np.diff(imu.gyro.time_s) >= 0)
    if imu.accel is not None:
        assert imu.accel.time_s[0] == pytest.approx(0.0)
        assert np.all(np.diff(imu.accel.time_s) >= 0)
