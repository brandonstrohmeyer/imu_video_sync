import os
from pathlib import Path

import numpy as np
import pytest

from imu_video_sync.sources.racechrono_csv import RaceChronoCsvSource, load_racechrono_csv


def _env_path(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default


def test_racechrono_sniff_smoke():
    csv_path = _env_path(
        "IMU_SYNC_RACECHRONO_CSV",
        Path("tests/fixtures/racechrono.csv"),
    )
    if not csv_path.exists():
        pytest.skip(f"Missing RaceChrono CSV fixture: {csv_path}")

    score = RaceChronoCsvSource.sniff(csv_path)
    assert score >= 0.5


def test_racechrono_load_smoke():
    csv_path = _env_path(
        "IMU_SYNC_RACECHRONO_CSV",
        Path("tests/fixtures/racechrono.csv"),
    )
    if not csv_path.exists():
        pytest.skip(f"Missing RaceChrono CSV fixture: {csv_path}")

    log = load_racechrono_csv(csv_path)
    assert log.time_s.size > 0
    assert log.time_s[0] == pytest.approx(0.0)
    assert np.all(np.diff(log.time_s) >= 0)
    assert log.imu.gyro is not None or log.imu.accel is not None
