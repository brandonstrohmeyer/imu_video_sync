from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imu_video_sync.sources.aim_csv import detect_delimiter, detect_header_row, load_aim_csv


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_detect_delimiter_semicolon(tmp_path):
    content = "a;b;c\n1;2;3\n"
    path = tmp_path / "sample.csv"
    _write_text(path, content)
    assert detect_delimiter(path) == ";"


def test_load_aim_csv_with_header_offset_and_ms_time(tmp_path):
    content = "\n".join(
        [
            "junk line 1",
            "junk line 2",
            "Time,GyroX,GyroY,GyroZ,AccX,AccY,AccZ",
            "1000,1,2,3,4,5,6",
            "2000,1,2,3,4,5,6",
            "3000,1,2,3,4,5,6",
        ]
    )
    path = tmp_path / "aim.csv"
    _write_text(path, content)

    delimiter = detect_delimiter(path)
    header_idx = detect_header_row(path, delimiter)
    assert header_idx == 2

    log = load_aim_csv(path)
    assert np.allclose(log.time_s, np.array([0.0, 1.0, 2.0]))
    assert log.imu.gyro is not None
    assert log.imu.accel is not None
    assert log.imu.gyro.values.shape == (3, 3)
    assert log.imu.accel.values.shape == (3, 3)


def test_load_aim_csv_missing_override_column(tmp_path):
    df = pd.DataFrame({"Time": [0, 1, 2], "GyroX": [0, 0, 0]})
    path = tmp_path / "aim.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_aim_csv(path, time_col="Nope")
