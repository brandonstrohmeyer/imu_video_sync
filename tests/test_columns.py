from imu_video_sync.sources.aim_csv import detect_columns


def test_detect_columns_basic():
    # Basic time/gyro/accel detection with common names.
    cols = ["Time", "GyroX", "Gyro Y", "GyroZ", "AccX", "AccY", "AccZ"]
    detected = detect_columns(cols)
    assert detected.time_col == "Time"
    assert detected.gyro_cols == ["GyroX", "Gyro Y", "GyroZ"]
    assert detected.acc_cols == ["AccX", "AccY", "AccZ"]
    assert detected.special_cols["yaw_rate"] is None


def test_detect_columns_specials():
    # Special channels (yaw rate, lateral/longitudinal accel) should be recognized.
    cols = ["Log Time", "Yaw Rate", "LatAcc", "LongAcc"]
    detected = detect_columns(cols)
    assert detected.time_col == "Log Time"
    assert "Yaw Rate" in detected.gyro_cols
    assert detected.special_cols["lat_acc"] == "LatAcc"
    assert detected.special_cols["long_acc"] == "LongAcc"
