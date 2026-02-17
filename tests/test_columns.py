from imu_video_sync.aim_csv import detect_columns


def test_detect_columns_basic():
    # Basic time/gyro/accel detection with common names.
    cols = ["Time", "GyroX", "Gyro Y", "GyroZ", "AccX", "AccY", "AccZ"]
    time_col, gyro_cols, acc_cols, special = detect_columns(cols)
    assert time_col == "Time"
    assert gyro_cols == ["GyroX", "Gyro Y", "GyroZ"]
    assert acc_cols == ["AccX", "AccY", "AccZ"]
    assert special["yaw_rate"] is None


def test_detect_columns_specials():
    # Special channels (yaw rate, lateral/longitudinal accel) should be recognized.
    cols = ["Log Time", "Yaw Rate", "LatAcc", "LongAcc"]
    time_col, gyro_cols, acc_cols, special = detect_columns(cols)
    assert time_col == "Log Time"
    assert "Yaw Rate" in gyro_cols
    assert special["lat_acc"] == "LatAcc"
    assert special["long_acc"] == "LongAcc"
