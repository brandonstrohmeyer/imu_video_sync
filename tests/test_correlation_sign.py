import numpy as np

from imu_video_sync.correlate import estimate_lag


def test_correlation_sign_convention():
    # Build a delayed copy and check that lag sign matches the convention.
    fs = 50.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 1.0 * t)

    lag_s = 1.2
    lag_n = int(round(lag_s * fs))
    delayed = np.concatenate([np.zeros(lag_n), signal[:-lag_n]])

    lag_seconds, _, _, _ = estimate_lag(signal, delayed, fs, max_lag_s=2.0)
    assert abs(lag_seconds - lag_s) < 1 / fs
