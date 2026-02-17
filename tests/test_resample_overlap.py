import numpy as np

from imu_video_sync.preprocess import resample_uniform, trim_window


def test_resample_and_trim_window():
    # Resample to uniform grid and ensure trimming returns the right window.
    t = np.linspace(0, 10, 51)
    y = np.sin(t)

    t_u, y_u = resample_uniform(t, y, fs=10.0)
    t_w, y_w = trim_window(t_u, y_u, start_s=2.0, window_s=5.0, fs=10.0)

    assert t_w[0] >= 2.0
    assert t_w[-1] <= 7.0
    assert y_w.size == 50
