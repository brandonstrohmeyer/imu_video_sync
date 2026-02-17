import numpy as np
import pytest

from imu_video_sync.correlate import estimate_lag, lag_stability, peak_to_sidelobe


def test_estimate_lag_respects_max_lag():
    fs = 10.0
    t = np.arange(0.0, 50.0, 1 / fs)
    signal = np.sin(2 * np.pi * 0.5 * t)
    lag_s = 5.0
    lag_n = int(round(lag_s * fs))
    delayed = np.concatenate([np.zeros(lag_n), signal[:-lag_n]])
    lag_seconds, _, _, _ = estimate_lag(signal, delayed, fs, max_lag_s=1.0)
    assert abs(lag_seconds) <= 1.01


def test_peak_to_sidelobe_infinite_when_no_sidelobe():
    corr = np.array([1.0])
    lags = np.array([0])
    psr = peak_to_sidelobe(corr, lags, fs=10.0, exclude_s=1.0)
    assert np.isinf(psr)


def test_peak_to_sidelobe_infinite_when_second_peak_nonpositive():
    corr = np.array([1.0, 0.0, 0.0])
    lags = np.array([-1, 0, 1])
    psr = peak_to_sidelobe(corr, lags, fs=10.0, exclude_s=0.0)
    assert np.isinf(psr)


def test_lag_stability_basic():
    fs = 50.0
    t = np.arange(0.0, 60.0, 1 / fs)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 1.5 * t)
    lag_s = 0.8
    lag_n = int(round(lag_s * fs))
    delayed = np.concatenate([np.zeros(lag_n), signal[:-lag_n]])
    mean_lag, std_lag = lag_stability(signal, delayed, fs, max_lag_s=2.0, segments=4)
    assert mean_lag == pytest.approx(lag_s, abs=1 / fs)
    assert std_lag < 0.05
