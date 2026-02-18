import numpy as np
import pytest

from imu_video_sync.preprocess import (
    filter_signal,
    infer_sample_rate,
    normalize_signal,
    resample_uniform,
    select_active_window,
    select_best_window_multi,
    trim_window,
)


def test_infer_sample_rate_basic():
    t = np.arange(0.0, 1.0, 0.1)
    assert infer_sample_rate(t) == pytest.approx(10.0)


def test_infer_sample_rate_nan_when_no_progress():
    t = np.array([1.0, 1.0, 1.0])
    assert np.isnan(infer_sample_rate(t))


def test_resample_uniform_basic():
    t = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    t_u, y_u = resample_uniform(t, y, fs=2.0)
    assert t_u[0] == pytest.approx(0.0)
    assert t_u[-1] == pytest.approx(0.5)
    assert len(t_u) == 2
    assert np.allclose(y_u, np.array([0.0, 1.0]))


def test_resample_uniform_errors():
    with pytest.raises(ValueError):
        resample_uniform(np.array([0.0]), np.array([1.0]), fs=10.0)
    with pytest.raises(ValueError):
        resample_uniform(np.array([1.0, 1.0]), np.array([1.0, 2.0]), fs=10.0)


def test_select_active_window_picks_activity():
    fs = 10.0
    t = np.arange(0.0, 6.0, 1 / fs)
    y = np.zeros_like(t)
    y[t >= 3.0] = np.sin(2 * np.pi * 1.0 * t[t >= 3.0])
    start = select_active_window(t, y, window_s=2.0, fs=fs, step_s=1.0)
    assert 2.0 <= start <= 3.0


def test_select_best_window_multi_prefers_overlap_activity():
    fs = 10.0
    t = np.arange(0.0, 20.0, 1 / fs)
    y = np.zeros_like(t)
    active = (t >= 5.0) & (t < 10.0)
    y[active] = np.sin(2 * np.pi * 1.0 * t[active])
    start = select_best_window_multi(t, y, t, y, window_s=3.0, fs=fs, step_s=1.0)
    assert 5.0 <= start <= 7.0


def test_trim_window_length_and_bounds():
    fs = 10.0
    t = np.arange(0.0, 10.0, 1 / fs)
    y = np.sin(t)
    t_w, y_w = trim_window(t, y, start_s=2.0, window_s=3.0, fs=fs)
    assert len(t_w) == int(round(3.0 * fs))
    assert len(y_w) == int(round(3.0 * fs))
    with pytest.raises(ValueError):
        trim_window(t, y, start_s=-1.0, window_s=3.0, fs=fs)


def test_filter_signal_validation_and_passthrough():
    values = np.random.randn(1000)
    with pytest.raises(ValueError):
        filter_signal(values, fs=100.0, lowpass_hz=4.0, highpass_hz=5.0)
    out = filter_signal(values, fs=100.0, lowpass_hz=0.0, highpass_hz=0.0)
    assert np.allclose(out, values)


def test_normalize_signal_properties():
    values = np.array([1.0, 2.0, 3.0])
    norm = normalize_signal(values)
    assert np.mean(norm) == pytest.approx(0.0, abs=1e-8)
    assert np.std(norm) == pytest.approx(1.0, abs=1e-8)
    with pytest.raises(ValueError):
        normalize_signal(np.ones(10))
