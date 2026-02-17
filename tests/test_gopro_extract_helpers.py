import numpy as np
import pytest

from imu_video_sync.sources.gopro_gpmf import (
    _Block,
    _apply_scale,
    _build_time_series,
    _choose_scale,
    _ensure_nx3,
)


def test_ensure_nx3_shapes():
    arr = np.array([1.0, 2.0, 3.0])
    out = _ensure_nx3(arr)
    assert out.shape == (3, 1)

    arr2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out2 = _ensure_nx3(arr2)
    assert out2.shape == (2, 3)


def test_apply_scale_scalar_and_vector():
    values = np.array([[2.0, 4.0, 6.0]])
    out = _apply_scale(values, 2.0)
    assert np.allclose(out, np.array([[1.0, 2.0, 3.0]]))

    out_vec = _apply_scale(values, np.array([2.0, 4.0, 6.0]))
    assert np.allclose(out_vec, np.array([[1.0, 1.0, 1.0]]))


def test_choose_scale_matches_expected_rate():
    step_raw = 2500.0
    scale = _choose_scale(step_raw, expected_hz=400.0)
    assert scale == 1e6


def test_build_time_series_with_timestamps():
    blocks = [
        _Block(stmp=0.0, data=np.ones((2, 3))),
        _Block(stmp=1_000_000.0, data=np.ones((2, 3)) * 2.0),
    ]
    time_s, values = _build_time_series(blocks, expected_hz=400.0)
    assert time_s[0] == pytest.approx(0.0)
    assert time_s[-1] == pytest.approx(1.5)
    assert values.shape == (4, 3)


def test_build_time_series_without_timestamps():
    blocks = [
        _Block(stmp=None, data=np.ones((2, 3))),
        _Block(stmp=None, data=np.ones((2, 3))),
    ]
    time_s, values = _build_time_series(blocks, expected_hz=4.0)
    assert time_s[0] == pytest.approx(0.0)
    assert len(time_s) == 4
    assert values.shape == (4, 3)
