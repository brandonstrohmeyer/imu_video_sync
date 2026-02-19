import json
from pathlib import Path

import numpy as np
import pytest

from imu_video_sync.cli import (
    _bundle_duration,
    _bundle_rate,
    _compute_metrics,
    _detect_video_fps,
    _confidence_score,
    _safe_duration,
    _select_window_size,
)
from imu_video_sync.core.models import ImuBundle, SourceMeta, TimeSeries
from imu_video_sync.core.signals import SIGNAL_PRIORITY, available_signals
from imu_video_sync.preprocess import infer_sample_rate
from imu_video_sync.sources import resolve_source


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def _load_manifest() -> list[dict]:
    manifest_path = _fixtures_dir() / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise AssertionError("Manifest must be a list of entries.")
    return data


def _axis_names(values: np.ndarray) -> list[str]:
    if values is None:
        return []
    values = np.asarray(values)
    if values.ndim == 1:
        return ["x"]
    n = values.shape[1]
    if n == 3:
        return ["x", "y", "z"]
    if n == 2:
        return ["x", "y"]
    if n == 1:
        return ["x"]
    return [f"a{i}" for i in range(n)]


def _load_npz_video(npz_path: Path) -> ImuBundle:
    data = np.load(npz_path)

    gyro_time = data.get("gyro_time_s")
    accel_time = data.get("accel_time_s")
    gyro = data.get("gyro")
    accel = data.get("accel")

    gyro_series = None
    if gyro is not None and gyro_time is not None:
        gyro_series = TimeSeries(
            time_s=gyro_time,
            values=gyro,
            axes=_axis_names(gyro),
            name="gyro",
        )
    accel_series = None
    if accel is not None and accel_time is not None:
        accel_series = TimeSeries(
            time_s=accel_time,
            values=accel,
            axes=_axis_names(accel),
            name="accel",
        )

    return ImuBundle(
        gyro=gyro_series,
        accel=accel_series,
        channels={},
        meta=SourceMeta(name="gopro_npz", kind="video", path=npz_path),
    )


def _load_video(path: Path) -> tuple[object | None, ImuBundle]:
    if path.suffix.lower() == ".npz":
        return None, _load_npz_video(path)
    source = resolve_source("video", path, forced=None)
    return source, source.load(path)


def _load_log(path: Path):
    source = resolve_source("log", path, forced=None)
    return source, source.load(path)


def _is_video_fixture(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".mxf"}


def _default_params(log, video, signals: list[str]) -> dict:
    fs = 50.0
    window_s = 360.0
    window_step_s = 20.0
    max_lag_s = 600.0
    lowpass_hz = 8.0
    highpass_hz = 0.2

    log_rate = infer_sample_rate(np.asarray(log.time_s, dtype=float))
    video_rate = _bundle_rate(video)
    rates = [r for r in (log_rate, video_rate) if np.isfinite(r) and r > 0]
    if len(rates) == 2:
        fs = min(50.0, max(20.0, float(np.sqrt(rates[0] * rates[1]))))
    elif rates:
        fs = min(50.0, max(20.0, rates[0]))

    duration_s = min(
        _safe_duration(np.asarray(log.time_s, dtype=float)),
        _bundle_duration(video),
    )
    if duration_s > 0:
        max_lag_s = min(600.0, max(30.0, 0.5 * duration_s))

    selected_window, _ = _select_window_size(
        log=log,
        video=video,
        signals=signals,
        fs=fs,
        lowpass_hz=lowpass_hz,
        highpass_hz=highpass_hz,
        max_lag_s=max_lag_s,
        window_step_s=window_step_s,
        auto_window=True,
        window_step_is_default=True,
    )
    if selected_window > 0:
        window_s = selected_window

    lowpass_hz = min(8.0, 0.45 * fs)
    target_cycles = 3.0
    highpass_hz = target_cycles / max(10.0, window_s)
    highpass_hz = max(0.1, min(0.4, highpass_hz))

    return {
        "fs": fs,
        "window_s": window_s,
        "window_step_s": window_step_s,
        "max_lag_s": max_lag_s,
        "lowpass_hz": lowpass_hz,
        "highpass_hz": highpass_hz,
    }


def _best_metrics(log, video, params: dict) -> dict:
    available = sorted(
        available_signals(log.imu) & available_signals(video),
        key=lambda s: SIGNAL_PRIORITY.index(s) if s in SIGNAL_PRIORITY else 99,
    )
    if not available:
        raise AssertionError("No compatible signals found between log and video.")

    best = None
    for sig in available:
        metrics = _compute_metrics(
            log=log,
            video=video,
            signal=sig,
            fs=params["fs"],
            window_s=params["window_s"],
            lowpass_hz=params["lowpass_hz"],
            highpass_hz=params["highpass_hz"],
            max_lag_s=params["max_lag_s"],
            start_override=None,
            auto_window=True,
            window_step_s=params["window_step_s"],
            window_is_default=True,
            window_step_is_default=True,
            emit_warnings=False,
        )
        if best is None or metrics["score"] > best["score"]:
            best = metrics
    return best


@pytest.mark.parametrize(
    "entry",
    _load_manifest(),
    ids=lambda entry: entry.get("id", "fixture"),
)
def test_manifest_integration(entry: dict):
    fixtures_dir = _fixtures_dir()
    video_path = fixtures_dir / entry["video"]
    log_path = fixtures_dir / entry["log"]

    if not video_path.exists():
        pytest.skip(f"Missing video fixture: {video_path}")
    if not log_path.exists():
        pytest.skip(f"Missing log fixture: {log_path}")

    log_source, log = _load_log(log_path)
    video_source, video = _load_video(video_path)

    sniff_log = log_source.sniff(log_path)
    assert sniff_log > 0.0

    if video_source is not None:
        sniff_video = video_source.sniff(video_path)
        assert sniff_video > 0.0

    available = sorted(
        available_signals(log.imu) & available_signals(video),
        key=lambda s: SIGNAL_PRIORITY.index(s) if s in SIGNAL_PRIORITY else 99,
    )
    if not available:
        raise AssertionError("No compatible signals found between log and video.")

    params = _default_params(log, video, signals=available)
    if "override_window_s" in entry:
        params["window_s"] = float(entry["override_window_s"])
    if "override_window_step" in entry:
        params["window_step_s"] = float(entry["override_window_step"])

    assert log.time_s.size > 0
    assert log.time_s[0] == pytest.approx(0.0)
    assert np.all(np.diff(log.time_s) >= 0)

    if _is_video_fixture(video_path):
        fps = _detect_video_fps(video_path)
        assert fps is not None and fps > 0

        has_gyro = video.gyro is not None and video.gyro.values.size > 0
        has_accel = video.accel is not None and video.accel.values.size > 0
        assert has_gyro or has_accel
        if video.gyro is not None:
            assert video.gyro.time_s[0] == pytest.approx(0.0)
            assert np.all(np.diff(video.gyro.time_s) >= 0)
        if video.accel is not None:
            assert video.accel.time_s[0] == pytest.approx(0.0)
            assert np.all(np.diff(video.accel.time_s) >= 0)

    metrics = _best_metrics(log, video, params=params)

    lag = metrics["lag_seconds"]
    assert np.isfinite(lag)

    expected = float(entry["expected_lag_s"])
    tolerance = float(entry.get("tolerance_s", 0.1))
    assert abs(lag - expected) <= tolerance

    max_abs = entry.get("max_abs_lag_s")
    if max_abs is not None:
        assert abs(lag) <= float(max_abs)

    min_conf = entry.get("min_confidence")
    if min_conf is not None:
        conf = _confidence_score(metrics["peak"], metrics["psr"], metrics["stability"])
        assert conf >= float(min_conf)
