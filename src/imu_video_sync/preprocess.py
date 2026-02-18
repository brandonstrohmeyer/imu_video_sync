from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt


def infer_sample_rate(time_s: np.ndarray) -> float:
    # Median delta time gives a robust sample rate estimate.
    diffs = np.diff(time_s)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float("nan")
    return 1.0 / float(np.median(diffs))


def resample_uniform(time_s: np.ndarray, values: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if time_s.size < 2:
        raise ValueError("Not enough samples to resample.")
    t0 = float(time_s[0])
    t1 = float(time_s[-1])
    if t1 <= t0:
        raise ValueError("Non-increasing time series.")
    # Build a uniform time grid and linearly interpolate values.
    dt = 1.0 / fs
    t_uniform = np.arange(t0, t1, dt)
    y_uniform = np.interp(t_uniform, time_s, values)
    return t_uniform, y_uniform


def select_active_window(
    time_s: np.ndarray,
    values: np.ndarray,
    window_s: float,
    fs: float,
    threshold_ratio: float = 0.2,
    step_s: float = 1.0,
) -> float:
    win_n = int(round(window_s * fs))
    if win_n <= 1 or values.size < win_n:
        return float(time_s[0])
    step_n = max(1, int(round(step_s * fs)))
    vars_list = []
    start_times = []
    # Slide a window and track variance as a proxy for "activity".
    for idx in range(0, values.size - win_n + 1, step_n):
        window = values[idx : idx + win_n]
        vars_list.append(float(np.var(window)))
        start_times.append(float(time_s[idx]))
    max_var = max(vars_list) if vars_list else 0.0
    threshold = max(1e-8, threshold_ratio * max_var)
    for var, start in zip(vars_list, start_times):
        if var >= threshold:
            return float(start)
    return float(time_s[0])


def select_best_window_multi(
    aim_t: np.ndarray,
    aim_y: np.ndarray,
    gopro_t: np.ndarray,
    gopro_y: np.ndarray,
    window_s: float,
    fs: float,
    threshold_ratio: float = 0.2,
    step_s: float = 1.0,
) -> float:
    win_n = int(round(window_s * fs))
    if win_n <= 1 or aim_y.size < win_n or gopro_y.size < win_n:
        return float(max(aim_t[0], gopro_t[0]))

    start_min = float(max(aim_t[0], gopro_t[0]))
    start_max = float(min(aim_t[-1], gopro_t[-1]) - window_s)
    if start_max <= start_min:
        return start_min

    step_n = max(1, int(round(step_s * fs)))
    idx_start = int(np.searchsorted(aim_t, start_min, side="left"))
    idx_end = int(np.searchsorted(aim_t, start_max, side="left"))

    var_a_list = []
    var_b_list = []
    starts = []

    # Score candidate windows where both signals show activity.
    for i in range(idx_start, idx_end + 1, step_n):
        if i + win_n > aim_y.size:
            break
        start_s = float(aim_t[i])
        j = int(round((start_s - gopro_t[0]) * fs))
        if j < 0 or j + win_n > gopro_y.size:
            continue
        var_a = float(np.var(aim_y[i : i + win_n]))
        var_b = float(np.var(gopro_y[j : j + win_n]))
        var_a_list.append(var_a)
        var_b_list.append(var_b)
        starts.append(start_s)

    if not starts:
        return start_min

    max_a = max(var_a_list)
    max_b = max(var_b_list)
    min_a = max(1e-9, threshold_ratio * max_a)
    min_b = max(1e-9, threshold_ratio * max_b)
    med_a = float(np.median(var_a_list)) if var_a_list else 1.0
    med_b = float(np.median(var_b_list)) if var_b_list else 1.0
    med_a = med_a if med_a > 0 else 1.0
    med_b = med_b if med_b > 0 else 1.0

    best_score = float("-inf")
    best_start = starts[0]

    # Prefer windows that are above variance thresholds on both sources.
    for start_s, var_a, var_b in zip(starts, var_a_list, var_b_list):
        if var_a < min_a or var_b < min_b:
            continue
        score = (var_a / med_a) + (var_b / med_b)
        if score > best_score:
            best_score = score
            best_start = start_s

    if best_score == float("-inf"):
        # Fallback: pick the best score even if below thresholds.
        for start_s, var_a, var_b in zip(starts, var_a_list, var_b_list):
            score = (var_a / med_a) + (var_b / med_b)
            if score > best_score:
                best_score = score
                best_start = start_s

    return float(best_start)


def trim_window(
    time_s: np.ndarray, values: np.ndarray, start_s: float, window_s: float, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    end_s = start_s + window_s
    if start_s < time_s[0] or end_s > time_s[-1]:
        raise ValueError("Requested window is out of bounds for the signal.")
    idx_start = int(np.searchsorted(time_s, start_s, side="left"))
    idx_end = int(np.searchsorted(time_s, end_s, side="left"))
    # Ensure we have enough samples even if the end aligns between points.
    win_n = int(round(window_s * fs))
    if idx_end - idx_start < win_n:
        idx_end = idx_start + win_n
    if idx_end > values.size:
        raise ValueError("Requested window is too long for the signal.")
    return time_s[idx_start:idx_end], values[idx_start:idx_end]


def filter_signal(
    values: np.ndarray, fs: float, lowpass_hz: float, highpass_hz: float = 0.0
) -> np.ndarray:
    # Use a 4th-order Butterworth filter in low/high/band-pass mode.
    if lowpass_hz is not None and lowpass_hz > 0.0 and lowpass_hz < fs / 2.0:
        if highpass_hz is not None and highpass_hz > 0.0:
            if highpass_hz >= lowpass_hz:
                raise ValueError("High-pass cutoff must be below low-pass cutoff.")
            wn = [highpass_hz / (fs / 2.0), lowpass_hz / (fs / 2.0)]
            b, a = butter(4, wn, btype="band")
        else:
            b, a = butter(4, lowpass_hz / (fs / 2.0), btype="low")
        return filtfilt(b, a, values)

    if highpass_hz is not None and highpass_hz > 0.0 and highpass_hz < fs / 2.0:
        b, a = butter(4, highpass_hz / (fs / 2.0), btype="high")
        return filtfilt(b, a, values)

    return values


def normalize_signal(values: np.ndarray) -> np.ndarray:
    # Z-score normalization makes signals comparable across sensors.
    values = values - float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        raise ValueError("Signal variance too low after preprocessing.")
    return values / std


def preprocess_signal(
    values: np.ndarray, fs: float, lowpass_hz: float, highpass_hz: float = 0.0
) -> np.ndarray:
    filtered = filter_signal(values, fs, lowpass_hz, highpass_hz)
    return normalize_signal(filtered)
