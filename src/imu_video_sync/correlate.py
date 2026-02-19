from __future__ import annotations

from typing import Tuple

import numpy as np


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size + b.size - 1
    if n <= 0:
        return np.array([], dtype=float)
    n_fft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, n_fft)
    fb = np.fft.rfft(b, n_fft)
    out = np.fft.irfft(fa * fb, n_fft)
    return out[:n]


def estimate_lag(
    signal_a: np.ndarray, signal_b: np.ndarray, fs: float, max_lag_s: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # Full cross-correlation via FFT (signal_b is reversed for correlation).
    corr = _fft_convolve(signal_a, signal_b[::-1])
    # Lags are indexed in sample units: negative means b is ahead of a.
    lags = np.arange(-signal_b.size + 1, signal_a.size)

    # Restrict search to the requested max lag window for speed and stability.
    max_lag_samples = int(round(max_lag_s * fs))
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr = corr[mask]
    lags = lags[mask]

    # Peak correlation indicates the best alignment.
    max_val = float(np.max(corr))
    eps = max(1e-12, abs(max_val) * 1e-10)
    candidates = np.where(corr >= max_val - eps)[0]
    idx = int(candidates[0]) if candidates.size else int(np.argmax(corr))
    lag_samples = int(lags[idx])

    # Sign convention: positive lag means GoPro occurs later than AiM.
    lag_seconds = -lag_samples / fs
    # Normalize peak by length to keep comparable scales across windows.
    peak = float(corr[idx]) / float(signal_a.size)

    return lag_seconds, peak, corr / float(signal_a.size), lags


def peak_to_sidelobe(
    corr: np.ndarray, lags: np.ndarray, fs: float, exclude_s: float = 0.5
) -> float:
    if corr.size == 0:
        return float("nan")
    idx = int(np.argmax(corr))
    # Ignore the main lobe neighborhood to find the next-best local peak.
    exclude_n = int(round(exclude_s * fs))
    mask = np.ones_like(corr, dtype=bool)
    mask[np.abs(lags - lags[idx]) <= exclude_n] = False
    if not mask.any():
        return float("inf")
    second_peak = float(np.max(corr[mask]))
    if second_peak <= 0:
        return float("inf")
    return float(corr[idx]) / second_peak


def lag_stability(
    signal_a: np.ndarray, signal_b: np.ndarray, fs: float, max_lag_s: float, segments: int = 4
) -> Tuple[float, float]:
    if segments < 2:
        return float("nan"), float("nan")
    # Split into equal-length segments and estimate lag per segment.
    n = min(signal_a.size, signal_b.size)
    lags = []
    max_lag_samples = int(round(max_lag_s * fs))
    # Need enough samples to cover lag range with a bit of margin.
    min_len_needed = max_lag_samples * 2 + 5

    max_segments = n // min_len_needed
    if max_segments < 2:
        return float("nan"), float("nan")

    segments = min(segments, max_segments)
    segment_len = n // segments
    if segment_len < min_len_needed:
        return float("nan"), float("nan")

    for idx in range(segments):
        start = idx * segment_len
        end = start + segment_len
        seg_a = signal_a[start:end]
        seg_b = signal_b[start:end]
        if seg_a.size < min_len_needed or seg_b.size < min_len_needed:
            continue
        lag, _, _, _ = estimate_lag(seg_a, seg_b, fs, max_lag_s)
        lags.append(lag)

    if len(lags) < 2:
        return float("nan"), float("nan")

    return float(np.mean(lags)), float(np.std(lags))
