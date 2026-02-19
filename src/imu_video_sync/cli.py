from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import __version__
from .core.models import ImuBundle, LogData
from .core.signals import SIGNAL_PRIORITY, available_signals, choose_signal, derive_signal
from .correlate import estimate_lag, lag_stability, peak_to_sidelobe
from .io_out import write_imu_csv, write_shifted_log
from .preprocess import (
    filter_signal,
    infer_sample_rate,
    normalize_signal,
    resample_uniform,
    select_active_window,
    trim_window,
)
from .sources import resolve_source


def _parse_cols(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [col.strip() for col in value.split(",") if col.strip()]


def _parse_kv_args(values: List[str]) -> dict:
    opts: dict = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid option '{item}'. Expected key=value.")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Invalid option '{item}'. Key is required.")
        opts[key] = val
    return opts


def _autodetect_file(ext: str) -> Optional[Path]:
    # Find exactly one file with the given extension in the current directory.
    matches = [p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() == ext]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_paths(video_arg: Optional[str], log_arg: Optional[str]) -> tuple[Path, Path]:
    # Prefer explicit paths, otherwise auto-detect by extension.
    video_path = Path(video_arg) if video_arg else _autodetect_file(".mp4")
    log_path = Path(log_arg) if log_arg else _autodetect_file(".csv")

    if video_path is None or log_path is None:
        raise ValueError(
            "Auto-detect failed. Provide --video and --log when there is not exactly one .mp4 and one .csv in the directory."
        )
    if not video_path.exists():
        raise ValueError(f"Video not found: {video_path}")
    if not log_path.exists():
        raise ValueError(f"Log not found: {log_path}")
    return video_path, log_path


def _describe_derived(derived) -> str:
    base = derived.derived_from.replace("_", " ")
    if derived.axes:
        axes = ", ".join(derived.axes)
        return f"{base} ({axes})"
    return base


def _format_rate(rate: float) -> str:
    if np.isnan(rate):
        return "unknown"
    return f"{rate:.2f} Hz"


def _safe_duration(time_s: np.ndarray) -> float:
    if time_s.size < 2:
        return 0.0
    finite = np.isfinite(time_s)
    if not finite.any():
        return 0.0
    return float(np.nanmax(time_s[finite]) - np.nanmin(time_s[finite]))


def _bundle_duration(imu: ImuBundle) -> float:
    candidates = []
    if imu.gyro is not None:
        candidates.append(imu.gyro.time_s)
    if imu.accel is not None:
        candidates.append(imu.accel.time_s)
    if imu.channels:
        for series in imu.channels.values():
            candidates.append(series.time_s)
    for time_s in candidates:
        duration = _safe_duration(np.asarray(time_s, dtype=float))
        if duration > 0:
            return duration
    return 0.0


def _bundle_rate(imu: ImuBundle) -> float:
    candidates = []
    if imu.gyro is not None:
        candidates.append(imu.gyro.time_s)
    if imu.accel is not None:
        candidates.append(imu.accel.time_s)
    if imu.channels:
        for series in imu.channels.values():
            candidates.append(series.time_s)
    for time_s in candidates:
        rate = infer_sample_rate(np.asarray(time_s, dtype=float))
        if np.isfinite(rate) and rate > 0:
            return float(rate)
    return float("nan")


def _candidate_window_sizes(duration_s: float) -> list[float]:
    base = [30.0, 45.0, 60.0, 75.0, 90.0, 120.0]
    extra = [0.5 * duration_s, 0.6 * duration_s, 0.7 * duration_s]
    candidates = {float(round(val, 1)) for val in base + extra}
    filtered = [c for c in candidates if c > 5.0 and c < 0.9 * duration_s]
    return sorted(filtered)


def _select_window_size(
    log: LogData,
    video: ImuBundle,
    signals: List[str],
    fs: float,
    lowpass_hz: float,
    highpass_hz: float,
    max_lag_s: float,
    window_step_s: float,
    auto_window: bool,
    window_step_is_default: bool,
) -> tuple[float, list[float]]:
    log_duration = _safe_duration(np.asarray(log.time_s, dtype=float))
    video_duration = _bundle_duration(video)
    duration_s = min(log_duration, video_duration)

    if duration_s <= 0:
        return 0.0, []

    candidates = _candidate_window_sizes(duration_s)
    if not candidates:
        fallback = max(5.0, 0.6 * duration_s)
        return min(duration_s, fallback), [min(duration_s, fallback)]

    signals_to_eval = signals[:2] if signals else []
    best_window = candidates[0]
    best_score = float("-inf")
    best_conf = float("-inf")
    window_scores: dict[float, float] = {}
    window_confs: dict[float, float] = {}

    for window_s in candidates:
        for sig in signals_to_eval:
            try:
                metrics = _compute_metrics(
                    log=log,
                    video=video,
                    signal=sig,
                    fs=fs,
                    window_s=window_s,
                    lowpass_hz=lowpass_hz,
                    highpass_hz=highpass_hz,
                    max_lag_s=max_lag_s,
                    start_override=None,
                    auto_window=auto_window,
                    window_step_s=window_step_s,
                    window_is_default=True,
                    window_step_is_default=window_step_is_default,
                    emit_warnings=False,
                )
            except Exception:
                continue

            score = metrics["score"]
            if not np.isfinite(score):
                score = -1.0
            if not np.isfinite(metrics["stability"]):
                score -= 0.3
            if metrics.get("window_count", 1) < 3:
                score -= 0.5

            conf_score = _confidence_score(
                metrics["peak"], metrics["psr"], metrics["stability"]
            )

            if score > window_scores.get(window_s, float("-inf")):
                window_scores[window_s] = score
                window_confs[window_s] = conf_score

            if (score > best_score + 1e-6) or (
                abs(score - best_score) <= 1e-6 and conf_score > best_conf
            ):
                best_score = score
                best_conf = conf_score
                best_window = window_s
    if duration_s >= 240.0 and np.isfinite(best_score) and best_score > 0:
        cutoff_ratio = 0.85
        cutoff = cutoff_ratio * best_score
        near_best = [w for w, s in window_scores.items() if s >= cutoff]
        if near_best:
            best_window = max(near_best)

    return best_window, candidates


def _drop_nan(time_s: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Remove NaNs so filtering and correlation do not explode.
    mask = np.isfinite(time_s) & np.isfinite(values)
    if mask.sum() < 2:
        raise ValueError("Too few finite samples after removing NaNs.")
    return time_s[mask], values[mask]


def _clamp_start(start_s: float, log_t: np.ndarray, video_t: np.ndarray, window_s: float) -> float:
    # Clamp start time to the shared overlap region.
    if start_s < 0:
        raise ValueError("Start time must be >= 0 seconds.")
    latest_start = min(log_t[-1], video_t[-1]) - window_s
    if latest_start < 0:
        raise ValueError("Analysis window is longer than available data.")
    if start_s > latest_start:
        print(
            f"Warning: Selected start {start_s:.1f}s exceeds available range. Clamping to {latest_start:.1f}s."
        )
        start_s = max(0.0, latest_start)
    return start_s


def _score_metrics(peak: float, psr: float, stability: float) -> float:
    # Higher peak and PSR, lower stability variance -> better score.
    if not np.isfinite(peak):
        return -1.0
    psr_val = psr if np.isfinite(psr) and psr > 0 else 1.0
    stab_val = stability if np.isfinite(stability) else 0.5
    return float(peak) * float(psr_val) / (1.0 + float(stab_val))


def _confidence_score(peak: float, psr: float, stability: float) -> float:
    if not np.isfinite(peak):
        return 0.0
    peak_score = float(np.clip(peak, 0.0, 1.0))
    psr_score = float(np.clip(psr / 3.0, 0.0, 1.0)) if np.isfinite(psr) else 0.0
    if np.isfinite(stability):
        stability_score = float(np.clip(1.0 - (stability / 0.3), 0.0, 1.0))
    else:
        stability_score = 0.5
    return 100.0 * (0.5 * peak_score + 0.3 * psr_score + 0.2 * stability_score)


def _confidence_rating(score: float) -> str:
    if score >= 75.0:
        return "High"
    if score >= 55.0:
        return "Medium"
    return "Low"


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    # Robust median that respects per-window weights.
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return float("nan")
    if weights.size == 0 or np.all(weights <= 0):
        return float(np.median(values))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * cumulative[-1]
    return float(values[np.searchsorted(cumulative, cutoff)])


def _robust_std(values: np.ndarray) -> float:
    # Median absolute deviation (scaled) for outlier-resistant spread.
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def _estimate_drift_info(
    start_s: np.ndarray,
    lag_s: np.ndarray,
    min_windows: int,
    min_span_s: float,
    min_r2: float = 0.5,
) -> Optional[dict]:
    # Robust drift estimate (lag slope vs time) with reliability gating.
    start_s = np.asarray(start_s, dtype=float)
    lag_s = np.asarray(lag_s, dtype=float)

    n = start_s.size
    if n < min_windows:
        return None

    span = float(np.nanmax(start_s) - np.nanmin(start_s))
    if not np.isfinite(span) or span < min_span_s:
        return None

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = start_s[j] - start_s[i]
            if dt == 0:
                continue
            slopes.append((lag_s[j] - lag_s[i]) / dt)
    if not slopes:
        return None

    slope = float(np.median(slopes))
    intercept = float(np.median(lag_s - slope * start_s))
    pred = slope * start_s + intercept
    resid = lag_s - pred
    resid_mad = 1.4826 * float(np.median(np.abs(resid)))

    sst = float(np.sum((lag_s - float(np.mean(lag_s))) ** 2))
    sse = float(np.sum(resid ** 2))
    r2 = 1.0 - (sse / sst) if sst > 1e-9 else 0.0

    lag_spread = _robust_std(lag_s)
    resid_thresh = max(0.05, 0.5 * lag_spread)
    reliable = (
        np.isfinite(slope)
        and np.isfinite(r2)
        and r2 >= min_r2
        and np.isfinite(resid_mad)
        and resid_mad <= resid_thresh
    )

    return {
        "slope": slope,
        "r2": r2,
        "resid_mad": resid_mad,
        "span_s": span,
        "n": int(n),
        "reliable": bool(reliable),
    }


def _compute_window_candidates(
    log_t_full: np.ndarray,
    log_filt: np.ndarray,
    video_norm: np.ndarray,
    fs: float,
    window_s: float,
    step_s: float,
    max_lag_s: float,
    start_idx_min: int,
    video_duration: float,
    var_threshold_ratio: float = 0.2,
) -> tuple[list[dict], Optional[dict]]:
    win_n = int(round(window_s * fs))
    if win_n <= 1 or log_filt.size < win_n:
        return [], None

    step_n = max(1, int(round(step_s * fs)))
    var_list = []
    starts = []
    start_idx_min = max(0, start_idx_min)
    # Slide a window across log data and score by variance (activity proxy).
    for idx in range(start_idx_min, log_filt.size - win_n + 1, step_n):
        window = log_filt[idx : idx + win_n]
        var_list.append(float(np.var(window)))
        starts.append(float(log_t_full[idx]))

    if not starts:
        return [], None

    max_var = max(var_list) if var_list else 0.0
    var_arr = np.array(var_list, dtype=float) if var_list else np.array([], dtype=float)
    perc_threshold = float(np.percentile(var_arr, 60)) if var_arr.size else 0.0
    threshold = max(1e-9, var_threshold_ratio * max_var, perc_threshold)

    candidates: list[dict] = []
    best: Optional[dict] = None

    # For each active window, estimate a lag against the full video signal.
    for idx, start_s in enumerate(starts):
        var_val = var_list[idx]
        if var_val < threshold:
            continue
        start_idx = int(np.searchsorted(log_t_full, start_s, side="left"))
        log_seg = log_filt[start_idx : start_idx + win_n]
        if log_seg.size < win_n:
            continue
        try:
            log_norm = normalize_signal(log_seg)
        except Exception:
            continue

        lag_local, peak, corr, lags = estimate_lag(log_norm, video_norm, fs, max_lag_s)
        video_start = lag_local
        if video_start < 0:
            continue
        if video_start + window_s > video_duration:
            continue

        psr = peak_to_sidelobe(corr, lags, fs)
        score = _score_metrics(peak, psr, float("nan"))
        lag_global = start_s - lag_local

        cand = {
            "start_s": start_s,
            "lag_local": lag_local,
            "lag_global": lag_global,
            "peak": peak,
            "psr": psr,
            "score": score,
        }
        candidates.append(cand)

        if best is None or score > best["score"]:
            best = {
                **cand,
                "corr": corr,
                "lags": lags,
                "log_seg": log_norm,
            }

    return candidates, best


def _compute_metrics(
    log: LogData,
    video: ImuBundle,
    signal: str,
    fs: float,
    window_s: float,
    lowpass_hz: float,
    highpass_hz: float,
    max_lag_s: float,
    start_override: Optional[float],
    auto_window: bool,
    window_step_s: float,
    window_is_default: bool,
    window_step_is_default: bool,
    emit_warnings: bool,
) -> dict:
    # Build comparable signals for this requested signal type.
    log_sig = derive_signal(log.imu, signal)
    video_sig = derive_signal(video, signal)

    log_time, log_signal = _drop_nan(log_sig.time_s, log_sig.values)
    video_time, video_signal = _drop_nan(video_sig.time_s, video_sig.values)

    # Estimate source sample rates for reporting only.
    log_rate = infer_sample_rate(log_time)
    video_rate = infer_sample_rate(video_time)

    # Resample both signals to a common uniform rate.
    log_t_full, log_y_full = resample_uniform(log_time, log_signal, fs)
    video_t_full, video_y_full = resample_uniform(video_time, video_signal, fs)

    # Auto-shrink window if it exceeds available data.
    log_duration = float(log_t_full[-1] - log_t_full[0])
    video_duration = float(video_t_full[-1] - video_t_full[0])
    max_window = min(log_duration, video_duration)
    if max_window <= 0:
        raise ValueError("Not enough data to compute a correlation window.")
    if window_s > max_window:
        new_window = max(1.0, max_window)
        if emit_warnings:
            print(
                f"Warning: Window {window_s:.1f}s exceeds available data. "
                f"Shrinking to {new_window:.1f}s."
            )
        window_s = new_window
    if auto_window and window_s >= 0.99 * video_duration:
        if window_is_default:
            short_window = max(30.0, 0.6 * max_window)
            if short_window < window_s:
                if emit_warnings:
                    print(
                        f"Warning: Short clip detected. Using window {short_window:.1f}s "
                        "for auto-window."
                    )
                window_s = short_window
        if window_s >= 0.99 * video_duration:
            if emit_warnings:
                print(
                    "Warning: Auto-window disabled because the window length "
                    "nearly equals the video duration."
                )
            auto_window = False

    if auto_window and window_is_default and window_step_is_default and start_override is None:
        default_step = max(4.0, window_s * 0.08)
        if window_step_s > default_step:
            if emit_warnings:
                print(
                    f"Info: Using window step {default_step:.1f}s for short clip auto-window."
                )
            window_step_s = default_step

    # Filter and normalize to emphasize comparable motion.
    log_filt = filter_signal(log_y_full, fs, lowpass_hz, highpass_hz)
    video_filt = filter_signal(video_y_full, fs, lowpass_hz, highpass_hz)
    video_norm = normalize_signal(video_filt)

    win_n = int(round(window_s * fs))
    video_duration = float(video_t_full[-1]) if video_t_full.size else 0.0

    if not auto_window:
        # Single-window mode (manual or first active window).
        if start_override is None:
            start_s = select_active_window(log_t_full, log_y_full, window_s, fs)
        else:
            start_s = float(start_override)
        start_s = _clamp_start(start_s, log_t_full, video_t_full, window_s)
        log_t, log_y = trim_window(log_t_full, log_filt, start_s, window_s, fs)
        log_norm = normalize_signal(log_y)

        lag_local, peak, corr, lags = estimate_lag(log_norm, video_norm, fs, max_lag_s)
        lag_seconds = start_s - lag_local
        psr = peak_to_sidelobe(corr, lags, fs)

        stability_std = float("nan")
        video_start = lag_local
        if video_start >= 0 and video_start + window_s <= video_duration:
            _, video_aligned = trim_window(
                video_t_full, video_filt, video_start, window_s, fs
            )
            video_aligned = normalize_signal(video_aligned)
            stability_lag_s = min(30.0, max_lag_s, 0.2 * window_s)
            stability_lag_s = max(5.0, stability_lag_s)
            _, stability_std = lag_stability(log_norm, video_aligned, fs, stability_lag_s)

        score = _score_metrics(peak, psr, stability_std)

        return {
            "signal": signal,
            "lag_seconds": lag_seconds,
            "peak": peak,
            "psr": psr,
            "stability": stability_std,
            "score": score,
            "log_rate": log_rate,
            "video_rate": video_rate,
            "corr": corr,
            "lags": lags,
            "log_t": log_t,
            "log_y": log_norm,
            "video_y": video_norm,
            "start_s": start_s,
            "window_count": 1,
            "drift": None,
        }

    start_idx_min = 0
    if start_override is not None:
        start_idx_min = int(round(float(start_override) * fs))

    # Auto-window mode: scan windows and build a consensus lag.
    candidates, best = _compute_window_candidates(
        log_t_full,
        log_filt,
        video_norm,
        fs,
        window_s,
        window_step_s,
        max_lag_s,
        start_idx_min,
        video_duration,
    )

    if not candidates or best is None:
        raise ValueError("Auto-window selection failed to find valid windows.")

    # Keep the top scoring windows for robust aggregation.
    scores = np.array([c["score"] for c in candidates], dtype=float)
    order = np.argsort(scores)[::-1]
    best_score = float(scores[order[0]]) if order.size else float("nan")
    score_cutoff = float(np.percentile(scores, 60)) if scores.size else float("nan")
    if np.isfinite(best_score):
        score_cutoff = max(score_cutoff, 0.6 * best_score)
    keep_idx = [idx for idx in order if scores[idx] >= score_cutoff] if scores.size else []
    if len(keep_idx) < 5:
        keep_n = min(len(order), max(5, int(0.4 * len(order))))
        keep_idx = list(order[:keep_n])

    kept = [candidates[i] for i in keep_idx]
    lag_values = np.array([c["lag_global"] for c in kept], dtype=float)
    weight_values = np.array([max(0.0, c["score"]) for c in kept], dtype=float)
    peak_values = np.array([c["peak"] for c in kept], dtype=float)
    psr_values = np.array([c["psr"] for c in kept], dtype=float)

    lag_seconds = _weighted_median(lag_values, weight_values)
    stability_std = _robust_std(lag_values)
    peak = float(np.median(peak_values)) if peak_values.size else float("nan")
    psr = float(np.median(psr_values)) if psr_values.size else float("nan")
    score = _score_metrics(peak, psr, stability_std)

    min_span_s = max(60.0, 0.2 * video_duration) if video_duration > 0 else 60.0
    min_windows = max(4, min(8, int(0.4 * len(kept)))) if kept else 4
    drift_info = _estimate_drift_info(
        np.array([c["start_s"] for c in kept], dtype=float),
        lag_values,
        min_windows=min_windows,
        min_span_s=min_span_s,
    )

    best_start = best["start_s"]
    start_idx = int(np.searchsorted(log_t_full, best_start, side="left"))
    log_seg = log_filt[start_idx : start_idx + win_n]
    log_seg = normalize_signal(log_seg)
    video_start = best["lag_local"]
    if video_start >= 0 and video_start + window_s <= video_duration:
        _, video_seg = trim_window(
            video_t_full, video_filt, video_start, window_s, fs
        )
        video_seg = normalize_signal(video_seg)
    else:
        video_seg = video_norm[: win_n]

    return {
        "signal": signal,
        "lag_seconds": lag_seconds,
        "peak": peak,
        "psr": psr,
        "stability": stability_std,
        "score": score,
        "log_rate": log_rate,
        "video_rate": video_rate,
        "corr": best["corr"],
        "lags": best["lags"],
        "log_t": log_t_full[start_idx : start_idx + win_n],
        "log_y": log_seg,
        "video_y": video_seg,
        "start_s": best_start,
        "window_count": len(kept),
        "drift": drift_info,
    }


def _format_hhmmss_ms(value: float) -> str:
    total_ms = int(round(abs(value) * 1000.0))
    hours = total_ms // 3600000
    rem = total_ms % 3600000
    minutes = rem // 60000
    rem = rem % 60000
    seconds = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _format_timecode(value: float, fps: Optional[float]) -> str:
    if fps is None or fps <= 0:
        return "n/a (fps unknown)"
    sign = "+" if value >= 0 else "-"
    total_s = abs(value)
    whole_s = int(total_s)
    frac_s = total_s - whole_s
    frames = int(round(frac_s * fps))
    nominal_fps = int(round(fps))
    if nominal_fps <= 0:
        return "n/a (fps unknown)"
    if frames >= nominal_fps:
        frames = 0
        whole_s += 1
    hours = whole_s // 3600
    rem = whole_s % 3600
    minutes = rem // 60
    seconds = rem % 60
    frame_width = max(2, len(str(nominal_fps - 1)))
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d};{frames:0{frame_width}d}"


def _format_lag_frames(value: float, fps: Optional[float]) -> str:
    if fps is None or fps <= 0:
        return "n/a (fps unknown)"
    frames = int(round(value * fps))
    return f"{frames:+d}"


def _offset_summary_rows(lag_seconds: float, fps: Optional[float]) -> List[Tuple[str, str]]:
    if lag_seconds > 0:
        offset_label = "Video offset within project"
    elif lag_seconds < 0:
        offset_label = "Data offset within project"
    else:
        offset_label = "Video offset within project"
    rows = [
        ("Lag (seconds)", f"{lag_seconds:+.3f}"),
        (offset_label, _format_hhmmss_ms(lag_seconds)),
    ]
    if fps is not None and fps > 0:
        rows.insert(1, ("Lag (frames)", _format_lag_frames(lag_seconds, fps)))
        rows.insert(2, ("Timecode offset", _format_timecode(lag_seconds, fps)))
    return rows


def _detect_video_fps_from_telemetry_parser(video_path: Path) -> Optional[float]:
    if not video_path.exists():
        return None
    try:
        import telemetry_parser  # type: ignore
    except Exception:
        return None
    try:
        parser = telemetry_parser.Parser(str(video_path))
    except Exception:
        return None
    fps = _extract_fps_from_frame_info(parser)
    if fps is not None and fps > 0:
        return fps

    fps = _extract_fps_from_telemetry(parser)
    if fps is not None and fps > 0:
        return fps

    return None


def _extract_fps_from_frame_info(parser: object) -> Optional[float]:
    try:
        frame_info = parser.frame_info()  # type: ignore[attr-defined]
    except Exception:
        frame_info = None

    return _coerce_fps(frame_info) or _search_fps_in_value(frame_info)


def _extract_fps_from_telemetry(parser: object) -> Optional[float]:
    try:
        data = parser.telemetry()  # type: ignore[attr-defined]
    except Exception:
        return None

    for sample in _iter_telemetry_samples(data):
        if not isinstance(sample, dict):
            continue
        for group_val in sample.values():
            if not isinstance(group_val, dict):
                continue
            fps = _find_fps_in_tag_map(group_val)
            if fps is not None:
                return fps
    return None


def _iter_telemetry_samples(data: object):
    if isinstance(data, dict):
        yield data
    elif isinstance(data, (list, tuple)):
        for item in data:
            yield item


def _find_fps_in_tag_map(tag_map: dict) -> Optional[float]:
    for key, value in tag_map.items():
        key_str = str(key).strip().lower()
        if key_str in {"framerate", "frame_rate", "frame rate", "fps"}:
            fps = _coerce_fps(value)
            if fps is not None:
                return fps
        if "frameinfo" in key_str or "frame info" in key_str:
            fps = _search_fps_in_value(value)
            if fps is not None:
                return fps
    return _search_fps_in_value(tag_map)


def _coerce_fps(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("fps", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _search_fps_in_value(value: object, depth: int = 3) -> Optional[float]:
    if depth < 0:
        return None
    if isinstance(value, dict):
        for key, val in value.items():
            key_str = str(key).strip().lower()
            if key_str in {"fps", "framerate", "frame_rate", "frame rate", "framerate_hz"}:
                fps = _coerce_fps(val)
                if fps is not None:
                    return fps
            if "frameinfo" in key_str or "frame info" in key_str:
                fps = _search_fps_in_value(val, depth - 1)
                if fps is not None:
                    return fps
    elif isinstance(value, (list, tuple)):
        for item in value:
            fps = _search_fps_in_value(item, depth - 1)
            if fps is not None:
                return fps
    elif isinstance(value, str) and value.strip().startswith("{"):
        try:
            import json

            parsed = json.loads(value)
        except Exception:
            return None
        return _search_fps_in_value(parsed, depth - 1)
    return None


def _detect_video_fps(video_path: Path) -> Optional[float]:
    return _detect_video_fps_from_telemetry_parser(video_path)


def _format_columns(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = []
    header_parts = [headers[idx].ljust(widths[idx]) for idx in range(len(headers))]
    lines.append("  ".join(header_parts))
    for row in rows:
        row_parts = [row[idx].ljust(widths[idx]) for idx in range(len(headers))]
        lines.append("  ".join(row_parts))
    return "\n".join(lines)


def _format_candidates_table(metrics_all: List[dict], selected_idx: Optional[int] = None) -> str:
    # Simple aligned columns (no external formatting dependency).
    headers = ["signal", "lag_s", "peak", "psr", "stability_s", "score", "windows", ""]
    rows = []
    for idx, m in enumerate(metrics_all):
        selected_label = "[selected]" if selected_idx is not None and idx == selected_idx else ""
        rows.append(
            [
                str(m["signal"]),
                f"{m['lag_seconds']:+.3f}",
                f"{m['peak']:.3f}",
                f"{m['psr']:.3f}",
                f"{m['stability']:.3f}",
                f"{m['score']:.3f}",
                str(m.get("window_count", 1)),
                selected_label,
            ]
        )
    table = _format_columns(headers, rows)
    if selected_idx is None:
        return table
    lines = table.splitlines()
    selected_line_idx = selected_idx + 1  # header is first line
    if 0 <= selected_line_idx < len(lines):
        lines[selected_line_idx] = f"\x1b[1m{lines[selected_line_idx]}\x1b[0m"
    return "\n".join(lines)


def _color_line(width: int = 29, color_code: str = "\x1b[38;5;39m") -> str:
    # Thin colored divider for readability in terminal output.
    return f"{color_code}{'─' * width}\x1b[0m"


def _dump_video_telemetry_keys(video_path: Path, source_name: str) -> None:
    if source_name != "telemetry_parser":
        print("Telemetry key dump is only supported for telemetry_parser sources.")
        return

    try:
        from .sources import telemetry_parser_camera
    except Exception as exc:
        print(f"Telemetry key dump unavailable: {exc}")
        return

    try:
        info = telemetry_parser_camera.inspect_telemetry_keys(video_path)
    except Exception as exc:
        print(f"Telemetry key dump failed: {exc}")
        return

    keys = info.get("keys", [])
    sampled = info.get("sampled", 0)
    total = info.get("total")
    group_keys = info.get("group_keys", {})
    group_sampled = info.get("group_sampled", 0)
    group_total = info.get("group_total")
    human_keys = info.get("human_keys")
    human_sampled = info.get("human_sampled")
    human_total = info.get("human_total")
    human_group_keys = info.get("human_group_keys")
    human_group_sampled = info.get("human_group_sampled")
    human_group_total = info.get("human_group_total")
    parser_attrs = info.get("parser_attrs", {})
    frame_attrs = info.get("frame_attrs", {})
    public_attrs = info.get("public_attrs", [])
    frame_info = info.get("frame_info")
    frame_keys = [k for k in keys if "frame" in k.lower() or "fps" in k.lower()]
    nested_frame_keys = []
    for group, gkeys in group_keys.items():
        for key in gkeys:
            if "frame" in key.lower() or "fps" in key.lower():
                nested_frame_keys.append(f"{group}.{key}")

    print("")
    print("Telemetry Keys")
    print(_color_line())
    if total is not None:
        print(f"Samples\t{sampled}/{total}")
    else:
        print(f"Samples\t{sampled}")
    print(f"Key count\t{len(keys)}")
    if frame_keys:
        print(f"Frame/FPS keys\t{', '.join(frame_keys)}")
    else:
        print("Frame/FPS keys\tnone")
    if nested_frame_keys:
        print(f"Frame/FPS keys (nested)\t{', '.join(sorted(nested_frame_keys))}")
    if keys:
        print("Keys")
        print(", ".join(keys))
    print("")

    if group_keys:
        print("Telemetry Group Keys")
        print(_color_line())
        if group_total is not None:
            print(f"Samples\t{group_sampled}/{group_total}")
        else:
            print(f"Samples\t{group_sampled}")
        for group, gkeys in sorted(group_keys.items()):
            print(f"{group}\t{', '.join(gkeys)}")
        print("")

    if human_keys is not None:
        human_frame_keys = [k for k in human_keys if "frame" in k.lower() or "fps" in k.lower()]
        print("Telemetry Keys (human_readable=True)")
        print(_color_line())
        if human_total is not None:
            print(f"Samples\t{human_sampled}/{human_total}")
        else:
            print(f"Samples\t{human_sampled}")
        print(f"Key count\t{len(human_keys)}")
        if human_frame_keys:
            print(f"Frame/FPS keys\t{', '.join(human_frame_keys)}")
        else:
            print("Frame/FPS keys\tnone")
        if human_group_keys:
            nested_hr = []
            for group, gkeys in human_group_keys.items():
                for key in gkeys:
                    if "frame" in key.lower() or "fps" in key.lower():
                        nested_hr.append(f"{group}.{key}")
            if nested_hr:
                print(f"Frame/FPS keys (nested)\t{', '.join(sorted(nested_hr))}")
        if human_keys:
            print("Keys")
            print(", ".join(human_keys))
        print("")

    if human_group_keys:
        print("Telemetry Group Keys (human_readable=True)")
        print(_color_line())
        if human_group_total is not None:
            print(f"Samples\t{human_group_sampled}/{human_group_total}")
        else:
            print(f"Samples\t{human_group_sampled}")
        for group, gkeys in sorted(human_group_keys.items()):
            print(f"{group}\t{', '.join(gkeys)}")
        print("")

    print("Parser Attributes")
    print(_color_line())
    if frame_info is not None:
        print(f"frame_info\t{frame_info}")
    if frame_attrs:
        print("Frame/FPS attrs")
        for key, val in sorted(frame_attrs.items()):
            print(f"{key}\t{val}")
    else:
        print("Frame/FPS attrs\tnone")
    if parser_attrs:
        print("Scalar attrs")
        for key, val in sorted(parser_attrs.items()):
            print(f"{key}\t{val}")
    if public_attrs:
        print("")
        print("Public attrs")
        print(", ".join(public_attrs))
    print("")


def _status(message: str) -> None:
    # Lightweight progress indicator for long operations.
    print(message, flush=True)


def _format_kv(rows: List[Tuple[str, str]]) -> str:
    # Tab-align key/value pairs.
    width = max(len(key) for key, _ in rows) if rows else 0
    return "\n".join(f"{key.ljust(width)}\t{value}" for key, value in rows)


def _format_signal_selection(
    log: LogData,
    video: ImuBundle,
    signal: str,
    log_rate: float,
    video_rate: float,
) -> str:
    headers = ["source", "signal", "derived_from", "sample_rate"]
    log_sig = derive_signal(log.imu, signal)
    video_sig = derive_signal(video, signal)
    rows = [
        [
            "Video",
            signal,
            _describe_derived(video_sig),
            _format_rate(video_rate),
        ],
        [
            "Log",
            signal,
            _describe_derived(log_sig),
            _format_rate(log_rate),
        ],
    ]
    return _format_columns(headers, rows)


def _save_plot(time_s: np.ndarray, log: np.ndarray, video: np.ndarray, corr, lags, fs: float) -> None:
    # Optional diagnostics plot for visual inspection.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Plotting requested but matplotlib is not installed.") from exc

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(time_s, log, label="Log", linewidth=1.0)
    axes[0].plot(time_s, video, label="Video", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Preprocessed Signals")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="best")

    axes[1].plot(lags / fs, corr, color="black", linewidth=1.0)
    axes[1].set_title("Cross-Correlation")
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Correlation")

    axes[2].plot(time_s, log - video, color="tab:orange", linewidth=1.0)
    axes[2].set_title("Signal Difference (Log - Video)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")

    fig.tight_layout()
    fig.savefig("sync_plot.png", dpi=150)


def main(argv: Optional[List[str]] = None) -> None:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Sync telemetry logs to camera video using IMU cross-correlation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit.",
    )

    parser.add_argument("--video", help="Path to video file (MP4)")
    parser.add_argument("--log", help="Path to log file (CSV)")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Video frame rate for timecode offset output (e.g., 29.97, 59.94, 60)",
    )
    parser.add_argument("--video-source", help="Force video source by name")
    parser.add_argument("--log-source", help="Force log source by name")
    parser.add_argument(
        "--video-opt",
        action="append",
        default=[],
        help="Video source option key=value (repeatable).",
    )
    parser.add_argument(
        "--log-opt",
        action="append",
        default=[],
        help="Log source option key=value (repeatable).",
    )
    parser.add_argument("--signal", default="auto", help="Signal to use for correlation")
    parser.add_argument(
        "--signals",
        help="Comma-separated list of signals to evaluate (overrides --signal).",
    )
    parser.add_argument("--max-lag", type=float, default=600.0, help="Max lag to search (seconds)")
    parser.add_argument("--window", type=float, default=360.0, help="Window length (seconds)")
    parser.add_argument(
        "--no-auto-window-size",
        action="store_false",
        dest="auto_window_size",
        help="Disable automatic window-size selection.",
    )
    parser.add_argument("--start", type=float, default=None, help="Window start time (seconds)")
    parser.add_argument(
        "--no-auto-window",
        action="store_false",
        dest="auto_window",
        help="Disable multi-window consensus and use a single window.",
    )
    parser.add_argument(
        "--window-step",
        type=float,
        default=20.0,
        help="Step size for auto-window scan (seconds).",
    )
    parser.add_argument(
        "--show-drift",
        action="store_true",
        help="Show drift estimate (can be noisy on short or low-activity clips).",
    )
    parser.add_argument(
        "--dump-video-telemetry-keys",
        action="store_true",
        help="Print raw telemetry keys from telemetry-parser (debug aid).",
    )
    parser.add_argument("--fs", type=float, default=50.0, help="Resample rate (Hz)")
    parser.add_argument("--lowpass-hz", type=float, default=8.0, help="Low-pass cutoff (Hz)")
    parser.add_argument("--highpass-hz", type=float, default=0.2, help="High-pass cutoff (Hz)")
    parser.add_argument("--log-time-col", help="Override log time column name")
    parser.add_argument("--log-gyro-cols", help="Comma-separated log gyro column names")
    parser.add_argument("--log-acc-cols", help="Comma-separated log accel column names")
    parser.add_argument("--write-video-imu-csv", action="store_true", help="Write video_imu.csv")
    parser.add_argument("--write-shifted-log", action="store_true", help="Write log_shifted.csv")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plot to sync_plot.png")
    parser.set_defaults(auto_window=True)
    parser.set_defaults(auto_window_size=True)

    args = parser.parse_args(argv)
    window_is_default = "--window" not in raw_args
    window_step_is_default = "--window-step" not in raw_args
    max_lag_is_default = "--max-lag" not in raw_args
    fs_is_default = "--fs" not in raw_args
    lowpass_is_default = "--lowpass-hz" not in raw_args
    highpass_is_default = "--highpass-hz" not in raw_args

    try:
        _status("Resolving input files...")
        video_path, log_path = _resolve_paths(args.video, args.log)
        video_fps = args.video_fps if args.video_fps is not None else _detect_video_fps(video_path)

        video_opts = _parse_kv_args(args.video_opt)
        log_opts = _parse_kv_args(args.log_opt)
        if args.log_time_col:
            log_opts["time_col"] = args.log_time_col
        if args.log_gyro_cols:
            log_opts["gyro_cols"] = _parse_cols(args.log_gyro_cols)
        if args.log_acc_cols:
            log_opts["acc_cols"] = _parse_cols(args.log_acc_cols)

        video_source = resolve_source("video", video_path, forced=args.video_source)
        log_source = resolve_source("log", log_path, forced=args.log_source)

        if args.dump_video_telemetry_keys:
            _dump_video_telemetry_keys(video_path, video_source.name)

        _status(f"Loading log: {log_path.name} ({log_source.name})")
        log = log_source.load(log_path, **log_opts)

        _status(f"Loading video IMU: {video_path.name} ({video_source.name}) (this can take a while)")
        video = video_source.load(video_path, **video_opts)

        available = sorted(
            available_signals(log.imu) & available_signals(video),
            key=lambda s: SIGNAL_PRIORITY.index(s) if s in SIGNAL_PRIORITY else 99,
        )
        if not available:
            raise ValueError("No compatible signals found between log and video data.")

        selected_signals: List[str]
        if args.signals:
            requested = [s.strip() for s in args.signals.split(",") if s.strip()]
            selected_signals = []
            for sig in requested:
                if sig in available:
                    selected_signals.append(sig)
                else:
                    print(f"Warning: Requested signal {sig} not available; skipping.")
            if not selected_signals:
                selected_signals = available
        elif args.signal.lower() in ("auto", "all"):
            selected_signals = available
        else:
            signal, warning = choose_signal(args.signal, log.imu, video)
            if warning:
                print(f"Warning: {warning}")
            selected_signals = [signal]

        if fs_is_default:
            log_rate = infer_sample_rate(np.asarray(log.time_s, dtype=float))
            video_rate = _bundle_rate(video)
            rates = [r for r in (log_rate, video_rate) if np.isfinite(r) and r > 0]
            if len(rates) == 2:
                auto_fs = min(50.0, max(20.0, float(np.sqrt(rates[0] * rates[1]))))
            elif rates:
                auto_fs = min(50.0, max(20.0, rates[0]))
            else:
                auto_fs = args.fs
            if abs(auto_fs - args.fs) > 1e-6:
                print(f"Info: Auto sample rate set to {auto_fs:.1f} Hz.")
                args.fs = auto_fs

        duration_s = min(
            _safe_duration(np.asarray(log.time_s, dtype=float)),
            _bundle_duration(video),
        )
        if max_lag_is_default and duration_s > 0:
            auto_max_lag = min(600.0, max(30.0, 0.5 * duration_s))
            if auto_max_lag < args.max_lag - 1e-6:
                print(f"Info: Auto max lag set to {auto_max_lag:.1f}s.")
                args.max_lag = auto_max_lag

        if (
            args.auto_window_size
            and window_is_default
            and args.auto_window
            and args.start is None
        ):
            selected_window, candidates = _select_window_size(
                log=log,
                video=video,
                signals=selected_signals,
                fs=args.fs,
                lowpass_hz=args.lowpass_hz,
                highpass_hz=args.highpass_hz,
                max_lag_s=args.max_lag,
                window_step_s=args.window_step,
                auto_window=args.auto_window,
                window_step_is_default=window_step_is_default,
            )
            if selected_window > 0:
                args.window = selected_window
                if candidates:
                    cand_str = ", ".join(f"{c:.0f}" for c in candidates)
                    print(
                        f"Auto window size selected: {selected_window:.1f}s (candidates: {cand_str})"
                    )
                else:
                    print(f"Auto window size selected: {selected_window:.1f}s")

        if lowpass_is_default:
            auto_lowpass = min(8.0, 0.45 * args.fs)
            if abs(auto_lowpass - args.lowpass_hz) > 1e-6:
                print(f"Info: Auto lowpass set to {auto_lowpass:.2f} Hz.")
                args.lowpass_hz = auto_lowpass

        if highpass_is_default:
            target_cycles = 3.0
            auto_highpass = target_cycles / max(10.0, args.window)
            auto_highpass = max(0.1, min(0.4, auto_highpass))
            if abs(auto_highpass - args.highpass_hz) > 1e-6:
                print(f"Info: Auto highpass set to {auto_highpass:.2f} Hz.")
                args.highpass_hz = auto_highpass

        best = None
        best_idx: Optional[int] = None
        metrics_all = []
        for sig in selected_signals:
            _status(f"Computing correlation metrics for signal: {sig}")
            metrics = _compute_metrics(
                log=log,
                video=video,
                signal=sig,
                fs=args.fs,
                window_s=args.window,
                lowpass_hz=args.lowpass_hz,
                highpass_hz=args.highpass_hz,
                max_lag_s=args.max_lag,
                start_override=args.start,
                auto_window=args.auto_window,
                window_step_s=args.window_step,
                window_is_default=window_is_default,
                window_step_is_default=window_step_is_default,
                emit_warnings=(sig == selected_signals[0]),
            )
            metrics_all.append(metrics)
            if best is None or metrics["score"] > best["score"]:
                best = metrics
                best_idx = len(metrics_all) - 1

        if best is None:
            raise ValueError("Failed to compute lag for selected signals.")

        log_rate = best["log_rate"]
        video_rate = best["video_rate"]

        print("")
        print("Signal Candidates")
        print(_color_line())
        print(_format_candidates_table(metrics_all, selected_idx=best_idx))
        print("")

        lag_seconds = best["lag_seconds"]
        peak = best["peak"]
        psr = best["psr"]
        stability_std = best["stability"]
        drift_info = best.get("drift")
        corr = best["corr"]
        lags = best["lags"]

        print("")
        print("Sync Summary")
        print(_color_line())
        conf_score = _confidence_score(peak, psr, stability_std)
        conf_label = _confidence_rating(conf_score)
        summary_rows = [
            ("Correlation peak", f"{peak:.3f}"),
            ("Peak-to-sidelobe ratio", f"{psr:.3f}"),
            ("Stability (stddev s)", f"{stability_std:.3f}"),
            ("Confidence", f"{conf_label} ({conf_score:.0f}/100)"),
        ]
        if args.show_drift:
            if drift_info and drift_info.get("reliable"):
                summary_rows.append(("Drift (s/s)", f"{drift_info['slope']:+.6f}"))
            else:
                summary_rows.append(("Drift (s/s)", "n/a (insufficient reliability)"))
        print(_format_kv(summary_rows))
        print("")
        print("Offset Summary")
        print(_color_line())
        print(_format_kv(_offset_summary_rows(lag_seconds, video_fps)))
        if video_fps is None or video_fps <= 0:
            print("Warning: FPS unavailable; skipping frame/timecode offsets.")
        print("")

        if peak < 0.2:
            print("Warning: Low correlation peak; alignment may be unreliable.")
        if np.isfinite(stability_std) and stability_std > 0.2:
            print("Warning: High lag variability across subwindows; alignment may be unstable.")

        if args.write_video_imu_csv:
            write_imu_csv(Path("video_imu.csv"), video)
            print("Wrote video_imu.csv")

        if args.write_shifted_log:
            write_shifted_log(Path("log_shifted.csv"), log.df, log.time_col, log.time_s, lag_seconds)
            print("Wrote log_shifted.csv")

        if args.plot:
            time_rel = best["log_t"] - float(best["log_t"][0])
            _save_plot(time_rel, best["log_y"], best["video_y"], corr, lags, args.fs)
            print("Wrote sync_plot.png")

    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
