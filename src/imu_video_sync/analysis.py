from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

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


EmitFn = Callable[[str], None]


@dataclass
class SyncOptions:
    video_fps: Optional[float] = None
    video_source: Optional[str] = None
    log_source: Optional[str] = None
    video_opt: dict = field(default_factory=dict)
    log_opt: dict = field(default_factory=dict)
    signal: str = "auto"
    signals: Optional[List[str]] = None
    max_lag: float = 600.0
    window: float = 360.0
    auto_window: bool = True
    auto_window_size: bool = True
    window_step: float = 20.0
    start: Optional[float] = None
    fs: float = 50.0
    lowpass_hz: float = 8.0
    highpass_hz: float = 0.2
    show_drift: bool = False
    dump_video_telemetry_keys: bool = False
    write_video_imu_csv: bool = False
    write_shifted_log: bool = False
    plot: bool = False
    window_is_default: bool = True
    window_step_is_default: bool = True
    max_lag_is_default: bool = True
    fs_is_default: bool = True
    lowpass_is_default: bool = True
    highpass_is_default: bool = True


@dataclass
class SignalCandidate:
    signal: str
    lag_seconds: float
    peak: float
    psr: float
    stability: float
    score: float
    window_count: int


@dataclass
class Diagnostics:
    confidence_label: str
    confidence_score: float
    correlation_peak: float
    psr: float
    stability: float
    signal: str


@dataclass
class OffsetSummary:
    lag_seconds: float
    lag_seconds_str: str
    lag_frames: Optional[str]
    timecode_offset: Optional[str]
    project_position: str
    project_label: str
    is_video_offset: bool


@dataclass
class PlotData:
    time_s: np.ndarray
    log_y: np.ndarray
    video_time_s: np.ndarray
    video_y: np.ndarray
    corr: np.ndarray
    lags: np.ndarray
    fs: float
    lag_seconds: float


@dataclass
class SyncResult:
    diagnostics: Diagnostics
    offsets: OffsetSummary
    candidates: List[SignalCandidate]
    selected_index: int
    plot: PlotData
    log_rate: float
    video_rate: float
    video_fps: Optional[float]
    drift_info: Optional[dict]
    post_summary_warnings: List[str] = field(default_factory=list)
    available_signals: List[str] = field(default_factory=list)
    selected_signals: List[str] = field(default_factory=list)
    log_signals: List[str] = field(default_factory=list)
    video_signals: List[str] = field(default_factory=list)


def _emit(emit: Optional[EmitFn], message: str) -> None:
    if emit is not None:
        emit(message)


def _autodetect_file(ext: str) -> Optional[Path]:
    matches = [p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() == ext]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_paths(video_arg: Optional[str | Path], log_arg: Optional[str | Path]) -> tuple[Path, Path]:
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


def _sort_signals(signals: Iterable[str]) -> List[str]:
    return sorted(
        signals,
        key=lambda s: SIGNAL_PRIORITY.index(s) if s in SIGNAL_PRIORITY else 99,
    )


def _format_signal_list(signals: Sequence[str]) -> str:
    return ", ".join(signals) if signals else "none"


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


def _drop_nan(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(time_s) & np.isfinite(values)
    if mask.sum() < 2:
        raise ValueError("Too few finite samples after removing NaNs.")
    return time_s[mask], values[mask]


def _clamp_start(start_s: float, log_t: np.ndarray, video_t: np.ndarray, window_s: float) -> float:
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
    return 100.0 * (0.4 * peak_score + 0.4 * psr_score + 0.2 * stability_score)


def _confidence_rating(score: float) -> str:
    if score >= 75.0:
        return "High"
    if score >= 55.0:
        return "Medium"
    return "Low"


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
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
    log_sig = derive_signal(log.imu, signal)
    video_sig = derive_signal(video, signal)

    log_time, log_signal = _drop_nan(log_sig.time_s, log_sig.values)
    video_time, video_signal = _drop_nan(video_sig.time_s, video_sig.values)

    log_rate = infer_sample_rate(log_time)
    video_rate = infer_sample_rate(video_time)

    log_t_full, log_y_full = resample_uniform(log_time, log_signal, fs)
    video_t_full, video_y_full = resample_uniform(video_time, video_signal, fs)

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

    log_filt = filter_signal(log_y_full, fs, lowpass_hz, highpass_hz)
    video_filt = filter_signal(video_y_full, fs, lowpass_hz, highpass_hz)
    log_norm_full = normalize_signal(log_filt)
    video_norm = normalize_signal(video_filt)

    win_n = int(round(window_s * fs))
    video_duration = float(video_t_full[-1]) if video_t_full.size else 0.0

    if not auto_window:
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
            "log_t_full": log_t_full,
            "log_y_full": log_norm_full,
            "video_t_full": video_t_full,
            "video_y_full": video_norm,
            "start_s": start_s,
            "window_count": 1,
            "drift": None,
        }

    start_idx_min = 0
    if start_override is not None:
        start_idx_min = int(round(float(start_override) * fs))

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
        "log_t_full": log_t_full,
        "log_y_full": log_norm_full,
        "video_t_full": video_t_full,
        "video_y_full": video_norm,
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


def _offset_summary_rows(lag_seconds: float, fps: Optional[float]) -> list[tuple[str, str]]:
    if lag_seconds > 0:
        offset_label = "Video offset"
    elif lag_seconds < 0:
        offset_label = "Data offset"
    else:
        offset_label = "Video offset"
    rows = [
        ("Lag (seconds)", f"{lag_seconds:+.3f}"),
        (offset_label, _format_hhmmss_ms(lag_seconds)),
    ]
    if fps is not None and fps > 0:
        rows.insert(1, ("Lag (frames)", _format_lag_frames(lag_seconds, fps)))
        rows.insert(2, ("Timecode offset", _format_timecode(lag_seconds, fps)))
    return rows


def _offset_summary_payload(lag_seconds: float, fps: Optional[float]) -> dict:
    payload = {"lag_seconds": f"{lag_seconds:+.3f}"}
    if fps is not None and fps > 0:
        payload["lag_frames"] = _format_lag_frames(lag_seconds, fps)
        payload["timecode_offset"] = _format_timecode(lag_seconds, fps)
    else:
        payload["lag_frames"] = None
        payload["timecode_offset"] = None
    offset_key = "video_offset" if lag_seconds >= 0 else "data_offset"
    payload[offset_key] = _format_hhmmss_ms(lag_seconds)
    return payload


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
            fps = _search_fps_in_value(group_val)
            if fps is not None:
                return fps
    return None


def _iter_telemetry_samples(data: object) -> Iterable[object]:
    if isinstance(data, dict):
        values = data.get("samples")
        if isinstance(values, list):
            return values
    if isinstance(data, list):
        return data
    return []


def _coerce_fps(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        fps = float(value)
        if 1.0 < fps < 240.0:
            return fps
    return None


def _search_fps_in_value(value: object, depth: int = 3) -> Optional[float]:
    if depth <= 0:
        return None
    if isinstance(value, dict):
        for key, val in value.items():
            key_str = str(key).lower()
            if "fps" in key_str:
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


def _dump_video_telemetry_keys(video_path: Path, source_name: str, emit: Optional[EmitFn]) -> None:
    if source_name != "telemetry_parser":
        _emit(emit, "Telemetry key dump is only supported for telemetry_parser sources.")
        return

    try:
        from .sources import telemetry_parser_camera
    except Exception as exc:
        _emit(emit, f"Telemetry key dump unavailable: {exc}")
        return

    try:
        info = telemetry_parser_camera.inspect_telemetry_keys(video_path)
    except Exception as exc:
        _emit(emit, f"Telemetry key dump failed: {exc}")
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

    emit_lines: list[str] = []
    emit_lines.append("")
    emit_lines.append("Telemetry Keys")
    emit_lines.append(_color_line())
    if total is not None:
        emit_lines.append(f"Samples\t{sampled}/{total}")
    else:
        emit_lines.append(f"Samples\t{sampled}")
    emit_lines.append(f"Key count\t{len(keys)}")
    if frame_keys:
        emit_lines.append(f"Frame/FPS keys\t{', '.join(frame_keys)}")
    else:
        emit_lines.append("Frame/FPS keys\tnone")
    if nested_frame_keys:
        emit_lines.append(f"Frame/FPS keys (nested)\t{', '.join(sorted(nested_frame_keys))}")
    if keys:
        emit_lines.append("Keys")
        emit_lines.append(", ".join(keys))
    emit_lines.append("")

    if group_keys:
        emit_lines.append("Telemetry Group Keys")
        emit_lines.append(_color_line())
        if group_total is not None:
            emit_lines.append(f"Samples\t{group_sampled}/{group_total}")
        else:
            emit_lines.append(f"Samples\t{group_sampled}")
        for group, gkeys in sorted(group_keys.items()):
            emit_lines.append(f"{group}\t{', '.join(gkeys)}")
        emit_lines.append("")

    if human_keys is not None:
        human_frame_keys = [k for k in human_keys if "frame" in k.lower() or "fps" in k.lower()]
        emit_lines.append("Telemetry Keys (human_readable=True)")
        emit_lines.append(_color_line())
        if human_total is not None:
            emit_lines.append(f"Samples\t{human_sampled}/{human_total}")
        else:
            emit_lines.append(f"Samples\t{human_sampled}")
        emit_lines.append(f"Key count\t{len(human_keys)}")
        if human_frame_keys:
            emit_lines.append(f"Frame/FPS keys\t{', '.join(human_frame_keys)}")
        else:
            emit_lines.append("Frame/FPS keys\tnone")
        if human_group_keys:
            nested_hr = []
            for group, gkeys in human_group_keys.items():
                for key in gkeys:
                    if "frame" in key.lower() or "fps" in key.lower():
                        nested_hr.append(f"{group}.{key}")
            if nested_hr:
                emit_lines.append(f"Frame/FPS keys (nested)\t{', '.join(sorted(nested_hr))}")
        if human_keys:
            emit_lines.append("Keys")
            emit_lines.append(", ".join(human_keys))
        emit_lines.append("")

    if human_group_keys:
        emit_lines.append("Telemetry Group Keys (human_readable=True)")
        emit_lines.append(_color_line())
        if human_group_total is not None:
            emit_lines.append(f"Samples\t{human_group_sampled}/{human_group_total}")
        else:
            emit_lines.append(f"Samples\t{human_group_sampled}")
        for group, gkeys in sorted(human_group_keys.items()):
            emit_lines.append(f"{group}\t{', '.join(gkeys)}")
        emit_lines.append("")

    emit_lines.append("Parser Attributes")
    emit_lines.append(_color_line())
    if frame_info is not None:
        emit_lines.append(f"frame_info\t{frame_info}")
    if frame_attrs:
        emit_lines.append("Frame/FPS attrs")
        for key, val in sorted(frame_attrs.items()):
            emit_lines.append(f"{key}\t{val}")
    else:
        emit_lines.append("Frame/FPS attrs\tnone")
    if parser_attrs:
        emit_lines.append("Scalar attrs")
        for key, val in sorted(parser_attrs.items()):
            emit_lines.append(f"{key}\t{val}")
    if public_attrs:
        emit_lines.append("")
        emit_lines.append("Public attrs")
        emit_lines.append(", ".join(public_attrs))
    emit_lines.append("")

    for line in emit_lines:
        _emit(emit, line)


def _color_line(width: int = 29, color_code: str = "\x1b[38;5;39m") -> str:
    return f"{color_code}{'-' * width}\x1b[0m"


def run_sync(
    video_path: Optional[str | Path],
    log_path: Optional[str | Path],
    *,
    options: SyncOptions,
    emit: Optional[EmitFn] = None,
) -> SyncResult:
    _emit(emit, "Resolving input files...")
    video_path_resolved, log_path_resolved = _resolve_paths(video_path, log_path)
    video_fps = options.video_fps
    if video_fps is None:
        video_fps = _detect_video_fps(video_path_resolved)

    video_opts = dict(options.video_opt or {})
    log_opts = dict(options.log_opt or {})

    video_source = resolve_source("video", video_path_resolved, forced=options.video_source)
    log_source = resolve_source("log", log_path_resolved, forced=options.log_source)

    if options.dump_video_telemetry_keys:
        _dump_video_telemetry_keys(video_path_resolved, video_source.name, emit)

    _emit(emit, f"Loading log: {log_path_resolved.name} ({log_source.name})")
    log = log_source.load(log_path_resolved, **log_opts)

    _emit(
        emit,
        f"Loading video IMU: {video_path_resolved.name} ({video_source.name}) (this can take a while)",
    )
    video = video_source.load(video_path_resolved, **video_opts)

    log_signals = _sort_signals(available_signals(log.imu))
    video_signals = _sort_signals(available_signals(video))
    available = _sort_signals(set(log_signals) & set(video_signals))
    _emit(emit, f"Available signals (log): {_format_signal_list(log_signals)}")
    _emit(emit, f"Available signals (video): {_format_signal_list(video_signals)}")
    _emit(emit, f"Compatible signals: {_format_signal_list(available)}")
    if not available:
        raise ValueError("No compatible signals found between log and video data.")

    selected_signals: List[str]
    if options.signals:
        requested = [s.strip() for s in options.signals if s.strip()]
        selected_signals = []
        for sig in requested:
            if sig in available:
                selected_signals.append(sig)
            else:
                _emit(emit, f"Warning: Requested signal {sig} not available; skipping.")
        if not selected_signals:
            selected_signals = available
    elif options.signal.lower() in ("auto", "all"):
        selected_signals = available
    else:
        signal, warning = choose_signal(options.signal, log.imu, video)
        if warning:
            _emit(emit, f"Warning: {warning}")
        selected_signals = [signal]
    _emit(emit, f"Selected signals for evaluation: {_format_signal_list(selected_signals)}")

    if options.fs_is_default:
        log_rate = infer_sample_rate(np.asarray(log.time_s, dtype=float))
        video_rate = _bundle_rate(video)
        rates = [r for r in (log_rate, video_rate) if np.isfinite(r) and r > 0]
        if len(rates) == 2:
            auto_fs = min(50.0, max(20.0, float(np.sqrt(rates[0] * rates[1]))))
        elif rates:
            auto_fs = min(50.0, max(20.0, rates[0]))
        else:
            auto_fs = options.fs
        if abs(auto_fs - options.fs) > 1e-6:
            _emit(emit, f"Info: Auto sample rate set to {auto_fs:.1f} Hz.")
            options.fs = auto_fs

    duration_s = min(
        _safe_duration(np.asarray(log.time_s, dtype=float)),
        _bundle_duration(video),
    )
    if options.max_lag_is_default and duration_s > 0:
        auto_max_lag = min(600.0, max(30.0, 0.5 * duration_s))
        if auto_max_lag < options.max_lag - 1e-6:
            _emit(emit, f"Info: Auto max lag set to {auto_max_lag:.1f}s.")
            options.max_lag = auto_max_lag

    if (
        options.auto_window_size
        and options.window_is_default
        and options.auto_window
        and options.start is None
    ):
        selected_window, candidates = _select_window_size(
            log=log,
            video=video,
            signals=selected_signals,
            fs=options.fs,
            lowpass_hz=options.lowpass_hz,
            highpass_hz=options.highpass_hz,
            max_lag_s=options.max_lag,
            window_step_s=options.window_step,
            auto_window=options.auto_window,
            window_step_is_default=options.window_step_is_default,
        )
        if selected_window > 0:
            options.window = selected_window
            if candidates:
                cand_str = ", ".join(f"{c:.0f}" for c in candidates)
                _emit(
                    emit,
                    f"Auto window size selected: {selected_window:.1f}s (candidates: {cand_str})",
                )
            else:
                _emit(emit, f"Auto window size selected: {selected_window:.1f}s")

    if options.lowpass_is_default:
        auto_lowpass = min(8.0, 0.45 * options.fs)
        if abs(auto_lowpass - options.lowpass_hz) > 1e-6:
            _emit(emit, f"Info: Auto lowpass set to {auto_lowpass:.2f} Hz.")
            options.lowpass_hz = auto_lowpass

    if options.highpass_is_default:
        target_cycles = 3.0
        auto_highpass = target_cycles / max(10.0, options.window)
        auto_highpass = max(0.1, min(0.4, auto_highpass))
        if abs(auto_highpass - options.highpass_hz) > 1e-6:
            _emit(emit, f"Info: Auto highpass set to {auto_highpass:.2f} Hz.")
            options.highpass_hz = auto_highpass

    best = None
    best_idx: Optional[int] = None
    metrics_all: list[dict] = []
    for sig in selected_signals:
        _emit(emit, f"Computing correlation metrics for signal: {sig}")
        metrics = _compute_metrics(
            log=log,
            video=video,
            signal=sig,
            fs=options.fs,
            window_s=options.window,
            lowpass_hz=options.lowpass_hz,
            highpass_hz=options.highpass_hz,
            max_lag_s=options.max_lag,
            start_override=options.start,
            auto_window=options.auto_window,
            window_step_s=options.window_step,
            window_is_default=options.window_is_default,
            window_step_is_default=options.window_step_is_default,
            emit_warnings=(sig == selected_signals[0]),
        )
        metrics_all.append(metrics)
        if best is None or metrics["score"] > best["score"]:
            best = metrics
            best_idx = len(metrics_all) - 1

    if best is None or best_idx is None:
        raise ValueError("Failed to compute lag for selected signals.")

    log_rate = best["log_rate"]
    video_rate = best["video_rate"]

    lag_seconds = best["lag_seconds"]
    peak = best["peak"]
    psr = best["psr"]
    stability_std = best["stability"]
    drift_info = best.get("drift")
    corr = best["corr"]
    lags = best["lags"]

    conf_score = _confidence_score(peak, psr, stability_std)
    conf_label = _confidence_rating(conf_score)

    if options.write_video_imu_csv:
        write_imu_csv(Path("video_imu.csv"), video)
        _emit(emit, "Wrote video_imu.csv")

    if options.write_shifted_log:
        write_shifted_log(Path("log_shifted.csv"), log.df, log.time_col, log.time_s, lag_seconds)
        _emit(emit, "Wrote log_shifted.csv")

    if options.plot:
        _save_plot(best, options.fs)
        _emit(emit, "Wrote sync_plot.png")

    candidates = [
        SignalCandidate(
            signal=m["signal"],
            lag_seconds=float(m["lag_seconds"]),
            peak=float(m["peak"]),
            psr=float(m["psr"]),
            stability=float(m["stability"]),
            score=float(m["score"]),
            window_count=int(m.get("window_count", 1)),
        )
        for m in metrics_all
    ]

    is_video_offset = lag_seconds >= 0
    project_label = "Video offset" if is_video_offset else "Data offset"
    offsets = OffsetSummary(
        lag_seconds=float(lag_seconds),
        lag_seconds_str=f"{lag_seconds:+.3f}",
        lag_frames=_format_lag_frames(lag_seconds, video_fps)
        if video_fps is not None and video_fps > 0
        else None,
        timecode_offset=_format_timecode(lag_seconds, video_fps)
        if video_fps is not None and video_fps > 0
        else None,
        project_position=_format_hhmmss_ms(lag_seconds),
        project_label=project_label,
        is_video_offset=is_video_offset,
    )

    diagnostics = Diagnostics(
        confidence_label=conf_label,
        confidence_score=conf_score,
        correlation_peak=float(peak),
        psr=float(psr),
        stability=float(stability_std),
        signal=best["signal"],
    )

    plot = PlotData(
        time_s=np.asarray(best["log_t_full"], dtype=float),
        log_y=np.asarray(best["log_y_full"], dtype=float),
        video_time_s=np.asarray(best["video_t_full"], dtype=float),
        video_y=np.asarray(best["video_y_full"], dtype=float),
        corr=np.asarray(corr, dtype=float),
        lags=np.asarray(lags, dtype=float),
        fs=float(options.fs),
        lag_seconds=float(lag_seconds),
    )

    post_warnings: list[str] = []
    if video_fps is None or video_fps <= 0:
        post_warnings.append("Warning: FPS unavailable; skipping frame/timecode offsets.")
    if peak < 0.2:
        post_warnings.append("Warning: Low correlation peak; alignment may be unreliable.")
    if np.isfinite(stability_std) and stability_std > 0.2:
        post_warnings.append(
            "Warning: High lag variability across subwindows; alignment may be unstable."
        )

    return SyncResult(
        diagnostics=diagnostics,
        offsets=offsets,
        candidates=candidates,
        selected_index=best_idx,
        plot=plot,
        log_rate=float(log_rate),
        video_rate=float(video_rate),
        video_fps=video_fps,
        drift_info=drift_info,
        post_summary_warnings=post_warnings,
        available_signals=list(available),
        selected_signals=list(selected_signals),
        log_signals=list(log_signals),
        video_signals=list(video_signals),
    )


def _save_plot(best: dict, fs: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Plotting requested but matplotlib is not installed.") from exc

    time_s = np.asarray(best["log_t"], dtype=float)
    log = np.asarray(best["log_y"], dtype=float)
    video = np.asarray(best["video_y"], dtype=float)
    corr = np.asarray(best["corr"], dtype=float)
    lags = np.asarray(best["lags"], dtype=float)

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
