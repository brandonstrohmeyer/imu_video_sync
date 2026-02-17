from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

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
    threshold = max(1e-9, var_threshold_ratio * max_var)

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
        print(
            f"Warning: Window {window_s:.1f}s exceeds available data. "
            f"Shrinking to {new_window:.1f}s."
        )
        window_s = new_window
    if auto_window and window_s >= 0.99 * video_duration:
        print(
            "Warning: Auto-window disabled because the window length "
            "nearly equals the video duration."
        )
        auto_window = False

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
            stability_lag_s = min(30.0, max_lag_s)
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
    keep_n = max(5, int(0.4 * len(order)))
    keep_idx = order[:keep_n]

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

    min_span_s = max(120.0, 0.2 * video_duration) if video_duration > 0 else 120.0
    drift_info = _estimate_drift_info(
        np.array([c["start_s"] for c in kept], dtype=float),
        lag_values,
        min_windows=6,
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
    return [
        ("Lag (seconds)", f"{lag_seconds:+.3f}"),
        ("Lag (frames)", _format_lag_frames(lag_seconds, fps)),
        ("Timecode offset", _format_timecode(lag_seconds, fps)),
        (offset_label, _format_hhmmss_ms(lag_seconds)),
    ]


def _find_exiftool() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in ("tools/exiftool.exe", "tools/exiftool(-k).exe"):
        candidate = repo_root / rel
        if candidate.exists():
            return candidate
    system_path = shutil.which("exiftool")
    if system_path:
        return Path(system_path)
    return None


def _detect_video_fps(video_path: Path) -> Optional[float]:
    exiftool = _find_exiftool()
    if exiftool is None:
        return None
    tags = [
        "VideoFrameRate",
        "PlaybackFrameRate",
        "TrackFrameRate",
        "AvgFrameRate",
        "AverageFrameRate",
        "VideoAvgFrameRate",
        "FrameRate",
        "MovieFrameRate",
        "OriginalFrameRate",
        "CaptureFrameRate",
        "FPS",
    ]
    cmd = [str(exiftool), "-api", "largefilesupport=1", "-n", "-j"]
    cmd += [f"-{tag}" for tag in tags]
    cmd.append(str(video_path))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not payload:
        return None
    row = payload[0]
    for tag in tags:
        value = row.get(tag)
        if value is None:
            continue
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return fps
    return None


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


def _format_candidates_table(metrics_all: List[dict]) -> str:
    # Simple aligned columns (no external formatting dependency).
    headers = ["signal", "lag_s", "peak", "psr", "stability_s", "score", "windows"]
    rows = []
    for m in metrics_all:
        rows.append(
            [
                str(m["signal"]),
                f"{m['lag_seconds']:+.3f}",
                f"{m['peak']:.3f}",
                f"{m['psr']:.3f}",
                f"{m['stability']:.3f}",
                f"{m['score']:.3f}",
                str(m.get("window_count", 1)),
            ]
        )
    return _format_columns(headers, rows)


def _color_line(width: int = 29, color_code: str = "\x1b[38;5;39m") -> str:
    # Thin colored divider for readability in terminal output.
    return f"{color_code}{'─' * width}\x1b[0m"


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
    parser = argparse.ArgumentParser(
        description="Sync telemetry logs to camera video using IMU cross-correlation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument("--signal", default="gyroMag", help="Signal to use for correlation")
    parser.add_argument(
        "--signals",
        help="Comma-separated list of signals to evaluate (overrides --signal).",
    )
    parser.add_argument("--max-lag", type=float, default=600.0, help="Max lag to search (seconds)")
    parser.add_argument("--window", type=float, default=360.0, help="Window length (seconds)")
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

    args = parser.parse_args(argv)

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

        best = None
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
            )
            metrics_all.append(metrics)
            if best is None or metrics["score"] > best["score"]:
                best = metrics

        if best is None:
            raise ValueError("Failed to compute lag for selected signals.")

        log_rate = best["log_rate"]
        video_rate = best["video_rate"]

        print("")
        print("Signal Selection")
        print(_color_line())
        print(_format_signal_selection(log, video, best["signal"], log_rate, video_rate))
        if len(metrics_all) > 1:
            print("")
            print("Signal Candidates")
            print(_color_line())
            print(_format_candidates_table(metrics_all))
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
