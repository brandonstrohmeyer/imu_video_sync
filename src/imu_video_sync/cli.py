from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from .aim_csv import AimData, load_aim_csv
from .gopro_extract import GoproIMU, extract_gopro_imu
from .preprocess import (
    filter_signal,
    infer_sample_rate,
    normalize_signal,
    preprocess_signal,
    resample_uniform,
    select_active_window,
    trim_window,
)
from .correlate import estimate_lag, lag_stability, peak_to_sidelobe
from .io_out import write_gopro_csv, write_shifted_csv


SIGNAL_PRIORITY = ["gyroMag", "yawRate", "latAcc", "accMag", "gyroZ"]


def _parse_cols(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [col.strip() for col in value.split(",") if col.strip()]


def _autodetect_file(ext: str) -> Optional[Path]:
    # Find exactly one file with the given extension in the current directory.
    matches = [p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() == ext]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_paths(mp4_arg: Optional[str], csv_arg: Optional[str]) -> tuple[Path, Path]:
    # Prefer explicit paths, otherwise auto-detect.
    mp4_path = Path(mp4_arg) if mp4_arg else _autodetect_file(".mp4")
    csv_path = Path(csv_arg) if csv_arg else _autodetect_file(".csv")

    if mp4_path is None or csv_path is None:
        raise ValueError(
            "Auto-detect failed. Provide --mp4 and --csv when there is not exactly one .mp4 and one .csv in the directory."
        )
    if not mp4_path.exists():
        raise ValueError(f"MP4 not found: {mp4_path}")
    if not csv_path.exists():
        raise ValueError(f"CSV not found: {csv_path}")
    return mp4_path, csv_path


def _available_signals_aim(aim: AimData) -> set[str]:
    # Determine which signals can be derived from AiM data.
    available = set()
    if aim.gyro is not None and aim.gyro.size:
        available.update(["gyroMag", "gyroZ", "yawRate"])
    if aim.accel is not None and aim.accel.size:
        available.add("accMag")
    if aim.special_cols.get("lat_acc"):
        available.add("latAcc")
    if aim.special_cols.get("yaw_rate"):
        available.add("yawRate")
    return available


def _available_signals_gopro(gopro: GoproIMU) -> set[str]:
    # Determine which signals can be derived from GoPro data.
    available = set()
    if gopro.gyro is not None and gopro.gyro.size:
        available.update(["gyroMag", "gyroZ", "yawRate"])
    if gopro.accel is not None and gopro.accel.size:
        available.add("accMag")
    return available


def _choose_signal(requested: str, aim: AimData, gopro: GoproIMU) -> tuple[str, Optional[str]]:
    # Pick the best available signal, with fallback and warning.
    available_aim = _available_signals_aim(aim)
    available_gopro = _available_signals_gopro(gopro)

    if requested in available_aim and requested in available_gopro:
        return requested, None

    for candidate in SIGNAL_PRIORITY:
        if candidate in available_aim and candidate in available_gopro:
            return candidate, f"Requested {requested} not available in both files. Using {candidate} instead."

    raise ValueError("No compatible signal found between AiM and GoPro data.")


def _signal_from_aim(aim: AimData, signal: str) -> np.ndarray:
    # Map a signal name to an AiM value series.
    if signal == "gyroMag":
        if aim.gyro is None:
            raise ValueError("AiM gyro channels not available.")
        if aim.gyro.shape[1] == 1:
            return np.abs(aim.gyro[:, 0])
        return np.linalg.norm(aim.gyro, axis=1)
    if signal == "accMag":
        if aim.accel is None:
            raise ValueError("AiM accel channels not available.")
        if aim.accel.shape[1] == 1:
            return np.abs(aim.accel[:, 0])
        return np.linalg.norm(aim.accel, axis=1)
    if signal == "gyroZ":
        if aim.gyro is None:
            raise ValueError("AiM gyro channels not available.")
        return aim.gyro[:, -1]
    if signal == "yawRate":
        col = aim.special_cols.get("yaw_rate")
        if col and col in aim.df.columns:
            return aim.df[col].to_numpy(dtype=float)
        if aim.gyro is None:
            raise ValueError("AiM yaw rate not available.")
        return aim.gyro[:, -1]
    if signal == "latAcc":
        col = aim.special_cols.get("lat_acc")
        if col and col in aim.df.columns:
            return aim.df[col].to_numpy(dtype=float)
        if aim.accel is None:
            raise ValueError("AiM lateral accel not available.")
        return aim.accel[:, -1]
    raise ValueError(f"Unsupported signal type: {signal}")


def _signal_from_gopro(gopro: GoproIMU, signal: str) -> Tuple[np.ndarray, np.ndarray]:
    # Map a signal name to a GoPro time series.
    if signal == "gyroMag":
        if gopro.gyro is None or gopro.gyro_time_s is None:
            raise ValueError("GoPro gyro not available.")
        return gopro.gyro_time_s, np.linalg.norm(gopro.gyro, axis=1)
    if signal == "accMag":
        if gopro.accel is None or gopro.accel_time_s is None:
            raise ValueError("GoPro accel not available.")
        return gopro.accel_time_s, np.linalg.norm(gopro.accel, axis=1)
    if signal in ("gyroZ", "yawRate"):
        if gopro.gyro is None or gopro.gyro_time_s is None:
            raise ValueError("GoPro gyro not available.")
        return gopro.gyro_time_s, gopro.gyro[:, -1]
    raise ValueError(f"Unsupported signal type for GoPro: {signal}")


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


def _clamp_start(start_s: float, aim_t: np.ndarray, gopro_t: np.ndarray, window_s: float) -> float:
    # Clamp start time to the shared overlap region.
    if start_s < 0:
        raise ValueError("Start time must be >= 0 seconds.")
    latest_start = min(aim_t[-1], gopro_t[-1]) - window_s
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


def _estimate_drift(start_s: np.ndarray, lag_s: np.ndarray) -> Optional[float]:
    # Fit a simple slope of lag vs. time to estimate drift.
    if start_s.size < 3:
        return None
    try:
        coeffs = np.polyfit(start_s, lag_s, 1)
        return float(coeffs[0])
    except Exception:
        return None


def _compute_window_candidates(
    aim_t_full: np.ndarray,
    aim_filt: np.ndarray,
    gopro_norm: np.ndarray,
    fs: float,
    window_s: float,
    step_s: float,
    max_lag_s: float,
    start_idx_min: int,
    gopro_duration: float,
    var_threshold_ratio: float = 0.2,
) -> tuple[list[dict], Optional[dict]]:
    win_n = int(round(window_s * fs))
    if win_n <= 1 or aim_filt.size < win_n:
        return [], None

    step_n = max(1, int(round(step_s * fs)))
    var_list = []
    starts = []
    start_idx_min = max(0, start_idx_min)
    # Slide a window across AiM data and score by variance (activity proxy).
    for idx in range(start_idx_min, aim_filt.size - win_n + 1, step_n):
        window = aim_filt[idx : idx + win_n]
        var_list.append(float(np.var(window)))
        starts.append(float(aim_t_full[idx]))

    if not starts:
        return [], None

    max_var = max(var_list) if var_list else 0.0
    threshold = max(1e-9, var_threshold_ratio * max_var)

    candidates: list[dict] = []
    best: Optional[dict] = None

    # For each active window, estimate a lag against the full GoPro signal.
    for idx, start_s in enumerate(starts):
        var_val = var_list[idx]
        if var_val < threshold:
            continue
        start_idx = int(np.searchsorted(aim_t_full, start_s, side="left"))
        aim_seg = aim_filt[start_idx : start_idx + win_n]
        if aim_seg.size < win_n:
            continue
        try:
            aim_norm = normalize_signal(aim_seg)
        except Exception:
            continue

        lag_local, peak, corr, lags = estimate_lag(aim_norm, gopro_norm, fs, max_lag_s)
        gopro_start = lag_local
        if gopro_start < 0:
            continue
        if gopro_start + window_s > gopro_duration:
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
                "aim_seg": aim_norm,
            }

    return candidates, best


def _compute_metrics(
    aim: AimData,
    gopro: GoproIMU,
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
    aim_signal = _signal_from_aim(aim, signal)
    gopro_time, gopro_signal = _signal_from_gopro(gopro, signal)

    aim_time, aim_signal = _drop_nan(aim.time_s, aim_signal)
    gopro_time, gopro_signal = _drop_nan(gopro_time, gopro_signal)

    # Estimate source sample rates for reporting only.
    aim_rate = infer_sample_rate(aim_time)
    gopro_rate = infer_sample_rate(gopro_time)

    # Resample both signals to a common uniform rate.
    aim_t_full, aim_y_full = resample_uniform(aim_time, aim_signal, fs)
    gopro_t_full, gopro_y_full = resample_uniform(gopro_time, gopro_signal, fs)

    # Filter and normalize to emphasize comparable motion.
    aim_filt = filter_signal(aim_y_full, fs, lowpass_hz, highpass_hz)
    gopro_filt = filter_signal(gopro_y_full, fs, lowpass_hz, highpass_hz)
    gopro_norm = normalize_signal(gopro_filt)

    win_n = int(round(window_s * fs))
    gopro_duration = float(gopro_t_full[-1]) if gopro_t_full.size else 0.0

    if not auto_window:
        # Single-window mode (manual or first active window).
        if start_override is None:
            start_s = select_active_window(aim_t_full, aim_y_full, window_s, fs)
        else:
            start_s = float(start_override)
        start_s = _clamp_start(start_s, aim_t_full, gopro_t_full, window_s)
        aim_t, aim_y = trim_window(aim_t_full, aim_filt, start_s, window_s, fs)
        aim_norm = normalize_signal(aim_y)

        lag_local, peak, corr, lags = estimate_lag(aim_norm, gopro_norm, fs, max_lag_s)
        lag_seconds = start_s - lag_local
        psr = peak_to_sidelobe(corr, lags, fs)

        stability_std = float("nan")
        gopro_start = lag_local
        if gopro_start >= 0 and gopro_start + window_s <= gopro_duration:
            _, gopro_aligned = trim_window(
                gopro_t_full, gopro_filt, gopro_start, window_s, fs
            )
            gopro_aligned = normalize_signal(gopro_aligned)
            stability_lag_s = min(30.0, max_lag_s)
            _, stability_std = lag_stability(aim_norm, gopro_aligned, fs, stability_lag_s)

        score = _score_metrics(peak, psr, stability_std)

        return {
            "signal": signal,
            "lag_seconds": lag_seconds,
            "peak": peak,
            "psr": psr,
            "stability": stability_std,
            "score": score,
            "aim_rate": aim_rate,
            "gopro_rate": gopro_rate,
            "corr": corr,
            "lags": lags,
            "aim_t": aim_t,
            "aim_y": aim_norm,
            "gopro_y": gopro_norm,
            "start_s": start_s,
            "window_count": 1,
            "drift_slope": None,
        }

    start_idx_min = 0
    if start_override is not None:
        start_idx_min = int(round(float(start_override) * fs))

    # Auto-window mode: scan windows and build a consensus lag.
    candidates, best = _compute_window_candidates(
        aim_t_full,
        aim_filt,
        gopro_norm,
        fs,
        window_s,
        window_step_s,
        max_lag_s,
        start_idx_min,
        gopro_duration,
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

    drift_slope = _estimate_drift(
        np.array([c["start_s"] for c in kept], dtype=float), lag_values
    )

    best_start = best["start_s"]
    start_idx = int(np.searchsorted(aim_t_full, best_start, side="left"))
    aim_seg = aim_filt[start_idx : start_idx + win_n]
    aim_seg = normalize_signal(aim_seg)
    gopro_start = best["lag_local"]
    if gopro_start >= 0 and gopro_start + window_s <= gopro_duration:
        _, gopro_seg = trim_window(
            gopro_t_full, gopro_filt, gopro_start, window_s, fs
        )
        gopro_seg = normalize_signal(gopro_seg)
    else:
        gopro_seg = gopro_norm[: win_n]

    return {
        "signal": signal,
        "lag_seconds": lag_seconds,
        "peak": peak,
        "psr": psr,
        "stability": stability_std,
        "score": score,
        "aim_rate": aim_rate,
        "gopro_rate": gopro_rate,
        "corr": best["corr"],
        "lags": best["lags"],
        "aim_t": aim_t_full[start_idx : start_idx + win_n],
        "aim_y": aim_seg,
        "gopro_y": gopro_seg,
        "start_s": best_start,
        "window_count": len(kept),
        "drift_slope": drift_slope,
    }


def _race_render_instruction(lag_seconds: float) -> str:
    def _format_hhmmss_ms(value: float) -> str:
        total_ms = int(round(abs(value) * 1000.0))
        hours = total_ms // 3600000
        rem = total_ms % 3600000
        minutes = rem // 60000
        rem = rem % 60000
        seconds = rem // 1000
        millis = rem % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    if lag_seconds > 0:
        return f"Video offset within project: {_format_hhmmss_ms(lag_seconds)}"
    if lag_seconds < 0:
        return f"Data offset within project: {_format_hhmmss_ms(lag_seconds)}"
    return "Video offset within project: 00:00:00.000"


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


def _describe_signal_source_aim(aim: AimData, signal: str) -> str:
    if signal == "gyroMag":
        if aim.gyro_cols:
            return ", ".join(aim.gyro_cols)
        return "n/a"
    if signal == "accMag":
        if aim.acc_cols:
            return ", ".join(aim.acc_cols)
        return "n/a"
    if signal == "gyroZ":
        if aim.gyro_cols:
            return aim.gyro_cols[-1]
        return "n/a"
    if signal == "yawRate":
        col = aim.special_cols.get("yaw_rate")
        if col and col in aim.df.columns:
            return col
        if aim.gyro_cols:
            return aim.gyro_cols[-1]
        return "n/a"
    if signal == "latAcc":
        col = aim.special_cols.get("lat_acc")
        if col and col in aim.df.columns:
            return col
        if aim.acc_cols:
            return aim.acc_cols[-1]
        return "n/a"
    return "n/a"


def _describe_signal_source_gopro(signal: str) -> str:
    if signal == "gyroMag":
        return "GYRO x/y/z"
    if signal == "accMag":
        return "ACCL x/y/z"
    if signal == "gyroZ":
        return "GYRO z"
    if signal == "yawRate":
        return "GYRO z"
    return "n/a"


def _format_signal_selection(
    aim: AimData,
    signal: str,
    aim_rate: float,
    gopro_rate: float,
) -> str:
    headers = ["source", "signal", "derived_from", "sample_rate"]
    rows = [
        [
            "GoPro",
            signal,
            _describe_signal_source_gopro(signal),
            _format_rate(gopro_rate),
        ],
        [
            "AiM",
            signal,
            _describe_signal_source_aim(aim, signal),
            _format_rate(aim_rate),
        ],
    ]
    return _format_columns(headers, rows)


def _save_plot(time_s: np.ndarray, aim: np.ndarray, gopro: np.ndarray, corr, lags, fs: float) -> None:
    # Optional diagnostics plot for visual inspection.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("Plotting requested but matplotlib is not installed.") from exc

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(time_s, aim, label="AiM", linewidth=1.0)
    axes[0].plot(time_s, gopro, label="GoPro", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Preprocessed Signals")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="best")

    axes[1].plot(lags / fs, corr, color="black", linewidth=1.0)
    axes[1].set_title("Cross-Correlation")
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Correlation")

    axes[2].plot(time_s, aim - gopro, color="tab:orange", linewidth=1.0)
    axes[2].set_title("Signal Difference (AiM - GoPro)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")

    fig.tight_layout()
    fig.savefig("sync_plot.png", dpi=150)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sync AiM CSV logs to GoPro MP4 video using IMU cross-correlation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mp4", help="Path to GoPro MP4 file")
    parser.add_argument("--csv", help="Path to AiM CSV file")
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
    parser.add_argument("--fs", type=float, default=50.0, help="Resample rate (Hz)")
    parser.add_argument("--lowpass-hz", type=float, default=8.0, help="Low-pass cutoff (Hz)")
    parser.add_argument("--highpass-hz", type=float, default=0.2, help="High-pass cutoff (Hz)")
    parser.add_argument("--aim-time-col", help="Override AiM time column name")
    parser.add_argument("--aim-gyro-cols", help="Comma-separated AiM gyro column names")
    parser.add_argument("--aim-acc-cols", help="Comma-separated AiM accel column names")
    parser.add_argument("--write-gopro-csv", action="store_true", help="Write gopro_imu.csv")
    parser.add_argument("--write-shifted-csv", action="store_true", help="Write aim_shifted.csv")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plot to sync_plot.png")
    parser.set_defaults(auto_window=True)

    args = parser.parse_args(argv)

    try:
        _status("Resolving input files...")
        mp4_path, csv_path = _resolve_paths(args.mp4, args.csv)
        _status(f"Loading AiM CSV: {csv_path.name}")
        aim = load_aim_csv(
            csv_path,
            time_col=args.aim_time_col,
            gyro_cols=_parse_cols(args.aim_gyro_cols),
            acc_cols=_parse_cols(args.aim_acc_cols),
        )
        _status(f"Extracting GoPro IMU: {mp4_path.name} (this can take a while)")
        gopro = extract_gopro_imu(mp4_path)

        available = sorted(
            _available_signals_aim(aim) & _available_signals_gopro(gopro),
            key=lambda s: SIGNAL_PRIORITY.index(s) if s in SIGNAL_PRIORITY else 99,
        )
        if not available:
            raise ValueError("No compatible signals found between AiM and GoPro data.")

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
            signal, warning = _choose_signal(args.signal, aim, gopro)
            if warning:
                print(f"Warning: {warning}")
            selected_signals = [signal]

        best = None
        metrics_all = []
        for sig in selected_signals:
            _status(f"Computing correlation metrics for signal: {sig}")
            metrics = _compute_metrics(
                aim=aim,
                gopro=gopro,
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

        aim_rate = best["aim_rate"]
        gopro_rate = best["gopro_rate"]

        print("")
        print("Signal Selection")
        print(_color_line())
        print(_format_signal_selection(aim, best["signal"], aim_rate, gopro_rate))
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
        drift_slope = best.get("drift_slope")
        corr = best["corr"]
        lags = best["lags"]

        print("")
        print("Sync Summary")
        print(_color_line())
        summary_rows = [
            ("Lag (seconds)", f"{lag_seconds:+.3f}"),
            ("Correlation peak", f"{peak:.3f}"),
            ("Peak-to-sidelobe ratio", f"{psr:.3f}"),
            ("Stability (stddev s)", f"{stability_std:.3f}"),
        ]
        if drift_slope is not None and np.isfinite(drift_slope):
            summary_rows.append(("Drift (s/s)", f"{drift_slope:+.6f}"))
        print(_format_kv(summary_rows))
        print("")
        print("RaceRender Offset")
        print(_color_line())
        print(_race_render_instruction(lag_seconds))
        print("")

        if peak < 0.2:
            print("Warning: Low correlation peak; alignment may be unreliable.")
        if np.isfinite(stability_std) and stability_std > 0.2:
            print("Warning: High lag variability across subwindows; alignment may be unstable.")

        if args.write_gopro_csv:
            write_gopro_csv(
                Path("gopro_imu.csv"),
                gopro.gyro_time_s,
                gopro.gyro,
                gopro.accel_time_s,
                gopro.accel,
            )
            print("Wrote gopro_imu.csv")

        if args.write_shifted_csv:
            write_shifted_csv(Path("aim_shifted.csv"), aim.df, aim.time_col, aim.time_s, lag_seconds)
            print("Wrote aim_shifted.csv")

        if args.plot:
            time_rel = best["aim_t"] - float(best["aim_t"][0])
            _save_plot(time_rel, best["aim_y"], best["gopro_y"], corr, lags, args.fs)
            print("Wrote sync_plot.png")

    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
