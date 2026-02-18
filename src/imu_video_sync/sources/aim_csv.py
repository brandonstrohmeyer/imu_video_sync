from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import csv
import re

import numpy as np
import pandas as pd

from ..core.models import ImuBundle, LogData, SourceMeta, TimeSeries
from .base import LogSource
from .registry import register_source


TIME_CANDIDATES = [
    "time",
    "timestamp",
    "logtime",
    "logtime",
    "sessiontime",
    "sessiontime",
    "laptime",
    "time_s",
    "times",
    "t",
]

GYRO_AXIS_CANDIDATES = {
    "x": ["gyrox", "gyrx", "gx", "rollrate", "roll"],
    "y": ["gyroy", "gyry", "gy", "pitchrate", "pitch"],
    "z": ["gyroz", "gyrz", "gz"],
}

ACC_AXIS_CANDIDATES = {
    "x": [
        "accx",
        "accelx",
        "ax",
        "longacc",
        "longaccel",
        "longitudinalacc",
        "inlineacc",
        "inlineaccel",
    ],
    "y": ["accy", "accely", "ay", "latacc", "lataccel", "lateralacc"],
    "z": ["accz", "accelz", "az", "vertacc", "verticalacc"],
}

YAW_RATE_CANDIDATES = ["yawrate", "yaw_rate", "yaw"]
LAT_ACC_CANDIDATES = ["latacc", "lat_acc", "lataccel", "lateralacc"]
LONG_ACC_CANDIDATES = ["longacc", "long_acc", "longaccel", "longitudinalacc"]


@dataclass
class AimColumns:
    time_col: str
    gyro_cols: List[str]
    acc_cols: List[str]
    special_cols: Dict[str, Optional[str]]


def detect_delimiter(path: Path) -> str:
    # Read a small sample and let csv.Sniffer guess the delimiter.
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        # Fallback heuristic if Sniffer fails.
        if sample.count(";") > sample.count(","):
            return ";"
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def normalize_col(name: str) -> str:
    # Remove punctuation and lower-case to make matching robust.
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_first(norm_map: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in norm_map:
            return norm_map[candidate]
    return None


def detect_columns(columns: Iterable[str]) -> AimColumns:
    columns = list(columns)
    norm_map = {normalize_col(col): col for col in columns}

    # Pick the most likely time column.
    time_col = _find_first(norm_map, TIME_CANDIDATES)
    if time_col is None:
        time_col = next((col for col in columns if "time" in normalize_col(col)), None)
    if time_col is None:
        raise ValueError("Unable to detect time column in log CSV.")

    # Optional special channels (yaw rate, lateral accel, etc.).
    special_cols: Dict[str, Optional[str]] = {
        "yaw_rate": _find_first(norm_map, YAW_RATE_CANDIDATES),
        "lat_acc": _find_first(norm_map, LAT_ACC_CANDIDATES),
        "long_acc": _find_first(norm_map, LONG_ACC_CANDIDATES),
    }

    # Try to detect gyro axes by name.
    gyro_cols: List[str] = []
    for axis in ("x", "y", "z"):
        col = _find_first(norm_map, GYRO_AXIS_CANDIDATES[axis])
        if col is not None:
            gyro_cols.append(col)

    if special_cols["yaw_rate"] and special_cols["yaw_rate"] not in gyro_cols:
        if len(gyro_cols) < 3:
            gyro_cols.append(special_cols["yaw_rate"])

    # Try to detect accel axes by name.
    acc_cols: List[str] = []
    for axis in ("x", "y", "z"):
        col = _find_first(norm_map, ACC_AXIS_CANDIDATES[axis])
        if col is not None:
            acc_cols.append(col)

    if len(acc_cols) == 0:
        if special_cols["long_acc"]:
            acc_cols.append(special_cols["long_acc"])
        if special_cols["lat_acc"]:
            acc_cols.append(special_cols["lat_acc"])

    return AimColumns(
        time_col=time_col,
        gyro_cols=gyro_cols,
        acc_cols=acc_cols,
        special_cols=special_cols,
    )


def _score_header(fields: List[str]) -> int:
    if len(fields) < 2:
        return 0
    norm_fields = [normalize_col(f) for f in fields]
    score = 0
    for candidate in TIME_CANDIDATES:
        if candidate in norm_fields:
            score += 5
    for axis in ("x", "y", "z"):
        for candidate in GYRO_AXIS_CANDIDATES[axis]:
            if candidate in norm_fields:
                score += 2
        for candidate in ACC_AXIS_CANDIDATES[axis]:
            if candidate in norm_fields:
                score += 2
    for candidate in YAW_RATE_CANDIDATES + LAT_ACC_CANDIDATES + LONG_ACC_CANDIDATES:
        if candidate in norm_fields:
            score += 2
    score += min(len(fields), 30)
    return score


def detect_header_row(path: Path, delimiter: str, max_lines: int = 80) -> int:
    best_idx = 0
    best_score = -1
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            if not line.strip():
                continue
            fields = [f.strip() for f in line.strip().split(delimiter)]
            if len(fields) < 3:
                continue
            # Score each line based on expected column names.
            score = _score_header(fields)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx


def _parse_time_column(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        time_s = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    else:
        try:
            time_s = pd.to_timedelta(series).dt.total_seconds().to_numpy(dtype=float)
        except Exception:
            time_s = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()

    # Heuristic: convert ms to s if values look too large.
    finite = np.isfinite(time_s)
    if finite.any():
        diffs = np.diff(time_s[finite])
        diffs = diffs[diffs > 0]
        med_dt = float(np.nanmedian(diffs)) if diffs.size else 0.0
        max_t = float(np.nanmax(time_s[finite]))
        if max_t > 1e5 or med_dt > 10.0:
            time_s = time_s / 1000.0
    return time_s


def _series_from_columns(
    time_s: np.ndarray, df: pd.DataFrame, cols: List[str], name: str
) -> Optional[TimeSeries]:
    if not cols:
        return None
    values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return TimeSeries(
        time_s=time_s,
        values=values,
        axes=list(cols),
        name=name,
    )


def load_aim_csv(
    path: Path,
    time_col: Optional[str] = None,
    gyro_cols: Optional[List[str]] = None,
    acc_cols: Optional[List[str]] = None,
) -> LogData:
    # Detect delimiter and header row before parsing.
    delimiter = detect_delimiter(path)
    header_idx = detect_header_row(path, delimiter)
    df = pd.read_csv(
        path,
        sep=delimiter,
        header=0,
        skiprows=header_idx,
        engine="python",
        on_bad_lines="skip",
    )

    detected = detect_columns(df.columns)

    time_col = time_col or detected.time_col
    gyro_cols = gyro_cols or detected.gyro_cols
    acc_cols = acc_cols or detected.acc_cols

    if time_col not in df.columns:
        raise ValueError(f"Specified time column not found: {time_col}")

    for col in gyro_cols:
        if col not in df.columns:
            raise ValueError(f"Specified gyro column not found: {col}")

    for col in acc_cols:
        if col not in df.columns:
            raise ValueError(f"Specified accel column not found: {col}")

    # Convert time column to seconds-from-start.
    time_s = _parse_time_column(df[time_col])

    gyro = _series_from_columns(time_s, df, gyro_cols, "gyro")
    accel = _series_from_columns(time_s, df, acc_cols, "accel")

    # Drop rows with NaNs in time or selected channels.
    valid = np.isfinite(time_s)
    if gyro is not None:
        valid &= np.all(np.isfinite(gyro.values), axis=1)
    if accel is not None:
        valid &= np.all(np.isfinite(accel.values), axis=1)

    df = df.loc[valid].reset_index(drop=True)
    time_s = time_s[valid]

    if gyro is not None:
        gyro = TimeSeries(
            time_s=time_s,
            values=gyro.values[valid],
            axes=gyro.axes,
            name=gyro.name,
        )
    if accel is not None:
        accel = TimeSeries(
            time_s=time_s,
            values=accel.values[valid],
            axes=accel.axes,
            name=accel.name,
        )

    if time_s.size == 0:
        raise ValueError("No valid time samples found in log CSV after cleaning.")

    # Normalize time to start at 0.0 seconds.
    time_s = time_s - float(time_s[0])
    if gyro is not None:
        gyro = TimeSeries(
            time_s=time_s,
            values=gyro.values,
            axes=gyro.axes,
            name=gyro.name,
        )
    if accel is not None:
        accel = TimeSeries(
            time_s=time_s,
            values=accel.values,
            axes=accel.axes,
            name=accel.name,
        )

    channels: Dict[str, TimeSeries] = {}
    for key, col in detected.special_cols.items():
        if col and col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            channels[key] = TimeSeries(
                time_s=time_s,
                values=values,
                axes=[col],
                name=key,
            )

    imu = ImuBundle(
        gyro=gyro,
        accel=accel,
        channels=channels,
        meta=SourceMeta(name="aim_csv", kind="log", path=Path(path)),
    )

    return LogData(
        imu=imu,
        df=df,
        time_col=time_col,
        time_s=time_s,
    )


@register_source("log")
class AimCsvSource(LogSource):
    name = "aim_csv"

    @classmethod
    def sniff(cls, path: Path) -> float:
        if path.suffix.lower() != ".csv":
            return 0.0
        try:
            delimiter = detect_delimiter(path)
            header_idx = detect_header_row(path, delimiter)
            df = pd.read_csv(
                path,
                sep=delimiter,
                header=0,
                skiprows=header_idx,
                engine="python",
                on_bad_lines="skip",
                nrows=1,
            )
            _ = detect_columns(df.columns)
            return 0.8
        except Exception:
            return 0.2

    @classmethod
    def load(cls, path: Path, **opts) -> LogData:
        return load_aim_csv(
            path,
            time_col=opts.get("time_col"),
            gyro_cols=opts.get("gyro_cols"),
            acc_cols=opts.get("acc_cols"),
        )
