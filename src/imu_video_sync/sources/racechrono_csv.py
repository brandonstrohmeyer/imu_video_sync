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
    "time (s)",
    "time(s)",
    "time",
    "elapsed time (s)",
    "elapsed time",
]

LAT_ACC_HINTS = ["lateral acceleration", "lat acceleration", "lat acc"]
LONG_ACC_HINTS = ["longitudinal acceleration", "long acceleration", "long acc"]

GYRO_AXIS_RE = re.compile(r"^\s*([xyz])\s*rate of rotation", re.IGNORECASE)
ACC_AXIS_RE = re.compile(r"^\s*([xyz])\s*acceleration", re.IGNORECASE)


@dataclass
class RaceChronoColumns:
    time_col: str
    gyro_cols: List[str]
    acc_cols: List[str]
    special_cols: Dict[str, Optional[str]]


def detect_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        if sample.count(";") > sample.count(","):
            return ";"
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def detect_header_row(path: Path, delimiter: str, max_lines: int = 120) -> int:
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
            norm_fields = [normalize_col(f) for f in fields]
            score = 0
            if any("time" in f for f in norm_fields):
                score += 5
            if any("gyro" in f for f in norm_fields):
                score += 3
            if any("acc" in f for f in norm_fields):
                score += 2
            score += min(len(fields), 40)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx


def _find_time_col(columns: Iterable[str]) -> Optional[str]:
    columns = list(columns)
    norm_map = {normalize_col(col): col for col in columns}
    for candidate in TIME_CANDIDATES:
        norm = normalize_col(candidate)
        if norm in norm_map:
            return norm_map[norm]
    for col in columns:
        if "time" in normalize_col(col):
            return col
    return None


def _find_axis_cols(columns: Iterable[str], axis_re: re.Pattern, hint: str) -> List[str]:
    cols: Dict[str, str] = {}
    for col in columns:
        text = col.lower()
        if hint not in text:
            continue
        match = axis_re.match(col)
        if not match:
            continue
        axis = match.group(1).lower()
        cols[axis] = col
    ordered = [cols[a] for a in ("x", "y", "z") if a in cols]
    return ordered


def _find_first(columns: Iterable[str], hints: Iterable[str]) -> Optional[str]:
    columns = list(columns)
    for col in columns:
        norm = col.lower()
        for hint in hints:
            if hint in norm:
                return col
    return None


def detect_columns(columns: Iterable[str]) -> RaceChronoColumns:
    columns = list(columns)
    time_col = _find_time_col(columns)
    if time_col is None:
        raise ValueError("Unable to detect time column in RaceChrono CSV.")

    gyro_cols = _find_axis_cols(columns, GYRO_AXIS_RE, "gyro")
    acc_cols = _find_axis_cols(columns, ACC_AXIS_RE, "acc")

    special_cols: Dict[str, Optional[str]] = {
        "lat_acc": _find_first(columns, LAT_ACC_HINTS),
        "long_acc": _find_first(columns, LONG_ACC_HINTS),
    }

    return RaceChronoColumns(
        time_col=time_col,
        gyro_cols=gyro_cols,
        acc_cols=acc_cols,
        special_cols=special_cols,
    )


def _parse_time_column(series: pd.Series) -> np.ndarray:
    time_s = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    finite = np.isfinite(time_s)
    if not finite.any():
        return time_s

    diffs = np.diff(time_s[finite])
    diffs = diffs[diffs > 0]
    med_dt = float(np.nanmedian(diffs)) if diffs.size else 0.0
    max_t = float(np.nanmax(time_s[finite]))

    # RaceChrono typically stores time in seconds (often epoch seconds).
    if max_t > 1e8 and med_dt < 2.0:
        return time_s
    if max_t > 1e5 or med_dt > 10.0:
        return time_s / 1000.0
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


def load_racechrono_csv(
    path: Path,
    time_col: Optional[str] = None,
    gyro_cols: Optional[List[str]] = None,
    acc_cols: Optional[List[str]] = None,
) -> LogData:
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

    time_s = _parse_time_column(df[time_col])

    gyro = _series_from_columns(time_s, df, gyro_cols, "gyro")
    accel = _series_from_columns(time_s, df, acc_cols, "accel")

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
            units="deg/s",
        )
    if accel is not None:
        accel = TimeSeries(
            time_s=time_s,
            values=accel.values[valid],
            axes=accel.axes,
            name=accel.name,
            units="g",
        )

    if time_s.size == 0:
        raise ValueError("No valid time samples found in RaceChrono CSV after cleaning.")

    time_s = time_s - float(time_s[0])
    if gyro is not None:
        gyro = TimeSeries(
            time_s=time_s,
            values=gyro.values,
            axes=gyro.axes,
            name=gyro.name,
            units=gyro.units,
        )
    if accel is not None:
        accel = TimeSeries(
            time_s=time_s,
            values=accel.values,
            axes=accel.axes,
            name=accel.name,
            units=accel.units,
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
                units="g",
            )

    imu = ImuBundle(
        gyro=gyro,
        accel=accel,
        channels=channels,
        meta=SourceMeta(name="racechrono_csv", kind="log", path=Path(path)),
    )

    return LogData(
        imu=imu,
        df=df,
        time_col=time_col,
        time_s=time_s,
    )


@register_source("log")
class RaceChronoCsvSource(LogSource):
    name = "racechrono_csv"

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
            columns = [c.lower() for c in df.columns]
            if any("racechrono" in c for c in columns):
                return 0.9
            if any("rate of rotation" in c for c in columns) and any("gyro" in c for c in columns):
                return 0.85
            if any("lateral acceleration" in c for c in columns):
                return 0.7
            _ = detect_columns(df.columns)
            return 0.6
        except Exception:
            return 0.2

    @classmethod
    def load(cls, path: Path, **opts) -> LogData:
        return load_racechrono_csv(
            path,
            time_col=opts.get("time_col"),
            gyro_cols=opts.get("gyro_cols"),
            acc_cols=opts.get("acc_cols"),
        )
