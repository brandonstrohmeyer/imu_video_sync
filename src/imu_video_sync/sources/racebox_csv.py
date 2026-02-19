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
    "time_s",
    "times",
    "t",
]

GYRO_AXIS_CANDIDATES = {
    "x": ["gyrox", "gyrx", "gx", "gyro_x"],
    "y": ["gyroy", "gyry", "gy", "gyro_y"],
    "z": ["gyroz", "gyrz", "gz", "gyro_z"],
}

ACC_AXIS_CANDIDATES = {
    "x": ["accx", "accelx", "ax", "acc_x"],
    "y": ["accy", "accely", "ay", "acc_y"],
    "z": ["accz", "accelz", "az", "acc_z"],
}


@dataclass
class RaceBoxColumns:
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


def _find_first(norm_map: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in norm_map:
            return norm_map[candidate]
    return None


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
                score += 3
        for candidate in ACC_AXIS_CANDIDATES[axis]:
            if candidate in norm_fields:
                score += 2
    if "kph" in norm_fields or "speed" in norm_fields:
        score += 2
    if "latitude" in norm_fields and "longitude" in norm_fields:
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
            score = _score_header(fields)
            if score > best_score:
                best_score = score
                best_idx = idx
    return best_idx


def detect_columns(columns: Iterable[str]) -> RaceBoxColumns:
    columns = list(columns)
    norm_map = {normalize_col(col): col for col in columns}

    time_col = _find_first(norm_map, TIME_CANDIDATES)
    if time_col is None:
        time_col = next((col for col in columns if "time" in normalize_col(col)), None)
    if time_col is None:
        raise ValueError("Unable to detect time column in RaceBox CSV.")

    gyro_cols: List[str] = []
    for axis in ("x", "y", "z"):
        col = _find_first(norm_map, GYRO_AXIS_CANDIDATES[axis])
        if col is not None:
            gyro_cols.append(col)

    acc_cols: List[str] = []
    for axis in ("x", "y", "z"):
        col = _find_first(norm_map, ACC_AXIS_CANDIDATES[axis])
        if col is not None:
            acc_cols.append(col)

    if not acc_cols:
        # RaceBox exports often use bare X/Y/Z for accel axes.
        for axis in ("x", "y", "z"):
            col = norm_map.get(axis)
            if col is not None:
                acc_cols.append(col)

    special_cols: Dict[str, Optional[str]] = {
        "lat_acc": None,
        "long_acc": None,
    }

    return RaceBoxColumns(
        time_col=time_col,
        gyro_cols=gyro_cols,
        acc_cols=acc_cols,
        special_cols=special_cols,
    )


def _parse_time_column(series: pd.Series) -> np.ndarray:
    time_s = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
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


def load_racebox_csv(
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
        raise ValueError("No valid time samples found in RaceBox CSV after cleaning.")

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
        meta=SourceMeta(name="racebox_csv", kind="log", path=Path(path)),
    )

    return LogData(
        imu=imu,
        df=df,
        time_col=time_col,
        time_s=time_s,
    )


@register_source("log")
class RaceBoxCsvSource(LogSource):
    name = "racebox_csv"

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
            columns = [normalize_col(c) for c in df.columns]
            if "racebox" in "".join(columns):
                return 0.9
            if "gyrox" in columns and "gyroy" in columns and "gyroz" in columns:
                return 0.85
            if "kph" in columns and "latitude" in columns:
                return 0.8
            _ = detect_columns(df.columns)
            return 0.6
        except Exception:
            return 0.2

    @classmethod
    def load(cls, path: Path, **opts) -> LogData:
        return load_racebox_csv(
            path,
            time_col=opts.get("time_col"),
            gyro_cols=opts.get("gyro_cols"),
            acc_cols=opts.get("acc_cols"),
        )
