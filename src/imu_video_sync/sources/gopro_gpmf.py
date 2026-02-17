from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import importlib.util
import os
import sys
import types
import tempfile

import numpy as np

from ..core.models import ImuBundle, SourceMeta, TimeSeries
from .base import VideoSource
from .registry import register_source


@dataclass
class _Block:
    stmp: Optional[float]
    data: np.ndarray


def _ensure_nx3(arr: np.ndarray) -> np.ndarray:
    # Normalize shape so we always have (N, 3) samples.
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        return arr.T
    return arr


def _axis_names(values: np.ndarray) -> list[str]:
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


def _as_scalar(value) -> Optional[float]:
    # Convert scalars from numpy or Python types to float.
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        if value.size > 0:
            return float(value.flat[0])
    try:
        return float(value)
    except Exception:
        return None


def _apply_scale(values: np.ndarray, scal) -> np.ndarray:
    # Apply GPMF SCAL scaling if present.
    if scal is None:
        return values.astype(float)
    scale = np.asarray(scal, dtype=float)
    if scale.ndim == 0:
        return values.astype(float) / float(scale)
    if scale.size == 3:
        return values.astype(float) / scale.reshape(1, 3)
    return values.astype(float) / scale


def _choose_scale(step_raw: float, expected_hz: float) -> float:
    # GPMF timestamps can be in different units; choose a scale that matches expected rate.
    if not np.isfinite(step_raw) or step_raw <= 0:
        return 1.0
    candidates = [1.0, 1e3, 1e6]
    best_scale = 1.0
    best_score = float("inf")
    for scale in candidates:
        rate = scale / step_raw
        in_range = 20.0 <= rate <= 2000.0
        score = abs(rate - expected_hz)
        if not in_range:
            score += 1e6
        if score < best_score:
            best_score = score
            best_scale = scale
    return best_scale


def _build_time_series(blocks: List[_Block], expected_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    if not blocks:
        raise ValueError("No IMU blocks found.")

    blocks = [b for b in blocks if b.data.size]
    if not blocks:
        raise ValueError("No IMU samples found in blocks.")

    blocks_with_data = blocks
    blocks = [b for b in blocks if b.stmp is not None]
    if not blocks:
        # If no timestamps exist, fall back to uniform sampling.
        values = np.vstack([b.data for b in blocks_with_data])
        time_s = np.arange(values.shape[0], dtype=float) / float(expected_hz)
        return time_s, values

    # Sort blocks by timestamp before stitching.
    blocks.sort(key=lambda b: b.stmp)

    step_raws = []
    for idx in range(len(blocks) - 1):
        dt = blocks[idx + 1].stmp - blocks[idx].stmp
        n = blocks[idx].data.shape[0]
        if dt > 0 and n > 0:
            step_raws.append(dt / n)

    if step_raws:
        step_raw_median = float(np.median(step_raws))
        scale = _choose_scale(step_raw_median, expected_hz)
        step_s_default = step_raw_median / scale
    else:
        scale = 1.0
        step_s_default = 1.0 / float(expected_hz)

    times = []
    values = []
    for idx, block in enumerate(blocks):
        n = block.data.shape[0]
        if idx < len(blocks) - 1:
            dt = blocks[idx + 1].stmp - block.stmp
            if dt > 0:
                step_raw = dt / n
            else:
                step_raw = step_s_default * scale
        else:
            step_raw = step_s_default * scale

        step_s = step_raw / scale
        t0 = block.stmp / scale
        times.append(t0 + np.arange(n, dtype=float) * step_s)
        values.append(block.data)

    time_s = np.concatenate(times)
    values = np.vstack(values)
    time_s = time_s - float(time_s[0])
    return time_s, values


def _iter_strm_blocks(gpmf_bytes, gp):
    # Iterate device/stream blocks and yield parsed STRM payloads.
    for item in gp.iter_klv(gpmf_bytes):
        if item.key != "DEVC":
            continue
        if isinstance(item.value, types.GeneratorType):
            for sub in item.value:
                if sub.key != "STRM":
                    continue
                if isinstance(sub.value, types.GeneratorType):
                    block = list(sub.value)
                else:
                    block = list(gp.iter_klv(sub.value))
                yield block


def extract_gopro_imu(path: str) -> ImuBundle:
    # Ensure temp/cache directories are writable for pygpmf-oz and ffmpeg.
    tmp_root = Path(tempfile.gettempdir()) / "imu_video_sync"
    joblib_dir = tmp_root / "joblib"
    tmp_dir = tmp_root / "tmp"
    joblib_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_dir))
    os.environ.setdefault("JOBLIB_CACHE_DIR", str(joblib_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMPDIR", str(tmp_dir))

    # Locate the gpmf module on sys.path without importing its __init__.
    gpmf_root = None
    for entry in sys.path:
        candidate = Path(entry) / "gpmf"
        if (candidate / "io.py").exists() and (candidate / "parse.py").exists():
            gpmf_root = candidate
            break
    if gpmf_root is None:
        raise ImportError("pygpmf-oz (gpmf) is not installed or not found on sys.path.")

    def _load_module(name: str, file_path: Path):
        # Load a module directly from file to avoid import side effects.
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module {name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    gpmf_io = _load_module("gpmf.io", gpmf_root / "io.py")
    gpmf_parse = _load_module("gpmf.parse", gpmf_root / "parse.py")

    # Extract the raw GPMF stream from the MP4.
    gpmf_bytes = gpmf_io.extract_gpmf_stream(str(path))

    gyro_blocks: List[_Block] = []
    accel_blocks: List[_Block] = []

    # Parse each STRM block and collect gyro/accel data.
    for block in _iter_strm_blocks(gpmf_bytes, gpmf_parse):
        keys = {item.key: item for item in block}
        scal = keys.get("SCAL")
        stmp = _as_scalar(keys.get("STMP").value) if "STMP" in keys else None

        if "GYRO" in keys:
            values = _ensure_nx3(keys["GYRO"].value)
            values = _apply_scale(values, scal.value if scal else None)
            gyro_blocks.append(_Block(stmp=stmp, data=values))

        if "ACCL" in keys:
            values = _ensure_nx3(keys["ACCL"].value)
            values = _apply_scale(values, scal.value if scal else None)
            accel_blocks.append(_Block(stmp=stmp, data=values))

    gyro_series = accel_series = None

    if gyro_blocks:
        gyro_time_s, gyro = _build_time_series(gyro_blocks, expected_hz=400.0)
        gyro_series = TimeSeries(
            time_s=gyro_time_s,
            values=gyro,
            axes=_axis_names(gyro),
            name="gyro",
        )

    if accel_blocks:
        accel_time_s, accel = _build_time_series(accel_blocks, expected_hz=200.0)
        accel_series = TimeSeries(
            time_s=accel_time_s,
            values=accel,
            axes=_axis_names(accel),
            name="accel",
        )

    if gyro_series is None and accel_series is None:
        raise ValueError("No GYRO/ACCL data found in GoPro telemetry.")

    meta = SourceMeta(name="gopro_gpmf", kind="video", path=Path(path))

    return ImuBundle(
        gyro=gyro_series,
        accel=accel_series,
        channels={},
        meta=meta,
    )


@register_source("video")
class GoproGpmfSource(VideoSource):
    name = "gopro_gpmf"

    @classmethod
    def sniff(cls, path: Path) -> float:
        if path.suffix.lower() in {".mp4", ".mov"}:
            return 0.6
        return 0.0

    @classmethod
    def load(cls, path: Path, **opts) -> ImuBundle:
        return extract_gopro_imu(str(path))
