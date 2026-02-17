from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import shutil
import struct
import subprocess

import numpy as np

from ..core.models import ImuBundle, SourceMeta, TimeSeries
from .base import VideoSource
from .registry import register_source


_ACCEL_PATH = (3, 2, 10)
_ACCEL_FIELDS = {2: "x", 3: "y", 4: "z"}


@dataclass(frozen=True)
class _PacketInfo:
    pts_time: float
    size: int


def _require_tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return exe


def _run_json(args: List[str]) -> dict:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"Command failed: {' '.join(args)} ({msg})")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse ffprobe JSON output.") from exc


def _ffprobe_streams(path: Path) -> List[dict]:
    _require_tool("ffprobe")
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ]
    )
    return data.get("streams", [])


def _find_dji_meta_stream(path: Path) -> int:
    streams = _ffprobe_streams(path)
    for stream in streams:
        if stream.get("codec_tag_string") == "djmd":
            return int(stream["index"])
        handler = (stream.get("tags") or {}).get("handler_name", "")
        if "dji meta" in handler.lower():
            return int(stream["index"])
    raise ValueError("DJI metadata stream (djmd) not found in video file.")


def _ffprobe_packets(path: Path, stream_index: int) -> List[_PacketInfo]:
    _require_tool("ffprobe")
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            f"{stream_index}",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,size",
            "-print_format",
            "json",
            str(path),
        ]
    )
    packets: List[_PacketInfo] = []
    for pkt in data.get("packets", []):
        pts_raw = pkt.get("pts_time")
        size_raw = pkt.get("size")
        if pts_raw is None or size_raw is None:
            continue
        try:
            pts = float(pts_raw)
            size = int(size_raw)
        except Exception:
            continue
        packets.append(_PacketInfo(pts_time=pts, size=size))
    return packets


def _extract_stream_data(path: Path, stream_index: int, out_path: Path) -> None:
    _require_tool("ffmpeg")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-c",
        "copy",
        "-f",
        "data",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed extracting DJI metadata ({msg})")


def _read_varint(data: bytes, idx: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while idx < len(data):
        byte = data[idx]
        idx += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, idx
        shift += 7
        if shift > 64:
            break
    raise ValueError("Invalid varint encoding.")


def _iter_fields(data: bytes) -> Iterable[Tuple[int, int, bytes]]:
    idx = 0
    length = len(data)
    while idx < length:
        key, idx = _read_varint(data, idx)
        field_num = key >> 3
        wire_type = key & 0x7

        if wire_type == 0:
            _, idx = _read_varint(data, idx)
            continue
        if wire_type == 1:
            if idx + 8 > length:
                break
            raw = data[idx : idx + 8]
            idx += 8
            yield field_num, wire_type, raw
            continue
        if wire_type == 2:
            size, idx = _read_varint(data, idx)
            if idx + size > length:
                break
            raw = data[idx : idx + size]
            idx += size
            yield field_num, wire_type, raw
            continue
        if wire_type == 5:
            if idx + 4 > length:
                break
            raw = data[idx : idx + 4]
            idx += 4
            yield field_num, wire_type, raw
            continue
        # Unsupported or deprecated wire types: skip safely.
        break


def _extract_fields_by_path(
    data: bytes, path: Tuple[int, ...], target_fields: Dict[int, str]
) -> Dict[str, float]:
    if not path:
        found: Dict[str, float] = {}
        for field_num, wire_type, raw in _iter_fields(data):
            if field_num not in target_fields:
                continue
            name = target_fields[field_num]
            if wire_type == 5 and len(raw) == 4:
                found[name] = struct.unpack("<f", raw)[0]
            elif wire_type == 1 and len(raw) == 8:
                found[name] = struct.unpack("<d", raw)[0]
            elif wire_type == 2 and len(raw) in (4, 8):
                if len(raw) == 4:
                    found[name] = struct.unpack("<f", raw)[0]
                else:
                    found[name] = struct.unpack("<d", raw)[0]
        return found

    results: Dict[str, float] = {}
    for field_num, wire_type, raw in _iter_fields(data):
        if field_num != path[0] or wire_type != 2:
            continue
        nested = _extract_fields_by_path(raw, path[1:], target_fields)
        if nested:
            results.update(nested)
    return results


def _extract_accel_from_message(data: bytes) -> Optional[np.ndarray]:
    fields = _extract_fields_by_path(data, _ACCEL_PATH, _ACCEL_FIELDS)
    if not fields:
        return None
    if not all(axis in fields for axis in ("x", "y", "z")):
        return None
    return np.array([fields["x"], fields["y"], fields["z"]], dtype=float)


def extract_dji_accel(path: Path) -> ImuBundle:
    stream_index = _find_dji_meta_stream(path)
    packets = _ffprobe_packets(path, stream_index)
    if not packets:
        raise ValueError("No DJI metadata packets found in video.")

    tmp_root = Path.cwd() / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_root / "dji_meta.bin"
    _extract_stream_data(path, stream_index, tmp_path)
    blob = tmp_path.read_bytes()
    try:
        tmp_path.unlink()
    except Exception:
        pass

    times: List[float] = []
    values: List[np.ndarray] = []
    offset = 0

    for pkt in packets:
        if pkt.size <= 0:
            continue
        if offset + pkt.size > len(blob):
            break
        chunk = blob[offset : offset + pkt.size]
        offset += pkt.size
        if not chunk:
            continue
        accel = _extract_accel_from_message(chunk)
        if accel is None:
            continue
        times.append(pkt.pts_time)
        values.append(accel)

    if not times or not values:
        raise ValueError("No accelerometer samples found in DJI metadata.")

    time_s = np.array(times, dtype=float)
    time_s = time_s - float(time_s[0])
    accel_values = np.vstack(values)

    accel_series = TimeSeries(
        time_s=time_s,
        values=accel_values,
        axes=["x", "y", "z"],
        name="accel",
    )

    meta = SourceMeta(name="dji_osmo_action", kind="video", path=Path(path))

    return ImuBundle(
        gyro=None,
        accel=accel_series,
        channels={},
        meta=meta,
    )


@register_source("video")
class DjiOsmoActionSource(VideoSource):
    name = "dji_osmo_action"

    @classmethod
    def sniff(cls, path: Path) -> float:
        if path.suffix.lower() not in {".mp4", ".mov"}:
            return 0.0
        try:
            streams = _ffprobe_streams(path)
            has_djmd = any(s.get("codec_tag_string") == "djmd" for s in streams)
            if has_djmd:
                return 0.7
            tags = {}
            for stream in streams:
                tags.update(stream.get("tags") or {})
            encoder = str(tags.get("encoder", "")).lower()
            if "dji" in encoder or "osmo" in encoder:
                return 0.5
            return 0.2
        except Exception:
            return 0.2

    @classmethod
    def load(cls, path: Path, **opts) -> ImuBundle:
        return extract_dji_accel(Path(path))
