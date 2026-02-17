from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceMeta:
    name: str
    kind: str
    path: Path
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimeSeries:
    time_s: np.ndarray
    values: np.ndarray
    axes: List[str]
    name: str
    units: Optional[str] = None


@dataclass(frozen=True)
class ImuBundle:
    gyro: Optional[TimeSeries] = None
    accel: Optional[TimeSeries] = None
    channels: Dict[str, TimeSeries] = field(default_factory=dict)
    meta: Optional[SourceMeta] = None


@dataclass(frozen=True)
class LogData:
    imu: ImuBundle
    df: pd.DataFrame
    time_col: str
    time_s: np.ndarray
