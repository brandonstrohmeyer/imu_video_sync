from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class VideoSource:
    name = "unknown-video"

    @classmethod
    def sniff(cls, path: Path) -> float:
        return 0.0

    @classmethod
    def load(cls, path: Path, **opts) -> Any:
        raise NotImplementedError


class LogSource:
    name = "unknown-log"

    @classmethod
    def sniff(cls, path: Path) -> float:
        return 0.0

    @classmethod
    def load(cls, path: Path, **opts) -> Any:
        raise NotImplementedError
