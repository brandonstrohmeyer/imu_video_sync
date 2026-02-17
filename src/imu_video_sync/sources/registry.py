from __future__ import annotations

from pathlib import Path
from typing import List, Type

from .base import LogSource, VideoSource


VIDEO_SOURCES: List[Type[VideoSource]] = []
LOG_SOURCES: List[Type[LogSource]] = []


def register_source(kind: str, cls=None):
    if cls is None:
        def _decorator(source_cls):
            return register_source(kind, source_cls)
        return _decorator

    if kind == "video":
        VIDEO_SOURCES.append(cls)
    elif kind == "log":
        LOG_SOURCES.append(cls)
    else:
        raise ValueError(f"Unknown source kind: {kind}")
    return cls


def _find_by_name(sources, name: str):
    for source in sources:
        if source.name.lower() == name.lower():
            return source
    return None


def resolve_source(kind: str, path: Path, forced: str | None = None):
    sources = VIDEO_SOURCES if kind == "video" else LOG_SOURCES
    if not sources:
        raise ValueError(f"No sources registered for kind: {kind}")

    if forced:
        match = _find_by_name(sources, forced)
        if match is None:
            known = ", ".join(sorted(s.name for s in sources))
            raise ValueError(f"Unknown {kind} source '{forced}'. Known: {known}")
        return match

    best = None
    best_score = 0.0
    for source in sources:
        score = float(source.sniff(path))
        if score > best_score:
            best = source
            best_score = score

    if best is None or best_score <= 0.0:
        known = ", ".join(sorted(s.name for s in sources))
        raise ValueError(f"Unable to resolve {kind} source for {path}. Known: {known}")

    return best
