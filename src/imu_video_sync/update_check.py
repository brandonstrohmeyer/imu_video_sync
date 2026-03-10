from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.request
from typing import Optional, Tuple

from . import __version__

RELEASES_URL = "https://api.github.com/repos/brandonstrohmeyer/imu_video_sync/releases"
FALLBACK_RELEASES_PAGE = "https://github.com/brandonstrohmeyer/imu_video_sync/releases"
DISABLE_ENV_VAR = "IMU_VIDEO_SYNC_DISABLE_UPDATE_CHECK"
DEFAULT_TIMEOUT_S = 2.5

_TAG_RE = re.compile(
    r"^v?(?P<maj>0|[1-9]\d*)\.(?P<min>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-rc\.(?P<rc>\d+))?$"
)
_DEV_TAG_RE = re.compile(
    r"^v?(?P<maj>0|[1-9]\d*)\.(?P<min>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"-(?P<ts>\d{12})$"
)


@dataclass(frozen=True)
class UpdateResult:
    current_version: str
    latest_version: str
    update_available: bool
    release_url: str


@dataclass(frozen=True)
class _VersionCandidate:
    display: str
    key: Tuple[int, int, int, int, int]
    url: str


def is_disabled() -> bool:
    value = os.getenv(DISABLE_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def check_for_updates(
    *,
    include_prereleases: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Optional[UpdateResult]:
    if is_disabled():
        return None

    current_display = __version__.strip()
    current_key = _parse_tag(current_display)
    if current_key is None:
        return None

    releases = _fetch_releases(timeout_s=timeout_s)
    latest = _select_latest(releases, include_prereleases=include_prereleases)
    if latest is None:
        return None

    update_available = latest.key > current_key
    return UpdateResult(
        current_version=current_display,
        latest_version=latest.display,
        update_available=update_available,
        release_url=latest.url,
    )


def format_update_notice(result: UpdateResult) -> str:
    return (
        f"Update available: {result.latest_version} "
        f"(current {result.current_version}). "
        f"Download: {result.release_url}"
    )


def _fetch_releases(*, timeout_s: float) -> list[dict]:
    headers = {
        "User-Agent": f"imu-video-sync/{__version__}",
        "Accept": "application/vnd.github+json",
    }
    request = urllib.request.Request(RELEASES_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8")
    except Exception:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return []


def _select_latest(
    releases: list[dict], *, include_prereleases: bool
) -> Optional[_VersionCandidate]:
    best: Optional[_VersionCandidate] = None
    for release in releases:
        if release.get("draft"):
            continue
        if release.get("prerelease") and not include_prereleases:
            continue
        tag_raw = str(release.get("tag_name", "")).strip()
        tag = _sanitize_tag(tag_raw)
        key = _parse_tag(tag)
        if key is None:
            continue
        url = str(release.get("html_url") or "").strip()
        if not url:
            url = f"{FALLBACK_RELEASES_PAGE}/tag/{tag}"
        cand = _VersionCandidate(display=tag, key=key, url=url)
        if best is None or cand.key > best.key:
            best = cand
    return best


def _sanitize_tag(tag: str) -> str:
    cleaned = tag.replace("\\n", "").replace("\\r", "")
    cleaned = cleaned.replace("\n", "").replace("\r", "")
    return cleaned.strip()


def _parse_tag(tag: str) -> Optional[Tuple[int, int, int, int, int]]:
    clean = _sanitize_tag(tag)
    match = _TAG_RE.match(clean)
    if not match:
        dev_match = _DEV_TAG_RE.match(clean)
        if not dev_match:
            return None
        major = int(dev_match.group("maj"))
        minor = int(dev_match.group("min"))
        patch = int(dev_match.group("patch"))
        return (major, minor, patch, 0, -1)
    major = int(match.group("maj"))
    minor = int(match.group("min"))
    patch = int(match.group("patch"))
    rc_raw = match.group("rc")
    if rc_raw is None:
        is_stable = 1
        rc_num = 0
    else:
        is_stable = 0
        rc_num = int(rc_raw)
    return (major, minor, patch, is_stable, rc_num)
