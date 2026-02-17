from pathlib import Path

import pytest

from imu_video_sync.cli import _detect_video_fps, _find_exiftool


@pytest.mark.parametrize(
    ("fixture_path", "expected_fps"),
    [
        (Path("tests/fixtures/gopro.mp4"), 29.97),
        (Path("tests/fixtures/dji.mp4"), 59.94),
    ],
)
def test_detect_video_fps_from_fixtures(fixture_path: Path, expected_fps: float) -> None:
    if not fixture_path.exists():
        pytest.skip(f"Missing video fixture: {fixture_path}")

    exiftool = _find_exiftool()
    assert exiftool is not None, "ExifTool not found; FPS detection requires tools/exiftool.exe."

    fps = _detect_video_fps(fixture_path)
    assert fps is not None
    assert fps == pytest.approx(expected_fps, abs=0.05)
