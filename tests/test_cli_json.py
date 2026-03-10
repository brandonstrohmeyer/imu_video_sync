from imu_video_sync import cli


def test_offset_summary_payload_positive():
    payload = cli._offset_summary_payload(1.0, 30.0)
    assert payload["lag_seconds"] == "+1.000"
    assert payload["lag_frames"] == "+30"
    assert payload["timecode_offset"] == "+00:00:01;00"
    assert payload["video_offset"] == "00:00:01.000"
    assert "data_offset" not in payload


def test_offset_summary_payload_negative():
    payload = cli._offset_summary_payload(-2.5, 20.0)
    assert payload["lag_seconds"] == "-2.500"
    assert payload["lag_frames"] == "-50"
    assert payload["timecode_offset"] == "-00:00:02;10"
    assert payload["data_offset"] == "00:00:02.500"
    assert "video_offset" not in payload


def test_offset_summary_payload_missing_fps():
    payload = cli._offset_summary_payload(0.5, None)
    assert payload["lag_seconds"] == "+0.500"
    assert payload["lag_frames"] is None
    assert payload["timecode_offset"] is None
    assert payload["video_offset"] == "00:00:00.500"
