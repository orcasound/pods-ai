# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import os
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import requests

import audio_utils
from audio_utils import (
    download_60s_audio,
    download_60s_audio_from_start,
    download_audio_from_start_utc,
    get_all_folders,
    get_difference_between_times_in_seconds,
)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="requires time.tzset")
def test_difference_between_times_across_dst_transition(monkeypatch):
    original_timezone = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()

    try:
        earlier = datetime(2024, 3, 10, 9, 30, tzinfo=timezone.utc).timestamp()
        later = datetime(2024, 3, 10, 10, 30, tzinfo=timezone.utc).timestamp()

        assert get_difference_between_times_in_seconds(later, earlier) == 3600
    finally:
        if original_timezone is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_timezone)
        time.tzset()


def test_s3_folder_listing_delegates_to_orca_hls_utils():
    """Delegate S3 folder enumeration to the shared utility."""
    with patch(
        "audio_utils.s3_utils.get_all_folders",
        return_value=["1700000000"],
    ) as get_shared_folders:
        assert get_all_folders("audio-orcasound-net", "node/hls/") == [
            "1700000000"
        ]
    get_shared_folders.assert_called_once_with(
        "audio-orcasound-net", "node/hls/"
    )


def test_download_audio_uses_date_range_hls_stream(tmp_path):
    start = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    wav_path = str(tmp_path / "clip.wav")
    with patch("audio_utils.DateRangeHLSStream") as stream_class:
        stream_class.return_value.get_next_clip.return_value = (wav_path, "clip", None)

        result = download_audio_from_start_utc(
            "rpi_orcasound_lab", start, str(tmp_path), 8
        )

    assert result == wav_path
    stream_class.assert_called_once_with(
        "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_orcasound_lab",
        8,
        start.timestamp(),
        start.timestamp() + 8,
        str(tmp_path),
        overwrite_output=True,
    )
    stream_class.return_value.get_next_clip.assert_called_once_with()


def test_download_audio_returns_none_when_date_range_has_no_clip(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with patch("audio_utils.DateRangeHLSStream") as stream_class:
        stream_class.return_value.get_next_clip.return_value = (None, None, None)

        result = download_audio_from_start_utc("node", start, str(tmp_path), 3)

    assert result is None


def test_download_from_url_retries_transient_network_failures(tmp_path):
    """Retry shared downloads after transient connection errors."""
    with (
        patch(
            "audio_utils.scraper.download_from_url",
            side_effect=[
                requests.exceptions.ConnectionError(),
                requests.exceptions.ChunkedEncodingError(),
                None,
            ],
        ) as download_from_url,
        patch("audio_utils.time.sleep") as sleep,
    ):
        audio_utils.download_from_url("https://example.org/audio.ts", str(tmp_path))

    assert download_from_url.call_count == 3
    assert sleep.call_count == 2


def test_download_from_url_raises_after_three_transient_failures(tmp_path):
    """Raise the final transient failure after exhausting download retries."""
    failure = requests.exceptions.ConnectionError()
    with (
        patch(
            "audio_utils.scraper.download_from_url",
            side_effect=failure,
        ) as download_from_url,
        patch("audio_utils.time.sleep") as sleep,
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        audio_utils.download_from_url("https://example.org/audio.ts", str(tmp_path))

    assert download_from_url.call_count == 3
    assert sleep.call_count == 2


def test_download_from_url_retries_urlretrieve_network_failures(tmp_path):
    """Retry URL errors raised by the shared urlretrieve-based downloader."""
    with (
        patch(
            "audio_utils.scraper.download_from_url",
            side_effect=[urllib.error.URLError("temporary failure"), None],
        ) as download_from_url,
        patch("audio_utils.time.sleep") as sleep,
    ):
        audio_utils.download_from_url("https://example.org/audio.ts", str(tmp_path))

    assert download_from_url.call_count == 2
    sleep.assert_called_once_with(1)


def test_download_from_url_removes_partial_file_before_retry(tmp_path):
    """Remove a failed partial file so retry does not treat it as complete."""
    destination = tmp_path / "audio.ts"
    attempts = 0

    def fail_once_then_complete(url, directory):
        """Simulate urlretrieve leaving a partial destination before retry."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            destination.write_bytes(b"partial")
            raise urllib.error.ContentTooShortError("incomplete", b"partial")
        assert not destination.exists()
        destination.write_bytes(b"complete")

    with (
        patch(
            "audio_utils.scraper.download_from_url",
            side_effect=fail_once_then_complete,
        ) as download_from_url,
        patch("audio_utils.time.sleep"),
    ):
        audio_utils.download_from_url("https://example.org/audio.ts", str(tmp_path))

    assert download_from_url.call_count == 2
    assert destination.read_bytes() == b"complete"


def test_download_60s_audio_rounds_end_then_subtracts_clip_and_startup_offset():
    with patch("audio_utils.download_audio_from_start_utc", return_value="clip.wav") as download:
        result = download_60s_audio("rpi_test", "2025_01_01_00_01_02_PST", "tmp")

    assert result == "clip.wav"
    download.assert_called_once_with(
        "rpi_test",
        datetime(2025, 1, 1, 8, 0, 8, tzinfo=timezone.utc),
        "tmp",
        60,
    )


def test_download_60s_audio_from_start_applies_orcasite_hls_offset():
    with patch("audio_utils.download_audio_from_start_utc", return_value="clip.wav") as download:
        result = download_60s_audio_from_start(
            "rpi_test", "2025_01_01_00_00_02_PST", "tmp"
        )

    assert result == "clip.wav"
    download.assert_called_once_with(
        "rpi_test", datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc), "tmp", 60
    )
