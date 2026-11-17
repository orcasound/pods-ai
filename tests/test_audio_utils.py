# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import os
import time
import urllib.error
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
import requests

import audio_utils
from audio_utils import (
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


def test_download_audio_uses_orca_hls_utils_and_crops_requested_interval(tmp_path):
    """Select shared HLS segments and crop the requested audio interval."""
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    folder_start = int(start.timestamp()) - 3 * 60 * 60
    segment = Mock(
        start=start.timestamp() - 10,
        duration=10,
        url="https://example.org/hls/live000.ts",
    )
    archive = Mock(segments=[segment])

    def load_archive_and_live(_, folder):
        """Only the long-running session folder contains the requested audio."""
        return (archive, None) if folder == folder_start else (None, None)

    def download_segment(url, directory):
        """Create a local stand-in for a downloaded HLS segment."""
        (tmp_path / url.rsplit("/", 1)[-1]).write_bytes(b"segment")

    with (
        patch(
            "audio_utils.get_cached_folders",
            return_value=[
                str(int(start.timestamp()) - 2 * 60 * 60),
                str(folder_start),
                str(int(start.timestamp()) - 60 * 60),
                str(int(start.timestamp())),
            ],
        ) as get_cached_folders,
        patch(
            "audio_utils.playlist_utils.load_archive_and_live",
            side_effect=load_archive_and_live,
        ) as load_playlists,
        patch(
            "audio_utils.playlist_utils.get_playlist_timeline",
            return_value=[segment],
        ),
        patch(
            "audio_utils.playlist_utils.fill_timeline_gaps",
            return_value=[segment],
        ) as fill_timeline,
        patch(
            "audio_utils.playlist_utils.select_segments_from_start",
            return_value=[segment],
        ) as select_segments,
        patch(
            "audio_utils.scraper.download_from_url",
            side_effect=download_segment,
        ) as download_from_url,
        patch("audio_utils.ffmpeg.input", return_value="input") as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output") as ffmpeg_output,
        patch("audio_utils.ffmpeg.run") as ffmpeg_run,
    ):
        result = download_audio_from_start_utc(
            "rpi_orcasound_lab", start, str(tmp_path), 3
        )

    get_cached_folders.assert_called_once_with(
        "audio-orcasound-net", "rpi_orcasound_lab/hls/"
    )
    load_playlists.assert_called_once_with(
        "https://s3-us-west-2.amazonaws.com/"
        "audio-orcasound-net/rpi_orcasound_lab",
        folder_start,
    )
    fill_timeline.assert_called_once_with(
        [segment],
        start.timestamp() - 10,
        start.timestamp() + 3,
        audio_utils.playlist_utils.url_exists,
        extrapolate_start=True,
    )
    select_segments.assert_called_once_with(
        [segment], start.timestamp() - 10, 13
    )
    download_from_url.assert_called_once_with(
        "https://example.org/hls/live000.ts", str(tmp_path)
    )
    ffmpeg_input.assert_called_once_with(
        tmp_path / "live000.ts", ss=10.0
    )
    assert ffmpeg_output.call_args.kwargs["t"] == 3
    ffmpeg_run.assert_called_once_with(
        "output", overwrite_output=True, quiet=True
    )
    assert result == str(
        tmp_path / "temp_3s_rpi_orcasound_lab_2024_12_31_16_00_00_PST.wav"
    )


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
