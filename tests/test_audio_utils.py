# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import os
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
import requests
import m3u8

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


def test_download_audio_crops_and_concatenates_released_playlist_segments(tmp_path):
    start = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    folder = int(start.timestamp()) - 5
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\nlive000.ts\n#EXTINF:10,\nlive001.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )

    def download_segment(url, directory):
        (tmp_path / url.rsplit("/", 1)[-1]).write_bytes(url[-6:].encode())

    with (
        patch("audio_utils.get_cached_folders", return_value=[str(folder)]),
        patch("audio_utils.m3u8.load", return_value=playlist) as load,
        patch("audio_utils.scraper.download_from_url", side_effect=download_segment) as download,
        patch("audio_utils.ffmpeg.input", return_value="input") as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output") as ffmpeg_output,
        patch("audio_utils.ffmpeg.run") as ffmpeg_run,
    ):
        result = download_audio_from_start_utc("rpi_orcasound_lab", start, str(tmp_path), 8)

    assert load.call_args.args[0].endswith(f"/{folder}/archive.m3u8")
    assert download.call_count == 2
    ffmpeg_input.assert_called_once_with(
        tmp_path / "temp_8s_rpi_orcasound_lab_segments.ts", ss=5
    )
    assert (tmp_path / "temp_8s_rpi_orcasound_lab_segments.ts").read_bytes() == b"000.ts001.ts"
    assert ffmpeg_output.call_args.kwargs["t"] == 8
    ffmpeg_run.assert_called_once_with("output", overwrite_output=True, quiet=True)
    assert result == str(tmp_path / "temp_8s_rpi_orcasound_lab_2024_12_31_16_00_05_PST.wav")


@pytest.mark.parametrize("has_pdt", [False, True])
def test_rolling_playlist_requires_timestamp(tmp_path, has_pdt):
    start = datetime(2025, 1, 1, 0, 0, 15, tzinfo=timezone.utc)
    folder = int(start.timestamp()) - 15
    pdt = "#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:10Z\n" if has_pdt else ""
    playlist = m3u8.loads(
        f"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:1\n{pdt}#EXTINF:10,\nlive001.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(folder)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url") as download,
        patch("audio_utils.ffmpeg.input", return_value="input") as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        result = download_audio_from_start_utc("node", start, str(tmp_path), 3)
    assert (result is not None) == has_pdt
    if has_pdt:
        download.assert_called_once_with("https://example.org/hls/live001.ts", str(tmp_path))
        ffmpeg_input.assert_called_once_with(tmp_path / "live001.ts", ss=5)
    else:
        download.assert_not_called()


def test_download_retries_and_falls_back_to_live_playlist(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\nlive000.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    missing = urllib.error.HTTPError("url", 404, "missing", None, None)
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()))]),
        patch("audio_utils.m3u8.load", side_effect=[
            missing, requests.exceptions.ConnectionError("temporary"), playlist,
        ]) as load,
        patch("audio_utils.time.sleep") as sleep,
        patch("audio_utils.scraper.download_from_url"),
        patch("audio_utils.ffmpeg.input", return_value="input"),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        result = download_audio_from_start_utc("node", start, str(tmp_path), 3)
    assert result is not None
    assert load.call_count == 3
    assert load.call_args.args[0].endswith("/live.m3u8")
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize("second_start", ["00:00:20", "00:00:09"])
def test_download_rejects_timestamp_gap_or_overlap(tmp_path, second_start):
    start = datetime(2025, 1, 1, 0, 0, 8, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:00Z\n"
        "#EXTINF:10,\nlive000.ts\n"
        f"#EXT-X-PROGRAM-DATE-TIME:2025-01-01T{second_start}Z\n"
        "#EXTINF:10,\nlive001.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 8)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url") as download,
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 15) is None
    download.assert_not_called()


@pytest.mark.parametrize(
    ("offset", "duration", "expected_uri"),
    [
        (5, 6, None),
        (12, 2, None),
        (19, 3, None),
        (0, 10, "live10.ts"),
        (20, 2, "live12.ts"),
    ],
)
def test_download_respects_gap_tags(tmp_path, offset, duration, expected_uri):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:00Z\n"
        "#EXTINF:10,\nlive10.ts\n"
        "#EXT-X-GAP\n#EXTINF:10,\nlive11.ts\n"
        "#EXTINF:10,\nlive12.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()))]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url") as download,
        patch("audio_utils.ffmpeg.input", return_value="input"),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        result = download_audio_from_start_utc(
            "node", start + timedelta(seconds=offset), str(tmp_path), duration
        )

    assert (result is not None) == (expected_uri is not None)
    if expected_uri is None:
        download.assert_not_called()
    else:
        download.assert_called_once_with(f"https://example.org/hls/{expected_uri}", str(tmp_path))


def test_download_falls_back_when_archive_has_gap(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    gap = m3u8.loads(
        "#EXTM3U\n#EXT-X-GAP\n#EXTINF:10,\nmissing.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )
    live = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\navailable.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()))]),
        patch("audio_utils.m3u8.load", side_effect=[gap, live]) as load,
        patch("audio_utils.scraper.download_from_url") as download,
        patch("audio_utils.ffmpeg.input", return_value="input"),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 3) is not None
    assert load.call_count == 2
    download.assert_called_once_with("https://example.org/hls/available.ts", str(tmp_path))


def test_segment_filenames_do_not_define_media_sequence():
    playlist = m3u8.loads(
        "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:10\n"
        "#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:00Z\n"
        "#EXTINF:10,\nlive10.ts\n#EXTINF:10,\nlive12.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    timeline = audio_utils._playlist_timeline(playlist, int(start))

    assert [segment.media_sequence for segment in playlist.segments] == [10, 11]
    assert audio_utils._segments_cover_interval(timeline, start, start + 20)


def test_download_checks_older_folders_and_empty_list(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    folder = int(start.timestamp()) - 20
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:30,\nlive000.ts\n",
        uri="https://example.org/hls/live.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[
            str(folder - 86400), str(folder), str(folder + 10),
        ]),
        patch("audio_utils.m3u8.load", side_effect=[
            m3u8.loads("#EXTM3U"), m3u8.loads("#EXTM3U"), playlist,
        ]) as load,
        patch("audio_utils.scraper.download_from_url"),
        patch("audio_utils.ffmpeg.input", return_value="input") as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 3) is not None
        ffmpeg_input.assert_called_once_with(tmp_path / "live000.ts", ss=20)
        assert load.call_count == 3
    with patch("audio_utils.get_cached_folders", return_value=[]):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 3) is None


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
