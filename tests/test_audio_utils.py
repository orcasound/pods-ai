# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import os
import shutil
import time
import urllib.error
import wave
from array import array
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import m3u8
import requests

import audio_utils
from audio_utils import (
    download_60s_audio,
    download_60s_audio_from_start,
    download_audio_from_start_utc,
    get_all_folders,
    get_difference_between_times_in_seconds,
)


def _write_segment(url, directory):
    (Path(directory) / os.path.basename(url)).write_bytes(b"segment")


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


def test_download_audio_crops_and_concatenates_playlist_segments(tmp_path):
    start = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\nlive000.ts\n#EXTINF:10,\nlive001.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )

    def download_segment(url, directory):
        (Path(directory) / url.rsplit("/", 1)[-1]).write_bytes(url[-6:].encode())

    def open_input(source_path, **kwargs):
        assert source_path.read_bytes() == b"000.ts001.ts"
        return "input"

    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 5)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url", side_effect=download_segment) as download,
        patch("audio_utils.ffmpeg.input", side_effect=open_input) as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output") as ffmpeg_output,
        patch("audio_utils.ffmpeg.run") as ffmpeg_run,
    ):
        result = download_audio_from_start_utc(
            "rpi_orcasound_lab", start, str(tmp_path), 8
        )

    assert download.call_count == 2
    assert ffmpeg_input.call_args.kwargs == {"ss": 3}
    ffmpeg_output.assert_called_once_with(
        "input", result, t=8, acodec="pcm_s16le", ar=44100, ac=1
    )
    ffmpeg_run.assert_called_once_with("output", overwrite_output=True, quiet=True)
    assert result.endswith("_2024_12_31_16_00_05_PST.wav")


def test_download_does_not_reuse_segments_with_the_same_basename(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\nfirst/live.ts\n#EXTINF:10,\nsecond/live.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )

    def download_segment(url, directory):
        (Path(directory) / "live.ts").write_bytes(url.split("/")[-2].encode())

    def open_input(source_path, **kwargs):
        assert source_path.read_bytes() == b"firstsecond"
        return "input"

    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 2)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url", side_effect=download_segment),
        patch("audio_utils.ffmpeg.input", side_effect=open_input),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 20) is not None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
def test_download_audio_crops_real_audio_to_requested_interval(tmp_path):
    source = tmp_path / "source.wav"
    samples = array("h", (second * 1000 for second in range(10) for _ in range(44100)))
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        wav.writeframes(samples.tobytes())
    start = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\nlive000.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 5)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url", side_effect=lambda url, directory: shutil.copyfile(
            source, Path(directory) / "live000.ts"
        )),
    ):
        result = download_audio_from_start_utc("node", start, str(tmp_path / "output"), 3)
    with wave.open(result, "rb") as wav:
        assert wav.getframerate() == 44100
        assert wav.getnframes() == 3 * 44100
        cropped = array("h", wav.readframes(wav.getnframes()))
    assert cropped[0] == 3000
    assert cropped[-1] == 5000


@pytest.mark.parametrize("second_start", ["00:00:20", "00:00:09"])
def test_download_rejects_timestamp_gap_or_overlap(tmp_path, second_start):
    start = datetime(2025, 1, 1, 0, 0, 8, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:00Z\n"
        "#EXTINF:10,\nlive000.ts\n"
        f"#EXT-X-PROGRAM-DATE-TIME:2025-01-01T{second_start}Z\n"
        "#EXTINF:10,\nlive001.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 8)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url") as download,
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 15) is None
    download.assert_not_called()


@pytest.mark.parametrize(("third_time", "accepted"), [
    ("00:00:32", False),
    ("00:00:21", False),
    ("00:00:22", True),
])
def test_thirty_second_request_requires_continuous_segments(tmp_path, third_time, accepted):
    start = datetime(2025, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:02Z\n"
        "#EXTINF:10,\nlive000.ts\n#EXTINF:10,\nlive001.ts\n"
        f"#EXT-X-PROGRAM-DATE-TIME:2025-01-01T{third_time}Z\n"
        "#EXTINF:10,\nlive002.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )

    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 2)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url", side_effect=_write_segment) as download,
        patch("audio_utils.ffmpeg.input", return_value="input"),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        result = download_audio_from_start_utc("node", start, str(tmp_path), 30)
    assert (result is not None) == accepted
    assert download.call_count == (3 if accepted else 0)


@pytest.mark.parametrize(("offset", "duration", "accepted"), [
    (5, 6, False), (12, 2, False), (20, 2, True),
])
def test_download_respects_gap_tags(tmp_path, offset, duration, accepted):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:00Z\n"
        "#EXTINF:10,\nlive000.ts\n#EXT-X-GAP\n#EXTINF:10,\nlive001.ts\n"
        "#EXTINF:10,\nlive002.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()))]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url", side_effect=_write_segment) as download,
        patch("audio_utils.ffmpeg.input", return_value="input"),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        result = download_audio_from_start_utc(
            "node", start + timedelta(seconds=offset), str(tmp_path), duration
        )
    assert (result is not None) == accepted
    assert download.call_count == int(accepted)


@pytest.mark.parametrize("sequence", [None, 0, 1])
@pytest.mark.parametrize("has_pdt", [False, True])
def test_playlist_start_timestamp(tmp_path, sequence, has_pdt):
    start = datetime(2025, 1, 1, 0, 0, 15, tzinfo=timezone.utc)
    sequence_tag = f"#EXT-X-MEDIA-SEQUENCE:{sequence}\n" if sequence is not None else ""
    pdt = "#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:10Z\n" if has_pdt else ""
    content = f"#EXTM3U\n{sequence_tag}{pdt}#EXTINF:10,\nlive001.ts\n"
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 5)]),
        patch("audio_utils.m3u8.load", side_effect=lambda url, **kwargs: m3u8.loads(
            content, uri="https://example.org/hls/live.m3u8", **kwargs
        )),
        patch("audio_utils.scraper.download_from_url", side_effect=_write_segment) as download,
        patch("audio_utils.ffmpeg.input", return_value="input") as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        result = download_audio_from_start_utc("node", start, str(tmp_path), 3)
    assert (result is not None) == (has_pdt or sequence is None)
    if result is not None:
        download.assert_called_once()
        assert ffmpeg_input.call_args.kwargs == {"ss": 5 if has_pdt else 3}
    else:
        download.assert_not_called()


def test_later_program_date_time_anchors_earlier_segment(tmp_path):
    start = datetime(2025, 1, 1, 0, 0, 12, tzinfo=timezone.utc)
    playlist = m3u8.loads(
        "#EXTM3U\n#EXTINF:10,\nlive000.ts\n"
        "#EXT-X-PROGRAM-DATE-TIME:2025-01-01T00:00:20Z\n"
        "#EXTINF:10,\nlive001.ts\n",
        uri="https://example.org/hls/archive.m3u8",
    )
    with (
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 12)]),
        patch("audio_utils.m3u8.load", return_value=playlist),
        patch("audio_utils.scraper.download_from_url", side_effect=_write_segment),
        patch("audio_utils.ffmpeg.input", return_value="input") as ffmpeg_input,
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 3) is not None
    assert ffmpeg_input.call_args.kwargs == {"ss": 2}


def test_download_falls_back_to_live_playlist(tmp_path):
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
        patch("audio_utils.get_cached_folders", return_value=[str(int(start.timestamp()) - 2)]),
        patch("audio_utils.m3u8.load", side_effect=[gap, live]),
        patch("audio_utils.scraper.download_from_url", side_effect=_write_segment) as download,
        patch("audio_utils.ffmpeg.input", return_value="input"),
        patch("audio_utils.ffmpeg.output", return_value="output"),
        patch("audio_utils.ffmpeg.run"),
    ):
        assert download_audio_from_start_utc("node", start, str(tmp_path), 3) is not None
    assert download.call_args.args[0] == "https://example.org/hls/available.ts"


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_download_rejects_invalid_duration(tmp_path, duration):
    with patch("audio_utils.get_cached_folders") as folders:
        assert download_audio_from_start_utc(
            "node", datetime(2025, 1, 1, tzinfo=timezone.utc), str(tmp_path), duration
        ) is None
    folders.assert_not_called()


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


@pytest.mark.parametrize(
    ("end_timestamp", "expected_start"),
    [
        ("2025_01_01_00_01_02_PST", datetime(2025, 1, 1, 8, 0, 10, tzinfo=timezone.utc)),
        ("2025_01_01_00_01_00_PST", datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_download_60s_audio_aligns_end_without_offset(end_timestamp, expected_start):
    with patch("audio_utils.download_audio_from_start_utc", return_value="clip.wav") as download:
        result = download_60s_audio("rpi_test", end_timestamp, "tmp")

    assert result == "clip.wav"
    download.assert_called_once_with(
        "rpi_test",
        expected_start,
        "tmp",
        60,
    )


def test_download_60s_audio_from_start_applies_no_offset():
    with patch("audio_utils.download_audio_from_start_utc", return_value="clip.wav") as download:
        result = download_60s_audio_from_start(
            "rpi_test", "2025_01_01_00_00_00_PST", "tmp"
        )

    assert result == "clip.wav"
    download.assert_called_once_with(
        "rpi_test", datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc), "tmp", 60
    )
