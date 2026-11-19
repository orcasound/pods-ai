# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

import os
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from audio_utils import (
    download_60s_audio,
    download_60s_audio_from_start,
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
