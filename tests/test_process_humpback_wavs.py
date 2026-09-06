# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Tests that signals-humpback clips are retained as labeled training windows."""

from pathlib import Path
from unittest.mock import MagicMock

from process_humpback_wavs import (
    process_external_humpback_wavs,
    should_retain_humpback_source,
)


def test_should_retain_humpback_source_keeps_long_clips_and_rejects_noise():
    assert should_retain_humpback_source(3.0, segment_duration=3) is True
    assert should_retain_humpback_source(10.0, segment_duration=3) is True
    assert should_retain_humpback_source(2.9, segment_duration=3) is False
    assert should_retain_humpback_source(0.4, segment_duration=3) is False


def test_process_external_humpback_wavs_keeps_humpback_and_skips_short_noise(
    tmp_path, monkeypatch
):
    external_dir = tmp_path / "signals-humpback"
    external_dir.mkdir()
    (external_dir / "song.wav").write_bytes(b"humpback")
    (external_dir / "noise_blip.wav").write_bytes(b"noise")
    output_root = tmp_path / "wav"

    def fake_probe(path):
        name = Path(path).name
        if name == "song.wav":
            return {"format": {"duration": "10.0"}}
        return {"format": {"duration": "1.2"}}

    def fake_input(*_args, **_kwargs):
        return MagicMock()

    def fake_output(stream, path, **_kwargs):
        stream.out_path = path
        return stream

    def fake_run(stream, **_kwargs):
        out_path = Path(stream.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"segment")

    monkeypatch.setattr("process_humpback_wavs.ffmpeg.probe", fake_probe)
    monkeypatch.setattr("process_humpback_wavs.ffmpeg.input", fake_input)
    monkeypatch.setattr("process_humpback_wavs.ffmpeg.output", fake_output)
    monkeypatch.setattr("process_humpback_wavs.ffmpeg.run", fake_run)

    process_external_humpback_wavs(external_dir, output_root, segment_duration=3)

    humpback_dir = output_root / "humpback"
    kept = sorted(path.name for path in humpback_dir.glob("*.wav"))
    assert kept == [
        "signals-humpback_song_0000s.wav",
        "signals-humpback_song_0003s.wav",
        "signals-humpback_song_0006s.wav",
    ]
    assert not list(humpback_dir.glob("*noise_blip*"))
