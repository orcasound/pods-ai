# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for add_samples.py."""

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from add_samples import (
    DEFAULT_OUTPUT_DIR,
    HOP_DURATION,
    SEGMENT_DURATION,
    add_samples,
    format_timestamp_pst,
    get_segment_prediction,
    parse_timestamp_pst,
    split_wav_into_segments,
)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


class TestParseTimestampPst:
    """Tests for parse_timestamp_pst."""

    def test_parses_timestamp_with_pst_suffix(self):
        """parse_timestamp_pst should strip _PST and return a localized datetime."""
        dt = parse_timestamp_pst("2025_01_15_12_30_00_PST")
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 0

    def test_parses_timestamp_without_pst_suffix(self):
        """parse_timestamp_pst should also accept strings without _PST."""
        dt = parse_timestamp_pst("2025_01_15_12_30_00")
        assert dt.year == 2025
        assert dt.hour == 12


class TestFormatTimestampPst:
    """Tests for format_timestamp_pst."""

    def test_roundtrip(self):
        """Formatting a parsed timestamp should reproduce the original string."""
        original = "2025_06_01_08_15_30_PST"
        dt = parse_timestamp_pst(original)
        assert format_timestamp_pst(dt) == original

    def test_adds_pst_suffix(self):
        """format_timestamp_pst should always end with _PST."""
        dt = parse_timestamp_pst("2025_01_01_00_00_00_PST")
        assert format_timestamp_pst(dt).endswith("_PST")


# ---------------------------------------------------------------------------
# split_wav_into_segments
# ---------------------------------------------------------------------------


class TestSplitWavIntoSegments:
    """Tests for split_wav_into_segments."""

    def _make_fake_ffmpeg(self, duration: float):
        """Return patch helpers that fake ffmpeg.probe and ffmpeg.run."""
        fake_probe_result = {"format": {"duration": str(duration)}}
        return fake_probe_result

    def test_saves_correct_number_of_segments_for_60s_audio(self, tmp_path):
        """A 60-second file should produce 29 segments with 3s/2s settings."""
        # floor((60 - 3) / 2) + 1 = 29
        fake_probe = {"format": {"duration": "60.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run") as mock_run:

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert len(segments) == 29
        assert mock_run.call_count == 29

    def test_single_segment_for_short_audio(self, tmp_path):
        """Audio shorter than segment_duration should still produce one segment."""
        fake_probe = {"format": {"duration": "2.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert len(segments) == 1

    def test_filename_uses_node_name_with_hyphens(self, tmp_path):
        """Output filenames should replace underscores in the node name with hyphens."""
        fake_probe = {"format": {"duration": "3.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert len(segments) == 1
        name = segments[0][0].name
        assert name.startswith("rpi-orcasound-lab_")

    def test_first_segment_timestamp_matches_base(self, tmp_path):
        """The first segment's filename should encode the base timestamp."""
        fake_probe = {"format": {"duration": "3.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert "2025_01_15_12_30_00_PST" in segments[0][0].name

    def test_second_segment_timestamp_incremented_by_hop(self, tmp_path):
        """Each subsequent segment should be offset by hop_duration seconds."""
        fake_probe = {"format": {"duration": "10.0"}}

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        # Second segment should start 2 seconds later.
        assert "2025_01_15_12_30_02_PST" in segments[1][0].name

    def test_returns_empty_on_probe_failure(self, tmp_path, capsys):
        """split_wav_into_segments should return [] and print an error if probing fails."""
        with patch("add_samples.ffmpeg.probe", side_effect=Exception("probe error")):
            segments = split_wav_into_segments(
                wav_file="missing.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert segments == []
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_skips_existing_segments(self, tmp_path, capsys):
        """split_wav_into_segments should skip segments that already exist."""
        fake_probe = {"format": {"duration": "3.0"}}
        # Pre-create the expected output file.
        existing = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
        existing.write_bytes(b"existing")

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.run") as mock_run:

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        # ffmpeg.run should not be called for an already-existing segment.
        mock_run.assert_not_called()
        assert len(segments) == 1
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    def test_output_dir_is_created(self, tmp_path):
        """split_wav_into_segments should create the output directory if needed."""
        fake_probe = {"format": {"duration": "3.0"}}
        new_dir = tmp_path / "new_subdir"
        assert not new_dir.exists()

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.input") as mock_input, \
             patch("add_samples.ffmpeg.output") as mock_output, \
             patch("add_samples.ffmpeg.run"):

            mock_stream = MagicMock()
            mock_input.return_value = mock_stream
            mock_output.return_value = mock_stream

            split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=new_dir,
            )

        assert new_dir.exists()


# ---------------------------------------------------------------------------
# get_segment_prediction
# ---------------------------------------------------------------------------


class TestGetSegmentPrediction:
    """Tests for get_segment_prediction."""

    def test_podsai_returns_global_prediction_label(self, tmp_path):
        """For podsai, prediction label comes from global_prediction_label."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction_label": "humpback"}

        label = get_segment_prediction(mock_model, fake_path, "podsai")

        assert label == "humpback"

    def test_fastai_resident_prediction(self, tmp_path):
        """For fastai, global_prediction=1 should map to 'resident'."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction": 1}

        label = get_segment_prediction(mock_model, fake_path, "fastai")

        assert label == "resident"

    def test_fastai_other_prediction(self, tmp_path):
        """For fastai, global_prediction=0 should map to 'other'."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction": 0}

        label = get_segment_prediction(mock_model, fake_path, "fastai")

        assert label == "other"

    def test_orcahello_resident_prediction(self, tmp_path):
        """For orcahello, global_prediction=1 should map to 'resident'."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction": 1}

        label = get_segment_prediction(mock_model, fake_path, "orcahello")

        assert label == "resident"

    def test_returns_unknown_on_inference_failure(self, tmp_path, capsys):
        """get_segment_prediction should return 'unknown' if inference raises."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")

        label = get_segment_prediction(mock_model, fake_path, "podsai")

        assert label == "unknown"
        captured = capsys.readouterr()
        assert "Warning" in captured.err


# ---------------------------------------------------------------------------
# add_samples (integration-style)
# ---------------------------------------------------------------------------


class TestAddSamples:
    """Integration-style tests for add_samples."""

    def _fake_split(self, tmp_path):
        """Return a fake segments list with two pre-created files."""
        seg1 = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
        seg2 = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav"
        seg1.write_bytes(b"")
        seg2.write_bytes(b"")
        return [
            (seg1, "2025_01_15_12_30_00_PST"),
            (seg2, "2025_01_15_12_30_02_PST"),
        ]

    def test_returns_list_of_filepath_label_tuples(self, tmp_path):
        """add_samples should return (filepath, label) pairs for each segment."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction_label": "water"}

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model):

            results = add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_type="podsai",
                model_path="/path/to/model",
            )

        assert len(results) == 2
        for filepath, label in results:
            assert isinstance(filepath, str)
            assert label == "water"

    def test_raises_for_missing_podsai_model_path(self, tmp_path):
        """add_samples should raise ValueError when podsai is used without model_path."""
        fake_segments = self._fake_split(tmp_path)
        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             pytest.raises(ValueError, match="model_path is required"):
            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_type="podsai",
                model_path=None,
            )

    def test_returns_empty_when_no_segments(self, tmp_path):
        """add_samples should return [] if split_wav_into_segments yields nothing."""
        with patch("add_samples.split_wav_into_segments", return_value=[]):
            results = add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_type="fastai",
                model_path="./model",
            )

        assert results == []

    def test_fastai_default_model_path_is_set(self, tmp_path):
        """add_samples should default to './model' for fastai when model_path is None."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction": 0}

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model:

            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_type="fastai",
                model_path=None,
            )

        mock_get_model.assert_called_once_with(model_type="fastai", model_path="./model")

    def test_orcahello_default_model_path_is_set(self, tmp_path):
        """add_samples should default to the orcahello HuggingFace ID when model_path is None."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction": 0}

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model:

            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_type="orcahello",
                model_path=None,
            )

        mock_get_model.assert_called_once_with(
            model_type="orcahello",
            model_path="orcasound/orcahello-srkw-detector-v1",
        )

    def test_model_loaded_once_for_all_segments(self, tmp_path):
        """The model should be loaded exactly once regardless of the number of segments."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction_label": "water"}

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model:

            add_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                base_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_type="podsai",
                model_path="/path/to/model",
            )

        assert mock_get_model.call_count == 1
        assert mock_model.predict.call_count == 2
