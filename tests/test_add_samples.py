# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for add_samples.py."""

import csv
import io
import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from model_revisions import LATEST_PODSAI_MODEL_REVISION
from audio_utils import (
    format_timestamp_pacific,
    parse_timestamp_pacific,
)
from add_samples import (
    DEFAULT_MODEL_PATH,
    add_testing_60s_sample,
    add_training_3s_samples,
    get_segment_prediction,
    lookup_detection_in_csv,
    main as add_samples_main,
    parse_node_and_timestamp_from_filename,
    parse_uri,
    split_wav_into_segments,
)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


class TestParseTimestampPacific:
    """Tests for parse_timestamp_pacific."""

    def test_parses_timestamp_with_pst_suffix(self):
        """parse_timestamp_pacific should strip _PST and return a localized datetime."""
        dt = parse_timestamp_pacific("2025_01_15_12_30_00_PST")
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 0

    def test_parses_daylight_and_standard_time_without_exception(self):
        """Both suffixes parse and retain their correct Pacific offset."""
        pdt = parse_timestamp_pacific("2025_07_15_12_30_00_PDT")
        pst = parse_timestamp_pacific("2025_01_15_12_30_00_PST")
        assert pdt.tzname() == "PDT"
        assert pst.tzname() == "PST"

class TestFormatTimestampPacific:
    """Tests for format_timestamp_pacific."""

    def test_roundtrip(self):
        """Formatting a parsed timestamp should reproduce the original string."""
        original = "2025_06_01_08_15_30_PDT"
        dt = parse_timestamp_pacific(original)
        assert format_timestamp_pacific(dt) == original

    def test_formats_winter_as_pst(self):
        """January timestamps use the standard-time abbreviation."""
        dt = parse_timestamp_pacific("2025_01_01_00_00_00_PST")
        assert format_timestamp_pacific(dt).endswith("_PST")

    @pytest.mark.parametrize(
        ("utc_timestamp", "expected"),
        [
            (datetime(2024, 7, 15, 12, tzinfo=timezone.utc), "2024_07_15_05_00_00_PDT"),
            (datetime(2024, 1, 15, 12, tzinfo=timezone.utc), "2024_01_15_04_00_00_PST"),
            (datetime(2024, 3, 10, 9, 59, 59, tzinfo=timezone.utc), "2024_03_10_01_59_59_PST"),
            (datetime(2024, 3, 10, 10, tzinfo=timezone.utc), "2024_03_10_03_00_00_PDT"),
            (datetime(2024, 11, 3, 8, 59, 59, tzinfo=timezone.utc), "2024_11_03_01_59_59_PDT"),
            (datetime(2024, 11, 3, 9, tzinfo=timezone.utc), "2024_11_03_01_00_00_PST"),
        ],
    )
    def test_formats_dynamic_pacific_abbreviation(self, utc_timestamp, expected):
        """UTC timestamps use the applicable Pacific standard or daylight abbreviation."""
        assert format_timestamp_pacific(utc_timestamp) == expected

    def test_parses_repeated_fall_back_hour_using_suffix(self):
        """The suffix identifies the occurrence during the repeated fall-back hour."""
        pdt = parse_timestamp_pacific("2024_11_03_01_30_00_PDT")
        pst = parse_timestamp_pacific("2024_11_03_01_30_00_PST")
        assert (pst - pdt).total_seconds() == 3600

    def test_fall_back_pst_row_does_not_match_distinct_pdt_instant(self, tmp_path):
        detections_csv = tmp_path / "detections.csv"
        with detections_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["NodeName", "StartTimestamp", "URI", "Description", "Notes", "Tags"],
            )
            writer.writeheader()
            writer.writerow({
                "NodeName": "rpi_sunset_bay",
                "StartTimestamp": "2024_11_03_01_30_00_PST",
                "URI": "https://example.com/standard-time",
                "Description": "Fall-back PST occurrence",
                "Notes": "fp_machine_only",
                "Tags": "human",
            })

        result = lookup_detection_in_csv(
            "rpi_sunset_bay", "2024_11_03_01_30_00_PDT", str(detections_csv)
        )

        assert result is None


def test_parse_uri_returns_correct_timestamp():
    """URI timestamps are converted to corrected local time."""
    with patch("add_samples.get_node_name_from_slug", return_value="rpi_test"):
        node_name, timestamp = parse_uri(
            "https://live.orcasound.net/bouts/new/test?time=2025-01-01T08%3A00%3A00.000Z"
        )

    assert node_name == "rpi_test"
    assert timestamp == "2025_01_01_00_00_02_PST"


def test_detection_lookup_matches_corrected_manifest_timestamp(tmp_path):
    detections_csv = tmp_path / "detections.csv"
    with detections_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["NodeName", "StartTimestamp", "URI", "Description", "Notes", "Tags"],
        )
        writer.writeheader()
        writer.writerow({
            "NodeName": "rpi_sunset_bay",
            "StartTimestamp": "2025_01_15_12_30_02_PST",
            "URI": "https://example.com/orcasite-time",
            "Description": "Existing detection",
            "Notes": "fp_machine_only",
            "Tags": "vessel",
        })

    result = lookup_detection_in_csv(
        "rpi_sunset_bay",
        "2025_01_15_12_30_02_PST",
        str(detections_csv),
    )

    assert result is not None
    assert result.uri == "https://example.com/orcasite-time"


# ---------------------------------------------------------------------------
# parse_node_and_timestamp_from_filename
# ---------------------------------------------------------------------------


class TestParseNodeAndTimestampFromFilename:
    """Tests for parse_node_and_timestamp_from_filename."""

    def test_parses_standard_filename(self):
        """Should extract node name (with underscores) and timestamp from a well-formed filename."""
        node, ts = parse_node_and_timestamp_from_filename(
            "rpi-orcasound-lab_2025_12_17_22_34_03_PST.wav"
        )
        assert node == "rpi_orcasound_lab"
        assert ts == "2025_12_17_22_34_03_PST"

    def test_parses_path_with_directory(self):
        """Should ignore leading directory components and parse only the basename."""
        node, ts = parse_node_and_timestamp_from_filename(
            "/some/path/rpi-sunset-bay_2026_01_01_00_00_00_PST.wav"
        )
        assert node == "rpi_sunset_bay"
        assert ts == "2026_01_01_00_00_00_PST"

    def test_node_name_with_single_segment(self):
        """Node names that contain no hyphens should still be parsed correctly."""
        node, ts = parse_node_and_timestamp_from_filename(
            "orcasound-lab_2025_06_01_08_15_30_PST.wav"
        )
        assert node == "orcasound_lab"
        assert ts == "2025_06_01_08_15_30_PST"

    def test_normalizes_lowercase_timezone_suffix(self):
        """Extracted timezone suffixes are normalized for downstream parsing."""
        node, ts = parse_node_and_timestamp_from_filename(
            "rpi-orcasound-lab_2025_06_01_08_15_30_pdt.wav"
        )
        assert node == "rpi_orcasound_lab"
        assert ts == "2025_06_01_08_15_30_PDT"

    def test_raises_for_filename_without_timestamp(self):
        """Should raise ValueError when the filename has no recognizable timestamp."""
        with pytest.raises(ValueError, match="Cannot infer"):
            parse_node_and_timestamp_from_filename("recording.wav")

    def test_raises_for_filename_missing_pst_suffix(self):
        """Should raise ValueError when _PST is absent from the timestamp portion."""
        with pytest.raises(ValueError, match="Cannot infer"):
            parse_node_and_timestamp_from_filename(
                "rpi-orcasound-lab_2025_12_17_22_34_03.wav"
            )

    def test_raises_for_non_wav_extension(self):
        """Should raise ValueError for files that do not end in .wav."""
        with pytest.raises(ValueError, match="Cannot infer"):
            parse_node_and_timestamp_from_filename(
                "rpi-orcasound-lab_2025_12_17_22_34_03_PST.mp3"
            )


# ---------------------------------------------------------------------------
# split_wav_into_segments
# ---------------------------------------------------------------------------


class TestSplitWavIntoSegments:
    """Tests for split_wav_into_segments."""

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
                start_timestamp="2025_01_15_12_30_02_PST",
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
                start_timestamp="2025_01_15_12_30_00_PST",
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
                start_timestamp="2025_01_15_12_30_00_PST",
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
                start_timestamp="2025_01_15_12_30_00_PST",
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
                start_timestamp="2025_01_15_12_30_00_PST",
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
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        assert segments == []
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_reextracts_existing_segment_without_metadata(self, tmp_path, capsys):
        """A filename-only cache hit is unverified and must be re-extracted."""
        fake_probe = {"format": {"duration": "3.0"}}
        existing = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
        existing.write_bytes(b"existing")

        def fake_run(*args, **kwargs):
            existing.write_bytes(b"re-extracted")

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.run", side_effect=fake_run) as mock_run:

            segments = split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
            )

        mock_run.assert_called_once()
        assert len(segments) == 1
        assert existing.read_bytes() == b"re-extracted"
        assert existing.with_name(f"{existing.name}.unverified").read_bytes() == b"existing"
        assert existing.with_name(f"{existing.name}.metadata.json").is_file()
        captured = capsys.readouterr()
        assert "Saved" in captured.out

    @pytest.mark.parametrize(
        "collision_timestamp",
        ["2025_01_15_12_30_00_PST", "2025_01_15_12_30_02_PST"],
    )
    def test_rejects_input_that_collides_with_a_segment_path(
        self, tmp_path, collision_timestamp
    ):
        """A colliding input is rejected before probing or modifying the source."""
        source_path = tmp_path / f"rpi-orcasound-lab_{collision_timestamp}.wav"
        original_contents = b"original source audio"
        source_path.write_bytes(original_contents)

        with patch("add_samples.ffmpeg.probe") as mock_probe:
            with pytest.raises(ValueError, match="collides with an output segment path"):
                split_wav_into_segments(
                    wav_file=str(source_path),
                    node_name="rpi_orcasound_lab",
                    start_timestamp="2025_01_15_12_30_00_PST",
                    output_dir=tmp_path,
                )

        mock_probe.assert_not_called()
        assert source_path.read_bytes() == original_contents
        assert not source_path.with_name(f"{source_path.name}.unverified").exists()

    def test_cleans_partial_output_and_metadata_when_extraction_fails(self, tmp_path):
        """A failed ffmpeg run must not leave an output or stale sidecar metadata."""
        source_path = tmp_path / "source.wav"
        source_path.write_bytes(b"source audio")
        output_dir = tmp_path / "segments"
        out_path = output_dir / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
        metadata_path = out_path.with_name(f"{out_path.name}.metadata.json")
        output_dir.mkdir()
        metadata_path.write_text("stale metadata", encoding="utf-8")

        def failing_run(*_args, **_kwargs):
            out_path.write_bytes(b"partial output")
            raise RuntimeError("ffmpeg failed")

        with patch("add_samples.ffmpeg.probe", return_value={"format": {"duration": "3.0"}}), \
             patch("add_samples.ffmpeg.run", side_effect=failing_run):
            segments = split_wav_into_segments(
                wav_file=str(source_path),
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=output_dir,
            )

        assert segments == []
        assert not out_path.exists()
        assert not metadata_path.exists()

    def test_source_id_reuses_cache_across_temporary_source_paths(self, tmp_path):
        """Stable source identities cache segments across ephemeral WAV paths."""
        fake_probe = {"format": {"duration": "3.0"}}
        wav_one = tmp_path / "first" / "download.wav"
        wav_two = tmp_path / "second" / "download.wav"
        wav_one.parent.mkdir()
        wav_two.parent.mkdir()
        wav_one.write_bytes(b"source one")
        wav_two.write_bytes(b"source two")
        output_dir = tmp_path / "segments"
        source_id = "orcasite:rpi_orcasound_lab:2025-01-15T20:30:00+00:00"

        def fake_run(*args, **kwargs):
            segment_path = output_dir / (
                "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"
            )
            segment_path.write_bytes(b"cached segment")

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.run", side_effect=fake_run) as mock_run:
            first_segments = split_wav_into_segments(
                wav_file=str(wav_one),
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=output_dir,
                source_id=source_id,
            )
            second_segments = split_wav_into_segments(
                wav_file=str(wav_two),
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=output_dir,
                source_id=source_id,
            )

        mock_run.assert_called_once()
        assert first_segments == second_segments
        metadata_path = first_segments[0][0].with_name(
            f"{first_segments[0][0].name}.metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["source"] == {"source_id": source_id}

    @pytest.mark.parametrize(
        ("metadata_key", "metadata_value"),
        [
            ("source", {"uri": "https://example.org/other"}),
            ("start_offset_seconds", 99),
        ],
    )
    def test_reextracts_when_cached_source_or_offset_metadata_differs(
        self, tmp_path, metadata_key, metadata_value
    ):
        fake_probe = {"format": {"duration": "3.0"}}
        existing = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav"

        def fake_run(*args, **kwargs):
            existing.write_bytes(b"fresh segment")

        with patch("add_samples.ffmpeg.probe", return_value=fake_probe), \
             patch("add_samples.ffmpeg.run", side_effect=fake_run) as mock_run:
            split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
                source_uri="https://example.org/source",
            )
            metadata_path = existing.with_name(f"{existing.name}.metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata[metadata_key] = metadata_value
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
                source_uri="https://example.org/source",
            )
            split_wav_into_segments(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=tmp_path,
                source_uri="https://example.org/source",
            )

        assert mock_run.call_count == 2
        assert existing.read_bytes() == b"fresh segment"
        assert existing.with_name(f"{existing.name}.unverified").exists()

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
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=new_dir,
            )

        assert new_dir.exists()


# ---------------------------------------------------------------------------
# get_segment_prediction
# ---------------------------------------------------------------------------


class TestGetSegmentPrediction:
    """Tests for get_segment_prediction."""

    def test_podsai_returns_label_and_confidence(self, tmp_path):
        """Prediction returns both label and confidence from model output."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "humpback",
            "global_confidence": 0.85
        }

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "humpback"
        assert confidence == 0.85

    def test_returns_unknown_and_zero_confidence_when_label_missing(self, tmp_path):
        """get_segment_prediction should return ('unknown', 0.0) when key is absent."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {}

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "unknown"
        assert confidence == 0.0

    def test_returns_label_with_default_confidence_when_confidence_missing(self, tmp_path):
        """get_segment_prediction should return label with 0.0 confidence if confidence key is absent."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {"global_prediction_label": "resident"}

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "resident"
        assert confidence == 0.0

    def test_returns_unknown_and_zero_on_inference_failure(self, tmp_path, capsys):
        """get_segment_prediction should return ('unknown', 0.0) if inference raises."""
        fake_path = tmp_path / "seg.wav"
        fake_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")

        label, confidence = get_segment_prediction(mock_model, fake_path)

        assert label == "unknown"
        assert confidence == 0.0
        captured = capsys.readouterr()
        assert "Warning" in captured.err


# ---------------------------------------------------------------------------
# add_samples (integration-style)
# ---------------------------------------------------------------------------


class TestAddSamples:
    """Integration-style tests for add_samples."""

    def _fake_split(self, tmp_path):
        """Return a fake segments list with two pre-created files."""
        seg1 = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_02_PST.wav"
        seg2 = tmp_path / "rpi-orcasound-lab_2025_01_15_12_30_04_PST.wav"
        seg1.write_bytes(b"")
        seg2.write_bytes(b"")
        return [
            (seg1, "2025_01_15_12_30_02_PST"),
            (seg2, "2025_01_15_12_30_04_PST"),
        ]

    def test_returns_list_of_dicts_with_manual_samples_format(self, tmp_path):
        """add_samples should return list of dicts matching manual_samples.csv format."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.92
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            results = add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_02_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        assert len(results) == 2
        for row in results:
            assert isinstance(row, dict)
            assert row["Category"] == "water"
            assert row["NodeName"] == "rpi_orcasound_lab"
            assert row["StartTimestamp"] in ["2025_01_15_12_30_02_PST", "2025_01_15_12_30_04_PST"]
            assert row["URI"] == "https://example.com/test"
            assert row["Description"] == ""
            assert row["Notes"] == "manual"
            assert row["Confidence"] == "92.0"  # Confidence is percentage string (0.92 * 100 = 92.0)

    def test_training_timestamp_is_two_seconds_after_uri_time(self, tmp_path):
        segment_path = tmp_path / "rpi-test_2025_01_01_00_00_02_PST.wav"
        segment_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.92,
        }

        with patch(
            "add_samples.split_wav_into_segments",
            return_value=[(segment_path, "2025_01_01_00_00_02_PST")],
        ), patch("add_samples.lookup_detection_in_csv", return_value=None), patch(
            "add_samples.get_node_slug", return_value="test"
        ):
            rows = add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_test",
                start_timestamp="2025_01_01_00_00_02_PST",
                output_dir=str(tmp_path),
                model=mock_model,
            )

        assert rows[0]["StartTimestamp"] == "2025_01_01_00_00_02_PST"
        assert rows[0]["URI"] == (
            "https://live.orcasound.net/bouts/new/test?time=2025-01-01T08%3A00%3A00.000Z"
        )

    def test_uri_download_applies_orcasite_hls_offset_to_corrected_timestamp(self, tmp_path):
        corrected_timestamp = "2025_01_01_00_00_02_PST"
        segment_path = tmp_path / f"rpi-test_{corrected_timestamp}.wav"
        segment_path.write_bytes(b"")
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.92,
        }

        with patch("add_samples.get_node_name_from_slug", return_value="rpi_test"), patch(
            "audio_utils.download_audio_from_start_utc", return_value="fake.wav"
        ) as mock_download, patch(
            "add_samples.split_wav_into_segments",
            return_value=[(segment_path, corrected_timestamp)],
        ) as mock_split, patch("add_samples.lookup_detection_in_csv", return_value=None), patch(
            "add_samples.get_node_slug", return_value="test"
        ):
            source_uri = (
                "https://live.orcasound.net/bouts/new/test?"
                "time=2025-01-01T08%3A00%3A00.000Z"
            )
            rows = add_training_3s_samples(
                uri=source_uri,
                output_dir=str(tmp_path),
                model=mock_model,
            )

        mock_download.assert_called_once()
        assert mock_download.call_args.args[0] == "rpi_test"
        assert mock_download.call_args.args[1] == datetime(
            2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc
        )
        assert mock_download.call_args.args[3] == 60
        assert mock_split.call_args.kwargs["source_uri"] == source_uri
        assert rows[0]["StartTimestamp"] == corrected_timestamp

    def test_default_model_path_is_used(self, tmp_path):
        """add_samples should use DEFAULT_MODEL_PATH when model_path is not provided."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.75
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_02_PST",
                output_dir=str(tmp_path),
            )

        mock_get_model.assert_called_once_with(
            model_type="podsai", model_path=DEFAULT_MODEL_PATH,
            model_revision=LATEST_PODSAI_MODEL_REVISION
        )

    def test_uses_fallback_description_when_detection_missing(self, tmp_path):
        """When detections.csv has no match, fallback_description should populate Description."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.75,
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):
            results = add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model=mock_model,
                fallback_description="Moderator note about this sample.",
            )

        assert len(results) == 2
        assert all(row["Description"] == "Moderator note about this sample." for row in results)
        assert all(row["Notes"] == "manual" for row in results)

    def test_uses_fallback_notes_when_detection_missing(self, tmp_path):
        """When detections.csv has no match, fallback_notes should populate Notes."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.75,
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):
            results = add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model=mock_model,
                fallback_notes="fp_machine",
            )

        assert len(results) == len(fake_segments)
        assert all(row["Notes"] == "fp_machine" for row in results)

    def test_corrected_class_suppresses_matching_printed_rows(self, tmp_path, capsys):
        """Printed output should omit rows whose predicted class matches corrected_class."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.side_effect = [
            {"global_prediction_label": "vessel", "global_confidence": 0.75},
            {"global_prediction_label": "resident", "global_confidence": 0.80},
        ]

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):
            results = add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model=mock_model,
                corrected_class="vessel",
            )

        assert [row["Category"] for row in results] == ["vessel", "resident"]
        captured = capsys.readouterr()
        assert "vessel,rpi_orcasound_lab,2025_01_15_12_30_00_PST" not in captured.out
        assert "vessel,rpi_orcasound_lab,2025_01_15_12_30_04_PST" in captured.out

    def test_printed_rows_are_valid_csv_with_special_description(self, tmp_path, capsys):
        """Printed CSV rows should remain valid when description contains commas/newlines."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "resident",
            "global_confidence": 0.85,
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):
            add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model=mock_model,
                fallback_description="comment,with comma\nand newline",
            )

        captured = capsys.readouterr()
        parsed_rows = list(csv.reader(io.StringIO(captured.out)))
        header_row = [
            "Category",
            "NodeName",
            "StartTimestamp",
            "URI",
            "Description",
            "Notes",
            "Confidence",
            "Tags",
        ]
        header_index = parsed_rows.index(header_row)
        rows = parsed_rows[header_index + 1: header_index + 1 + len(fake_segments)]

        assert len(rows) == len(fake_segments)
        assert all(row[4] == "comment,with comma\nand newline" for row in rows)
        assert all(row[5] == "manual" for row in rows)

    def test_returns_empty_when_no_segments(self, tmp_path):
        """add_training_3s_samples should return [] if split_wav_into_segments yields nothing."""
        with patch("add_samples.split_wav_into_segments", return_value=[]):
            results = add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
            )

        assert results == []

    def test_model_loaded_once_for_all_segments(self, tmp_path):
        """The model should be loaded exactly once regardless of the number of segments."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.80
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        assert mock_get_model.call_count == 1
        assert mock_model.predict.call_count == 2

    def test_uses_preloaded_model_without_loading_again(self, tmp_path):
        """A supplied model should be reused instead of calling get_model_inference."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.80
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference") as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
                model=mock_model,
            )

        mock_get_model.assert_not_called()
        assert mock_model.predict.call_count == 2

    def test_always_uses_podsai_model_type(self, tmp_path):
        """add_training_3s_samples should always call get_model_inference with model_type='podsai'."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.70
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments), \
             patch("add_samples.get_model_inference", return_value=mock_model) as mock_get_model, \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_training_3s_samples(
                wav_file="fake.wav",
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        call_args = mock_get_model.call_args
        assert call_args[1]["model_type"] == "podsai"

    def test_infers_node_and_timestamp_from_filename(self, tmp_path):
        """add_training_3s_samples should parse node_name and start_timestamp from a well-formed filename."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.65
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments) as mock_split, \
             patch("add_samples.get_model_inference", return_value=mock_model), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_training_3s_samples(
                wav_file="rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        # Verify the inferred values were passed to split_wav_into_segments.
        call_kwargs = mock_split.call_args
        assert call_kwargs[0][1] == "rpi_orcasound_lab"         # node_name
        assert call_kwargs[0][2] == "2025_01_15_12_30_00_PST"   # start_timestamp

    def test_explicit_args_override_filename_inference(self, tmp_path):
        """Explicit node_name and start_timestamp should take precedence over filename."""
        fake_segments = self._fake_split(tmp_path)
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "global_prediction_label": "water",
            "global_confidence": 0.88
        }

        with patch("add_samples.split_wav_into_segments", return_value=fake_segments) as mock_split, \
             patch("add_samples.get_model_inference", return_value=mock_model), \
             patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):

            add_training_3s_samples(
                wav_file="rpi-orcasound-lab_2025_01_15_12_30_00_PST.wav",
                node_name="rpi_sunset_bay",
                start_timestamp="2026_06_01_00_00_00_PST",
                output_dir=str(tmp_path),
                model_path="/path/to/model",
            )

        call_kwargs = mock_split.call_args
        assert call_kwargs[0][1] == "rpi_sunset_bay"
        assert call_kwargs[0][2] == "2026_06_01_00_00_00_PST"

    def test_raises_for_bad_filename_when_no_explicit_args(self):
        """add_training_3s_samples should raise ValueError when the filename cannot be parsed and no args given."""
        with pytest.raises(ValueError, match="Cannot infer"):
            add_training_3s_samples(wav_file="recording.wav")


class TestAddTesting60sSample:
    """Tests for add_testing_60s_sample."""

    def test_uses_detection_lookup_metadata_when_available(self):
        """Rows should use description/notes/tags from detections.csv when a match exists."""
        detection_info = type(
            "DetectionInfo",
            (),
            {"description": "from detections", "notes": "from_notes", "tags": "resident;vessel"},
        )()

        with patch("add_samples.lookup_detection_in_csv", return_value=detection_info) as mock_lookup, \
             patch("add_samples.generate_uri", return_value="https://example.com/test"):
            row = add_testing_60s_sample(
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_02_PST",
                corrected_class="vessel",
                fallback_description="fallback description",
                fallback_notes="fallback notes",
                fallback_tags="fallback tags",
            )

        mock_lookup.assert_called_once_with(
            "rpi_orcasound_lab",
            "2025_01_15_12_30_02_PST",
            "output/csv/testing_60s_samples.csv",
        )
        assert row["Description"] == "from detections"
        assert row["Notes"] == "from_notes"
        assert row["Tags"] == "resident;vessel"
        assert row["Category"] == "vessel"

    def test_uses_fallback_metadata_when_detection_missing(self):
        """Rows should use fallback description/notes/tags when detections.csv has no match."""
        with patch("add_samples.lookup_detection_in_csv", return_value=None), \
             patch("add_samples.generate_uri", return_value="https://example.com/test") as mock_generate_uri:
            row = add_testing_60s_sample(
                node_name="rpi_orcasound_lab",
                start_timestamp="2025_01_15_12_30_02_PST",
                corrected_class="vessel",
                fallback_description="  fallback description  ",
                fallback_notes="  fp_machine  ",
                fallback_tags="  vessel  ",
            )

        assert row["Description"] == "fallback description"
        assert row["Notes"] == "fp_machine"
        assert row["Tags"] == "vessel"
        assert row["Category"] == "vessel"
        assert row["StartTimestamp"] == "2025_01_15_12_30_02_PST"
        mock_generate_uri.assert_called_once_with(
            "rpi_orcasound_lab", "2025_01_15_12_30_00_PST"
        )


class TestMain:
    """Tests for the add_samples CLI."""

    def test_passes_corrected_class_argument(self, tmp_path, monkeypatch):
        """The CLI should pass --class through to add_samples()."""
        wav_path = tmp_path / "input.wav"
        wav_path.write_bytes(b"wav")

        with patch("add_samples.add_training_3s_samples", return_value=[{"Category": "resident"}]) as mock_add_samples:
            monkeypatch.setattr(
                sys,
                "argv",
                ["add_samples.py", str(wav_path), "--class", "vessel"],
            )

            assert add_samples_main() == 0

        assert mock_add_samples.call_args.kwargs["corrected_class"] == "vessel"
