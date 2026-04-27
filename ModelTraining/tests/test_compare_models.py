#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for compare_models.py.

Tests cover:
- derive_test_samples() CSV parsing and exclusion logic
- find_wav_file() path construction
- is_resident_prediction() label mapping
- evaluate_model() with mocked run_inference
- print_summary() output
- ModelResult property calculations
- main() CLI error handling
"""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, rows, fieldnames=None):
    """Write a CSV file with the given rows."""
    if not fieldnames:
        fieldnames = ["Category", "NodeName", "Timestamp", "URI", "Description", "Notes", "Confidence"]
    with open(path, "w", newline="") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_detection_rows():
    """Return a list of detection rows for testing."""
    return [
        {
            "Category": "resident",
            "NodeName": "rpi_orcasound_lab",
            "Timestamp": "2023_08_18_00_59_53_PST",
            "URI": "https://example.com/1",
            "Description": "J pod calls",
            "Notes": "tp_human_only",
            "Confidence": "",
        },
        {
            "Category": "human",
            "NodeName": "rpi_sunset_bay",
            "Timestamp": "2024_08_07_11_23_23_PST",
            "URI": "https://example.com/2",
            "Description": "Human voices",
            "Notes": "fp_machine_only",
            "Confidence": "62.3839",
        },
        {
            "Category": "humpback",
            "NodeName": "rpi_orcasound_lab",
            "Timestamp": "2023_10_28_07_33_52_PST",
            "URI": "https://example.com/3",
            "Description": "Humpback",
            "Notes": "tp_human_only",
            "Confidence": "",
        },
    ]


# ---------------------------------------------------------------------------
# Tests for derive_test_samples()
# ---------------------------------------------------------------------------

class TestDeriveTestSamples:
    """Tests for derive_test_samples()."""

    def test_excludes_training_uris(self, tmp_path):
        """derive_test_samples excludes rows whose URI appears in training_samples.csv."""
        from compare_models import derive_test_samples
        detections = _make_detection_rows()
        training_rows = [detections[0]]

        detections_csv = tmp_path / "detections.csv"
        training_csv = tmp_path / "training_samples.csv"
        _write_csv(detections_csv, detections)
        _write_csv(training_csv, training_rows)

        samples = derive_test_samples(detections_csv, training_csv)
        assert len(samples) == 2
        uris = {s.uri for s in samples}
        assert "https://example.com/1" not in uris
        assert "https://example.com/2" in uris
        assert "https://example.com/3" in uris

    def test_returns_all_detections_when_no_training_overlap(self, tmp_path):
        """derive_test_samples returns all detections when training has different URIs."""
        from compare_models import derive_test_samples
        detections = _make_detection_rows()
        training_rows = [{"URI": "https://example.com/999"}]

        detections_csv = tmp_path / "detections.csv"
        training_csv = tmp_path / "training_samples.csv"
        _write_csv(detections_csv, detections)
        _write_csv(training_csv, training_rows)

        samples = derive_test_samples(detections_csv, training_csv)
        assert len(samples) == len(detections)

    def test_parses_fields_correctly(self, tmp_path):
        """derive_test_samples correctly maps CSV columns to TestSample fields."""
        from compare_models import derive_test_samples
        detections = _make_detection_rows()

        detections_csv = tmp_path / "detections.csv"
        training_csv = tmp_path / "training_samples.csv"
        _write_csv(detections_csv, detections)
        _write_csv(training_csv, [])

        samples = derive_test_samples(detections_csv, training_csv)
        first = samples[0]
        assert first.category == "resident"
        assert first.node_name == "rpi_orcasound_lab"
        assert first.timestamp == "2023_08_18_00_59_53_PST"
        assert first.uri == "https://example.com/1"
        assert first.notes == "tp_human_only"

    def test_returns_empty_list_for_missing_detections_file(self, tmp_path):
        """derive_test_samples returns [] when detections.csv is missing."""
        from compare_models import derive_test_samples
        training_csv = tmp_path / "training_samples.csv"
        _write_csv(training_csv, [])

        samples = derive_test_samples(Path("/nonexistent/detections.csv"), training_csv)
        assert samples == []

    def test_returns_all_when_training_file_missing(self, tmp_path):
        """derive_test_samples returns all detections when training_samples.csv is missing."""
        from compare_models import derive_test_samples
        detections_csv = tmp_path / "detections.csv"
        _write_csv(detections_csv, _make_detection_rows())

        samples = derive_test_samples(detections_csv, Path("/nonexistent/training_samples.csv"))
        assert len(samples) == len(_make_detection_rows())

    def test_returns_empty_when_all_detections_in_training(self, tmp_path):
        """derive_test_samples returns [] when all detections are in training."""
        from compare_models import derive_test_samples
        detections = _make_detection_rows()

        detections_csv = tmp_path / "detections.csv"
        training_csv = tmp_path / "training_samples.csv"
        _write_csv(detections_csv, detections)
        _write_csv(training_csv, detections)

        samples = derive_test_samples(detections_csv, training_csv)
        assert samples == []


# ---------------------------------------------------------------------------
# Tests for find_wav_file()
# ---------------------------------------------------------------------------

class TestFindWavFile:
    """Tests for find_wav_file()."""

    def test_returns_path_when_wav_exists(self, tmp_path):
        """find_wav_file returns the correct path when the WAV file is present."""
        from compare_models import TestSample, find_wav_file

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = tmp_path / "testing-wav"
        expected = wav_dir / "resident" / "rpi-orcasound-lab_2023_08_18_00_59_53_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)
        assert result == expected

    def test_returns_none_when_wav_missing(self, tmp_path):
        """find_wav_file returns None when the WAV file does not exist."""
        from compare_models import TestSample, find_wav_file

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        result = find_wav_file(sample, wav_dir)
        assert result is None

    def test_replaces_underscores_with_dashes_in_node_name(self, tmp_path):
        """find_wav_file converts underscores to dashes in the node name portion of the filename."""
        from compare_models import TestSample, find_wav_file

        sample = TestSample(
            category="human",
            node_name="rpi_sunset_bay",
            timestamp="2024_08_07_11_23_23_PST",
            uri="",
            description="",
            notes="fp_machine_only",
        )
        wav_dir = tmp_path / "testing-wav"
        expected = wav_dir / "human" / "rpi-sunset-bay_2024_08_07_11_23_23_PST.wav"
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = find_wav_file(sample, wav_dir)
        assert result == expected


# ---------------------------------------------------------------------------
# Tests for is_resident_prediction()
# ---------------------------------------------------------------------------

class TestIsResidentPrediction:
    """Tests for is_resident_prediction()."""

    def test_fastai_resident_is_resident(self):
        """FastAI "resident" prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "fastai") is True

    def test_fastai_other_is_not_resident(self):
        """FastAI "other" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("other", "fastai") is False

    def test_orcahello_resident_is_resident(self):
        """OrcaHello "resident" prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "orcahello") is True

    def test_orcahello_other_is_not_resident(self):
        """OrcaHello "other" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("other", "orcahello") is False

    def test_podsai_resident_is_resident(self):
        """PODS-AI "resident" prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "podsai") is True

    def test_podsai_water_is_not_resident(self):
        """PODS-AI "water" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("water", "podsai") is False

    def test_podsai_transient_is_not_resident(self):
        """PODS-AI "transient" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("transient", "podsai") is False

    def test_podsai_humpback_is_not_resident(self):
        """PODS-AI "humpback" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("humpback", "podsai") is False

    def test_podsai_human_is_not_resident(self):
        """PODS-AI "human" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("human", "podsai") is False

    def test_podsai_vessel_is_not_resident(self):
        """PODS-AI "vessel" prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("vessel", "podsai") is False


# ---------------------------------------------------------------------------
# Tests for ModelResult properties
# ---------------------------------------------------------------------------

class TestModelResultProperties:
    """Tests for ModelResult computed properties."""

    def test_evaluated_excludes_skipped(self):
        """evaluated = total - skipped."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, skipped=3)
        assert r.evaluated == 7

    def test_accuracy_none_when_no_evaluated(self):
        """accuracy is None when evaluated == 0."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=5, skipped=5)
        assert r.accuracy is None

    def test_accuracy_correct_fraction(self):
        """accuracy is correct/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, correct=8, skipped=0)
        assert abs(r.accuracy - 0.8) < 1e-9

    def test_false_positive_rate_none_when_no_evaluated(self):
        """false_positive_rate is None when evaluated == 0."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=3, skipped=3)
        assert r.false_positive_rate is None

    def test_false_positive_rate_correct_fraction(self):
        """false_positive_rate is false_positives/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, false_positives=2, skipped=0)
        assert abs(r.false_positive_rate - 0.2) < 1e-9

    def test_false_negative_rate_correct_fraction(self):
        """false_negative_rate is false_negatives/evaluated."""
        from compare_models import ModelResult
        r = ModelResult(model_type="fastai", total=10, false_negatives=1, skipped=0)
        assert abs(r.false_negative_rate - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# Tests for evaluate_model()
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    """Tests for evaluate_model() with mocked run_inference."""

    def _make_wav_files(self, tmp_path, samples):
        """Create dummy WAV files for the given samples under tmp_path/testing-wav."""
        wav_dir = tmp_path / "testing-wav"
        for sample in samples:
            node_name_in_filename = sample.node_name.replace("_", "-")
            wav_filename = f"{node_name_in_filename}_{sample.timestamp}.wav"
            wav_file = wav_dir / sample.category / wav_filename
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            wav_file.touch()
        return wav_dir

    def test_correct_resident_prediction_counted(self, tmp_path):
        """A resident sample predicted as "resident" (fastai) counts as correct."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.skipped == 0

    def test_false_positive_counted(self, tmp_path):
        """A non-resident sample predicted as "resident" (fastai) counts as false positive."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="human",
            node_name="rpi_sunset_bay",
            timestamp="2024_08_07_11_23_23_PST",
            uri="",
            description="",
            notes="fp_machine_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.7}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 0
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_false_negative_counted(self, tmp_path):
        """A resident sample predicted as "other" (fastai) counts as false negative."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.1}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 0
        assert result.false_positives == 0
        assert result.false_negatives == 1

    def test_skips_sample_when_wav_missing(self, tmp_path):
        """Samples whose WAV file is missing are counted as skipped."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        with patch("compare_models.run_inference") as mock_infer:
            result = evaluate_model("fastai", "./model", [sample], wav_dir)
            mock_infer.assert_not_called()

        assert result.skipped == 1
        assert result.correct == 0

    def test_skips_sample_on_inference_error(self, tmp_path):
        """Samples that raise an exception during inference are counted as skipped."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        with patch("compare_models.run_inference", side_effect=RuntimeError("model error")):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.skipped == 1
        assert result.correct == 0

    def test_podsai_resident_prediction_correct(self, tmp_path):
        """PODS-AI "resident" prediction for a resident sample counts as correct."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="resident",
            node_name="rpi_orcasound_lab",
            timestamp="2023_08_18_00_59_53_PST",
            uri="",
            description="",
            notes="tp_human_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.9}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_podsai_water_prediction_correct_for_non_resident(self, tmp_path):
        """PODS-AI "water" prediction for a non-resident sample counts as correct."""
        from compare_models import TestSample, evaluate_model

        sample = TestSample(
            category="human",
            node_name="rpi_sunset_bay",
            timestamp="2024_08_07_11_23_23_PST",
            uri="",
            description="",
            notes="fp_machine_only",
        )
        wav_dir = self._make_wav_files(tmp_path, [sample])

        mock_result = {"global_prediction_label": "water", "global_confidence": 0.8}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("podsai", "/path/to/model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_total_equals_sample_count(self, tmp_path):
        """ModelResult.total always equals the number of samples passed."""
        from compare_models import TestSample, evaluate_model

        samples = [
            TestSample("resident", "rpi_orcasound_lab", "2023_08_18_00_59_53_PST", "", "", "tp_human_only"),
            TestSample("human", "rpi_sunset_bay", "2024_08_07_11_23_23_PST", "", "", "fp_machine_only"),
        ]
        wav_dir = self._make_wav_files(tmp_path, samples)

        mock_result = {"global_prediction_label": "other", "global_confidence": 0.1}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", samples, wav_dir)

        assert result.total == 2


# ---------------------------------------------------------------------------
# Tests for print_summary()
# ---------------------------------------------------------------------------

class TestPrintSummary:
    """Tests for print_summary() output formatting."""

    def test_prints_header_and_separator(self, capsys):
        """print_summary prints the "Model Comparison Summary" header."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Model Comparison Summary" in captured

    def test_prints_model_name(self, capsys):
        """print_summary includes the model type in the output."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="orcahello", total=5, correct=3, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "orcahello" in captured

    def test_prints_accuracy_percentage(self, capsys):
        """print_summary shows accuracy as a percentage."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=10, correct=8, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "80.0%" in captured

    def test_prints_na_when_no_evaluated_samples(self, capsys):
        """print_summary shows N/A for accuracy when all samples are skipped."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, skipped=5)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "N/A" in captured

    def test_prints_skipped_count_when_nonzero(self, capsys):
        """print_summary shows the skipped count when some samples were skipped."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=10, correct=5, skipped=3)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "3 skipped" in captured

    def test_prints_definitions(self, capsys):
        """print_summary includes the definitions block."""
        from compare_models import ModelResult, print_summary
        results = [ModelResult(model_type="fastai", total=5, correct=4, skipped=0)]
        print_summary(results)
        captured = capsys.readouterr().out
        assert "Definitions:" in captured
        assert "false+" in captured or "FP" in captured


# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Tests for the main() entry point."""

    def _write_csvs(self, tmp_path, detection_rows, training_rows=None):
        """Write detections.csv and training_samples.csv to tmp_path."""
        det_csv = tmp_path / "detections.csv"
        train_csv = tmp_path / "training_samples.csv"
        _write_csv(det_csv, detection_rows)
        _write_csv(train_csv, training_rows or [])
        return det_csv, train_csv

    def test_returns_1_for_missing_detections_csv(self, tmp_path):
        """main() returns 1 when detections.csv does not exist."""
        from compare_models import main
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        training_csv = tmp_path / "training_samples.csv"
        _write_csv(training_csv, [])
        test_args = [
            "compare_models.py",
            "--detections-csv", str(tmp_path / "nonexistent.csv"),
            "--training-csv", str(training_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_missing_training_csv(self, tmp_path):
        """main() returns 1 when training_samples.csv does not exist."""
        from compare_models import main
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        det_csv = tmp_path / "detections.csv"
        _write_csv(det_csv, _make_detection_rows())
        test_args = [
            "compare_models.py",
            "--detections-csv", str(det_csv),
            "--training-csv", str(tmp_path / "nonexistent_training.csv"),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_missing_wav_dir(self, tmp_path):
        """main() returns 1 when the WAV directory does not exist."""
        from compare_models import main

        det_csv, train_csv = self._write_csvs(tmp_path, _make_detection_rows())
        test_args = [
            "compare_models.py",
            "--detections-csv", str(det_csv),
            "--training-csv", str(train_csv),
            "--wav-dir", str(tmp_path / "nonexistent-wav-dir"),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_unknown_model(self, tmp_path):
        """main() returns 1 when an unrecognised model type is specified."""
        from compare_models import main

        det_csv, train_csv = self._write_csvs(tmp_path, _make_detection_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--detections-csv", str(det_csv),
            "--training-csv", str(train_csv),
            "--wav-dir", str(wav_dir),
            "--models", "unknown_model",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_podsai_without_model_path(self, tmp_path):
        """main() returns 1 when podsai is in --models but --podsai-model-path is absent."""
        from compare_models import main

        det_csv, train_csv = self._write_csvs(tmp_path, _make_detection_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--detections-csv", str(det_csv),
            "--training-csv", str(train_csv),
            "--wav-dir", str(wav_dir),
            "--models", "podsai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_0_on_success_with_fastai(self, tmp_path):
        """main() returns 0 when it successfully evaluates fastai on a derived test set."""
        from compare_models import main

        rows = _make_detection_rows()
        # One training row excluded; the other two form the test set.
        det_csv, train_csv = self._write_csvs(tmp_path, rows, training_rows=[rows[1]])

        wav_dir = tmp_path / "testing-wav"
        for row in [rows[0], rows[2]]:
            node = row["NodeName"].replace("_", "-")
            wav = wav_dir / row["Category"] / f"{node}_{row['Timestamp']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.touch()

        mock_result = {"global_prediction_label": "resident", "global_confidence": 0.8}
        test_args = [
            "compare_models.py",
            "--detections-csv", str(det_csv),
            "--training-csv", str(train_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--fastai-model-path", "./model",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.run_inference", return_value=mock_result):
                result = main()
        assert result == 0

    def test_returns_1_when_all_detections_are_in_training(self, tmp_path):
        """main() returns 1 when all detections are used for training (empty test set)."""
        from compare_models import main

        rows = _make_detection_rows()
        det_csv, train_csv = self._write_csvs(tmp_path, rows, training_rows=rows)
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--detections-csv", str(det_csv),
            "--training-csv", str(train_csv),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1
