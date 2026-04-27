#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for compare_models.py.

Tests cover:
- load_testing_samples() CSV parsing
- find_wav_file() path construction
- is_resident_prediction() label mapping
- evaluate_model() with mocked run_inference
- print_summary() output
- ModelResult property calculations
- main() CLI error handling
"""

import csv
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_testing_csv(path: Path, rows: list[dict]) -> None:
    """Write a testing_samples.csv file with the given rows."""
    fieldnames = ["Category", "NodeName", "Timestamp", "URI", "Description", "Notes", "Confidence"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_sample_rows() -> list[dict]:
    """Return a list of sample rows for testing."""
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
# Tests for load_testing_samples()
# ---------------------------------------------------------------------------

class TestLoadTestingSamples:
    """Tests for load_testing_samples()."""

    def test_returns_correct_number_of_samples(self, tmp_path):
        """load_testing_samples returns one TestSample per data row."""
        from compare_models import load_testing_samples
        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, _make_sample_rows())
        samples = load_testing_samples(csv_path)
        assert len(samples) == 3

    def test_parses_fields_correctly(self, tmp_path):
        """load_testing_samples correctly maps CSV columns to TestSample fields."""
        from compare_models import load_testing_samples
        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, _make_sample_rows())
        samples = load_testing_samples(csv_path)

        first = samples[0]
        assert first.category == "resident"
        assert first.node_name == "rpi_orcasound_lab"
        assert first.timestamp == "2023_08_18_00_59_53_PST"
        assert first.notes == "tp_human_only"

    def test_returns_empty_list_for_missing_file(self):
        """load_testing_samples returns [] and prints an error for a missing file."""
        from compare_models import load_testing_samples
        samples = load_testing_samples(Path("/nonexistent/path/testing_samples.csv"))
        assert samples == []

    def test_returns_empty_list_for_header_only_csv(self, tmp_path):
        """load_testing_samples returns [] when the CSV contains only a header."""
        from compare_models import load_testing_samples
        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, [])
        samples = load_testing_samples(csv_path)
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
        # Create the expected WAV file.
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
        # File should use dashes in node name.
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

    def test_fastai_whale_is_resident(self):
        """FastAI 'whale' prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("whale", "fastai") is True

    def test_fastai_other_is_not_resident(self):
        """FastAI 'other' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("other", "fastai") is False

    def test_orcahello_whale_is_resident(self):
        """OrcaHello 'whale' prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("whale", "orcahello") is True

    def test_orcahello_other_is_not_resident(self):
        """OrcaHello 'other' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("other", "orcahello") is False

    def test_huggingface_resident_is_resident(self):
        """HuggingFace 'resident' prediction maps to resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("resident", "huggingface") is True

    def test_huggingface_water_is_not_resident(self):
        """HuggingFace 'water' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("water", "huggingface") is False

    def test_huggingface_transient_is_not_resident(self):
        """HuggingFace 'transient' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("transient", "huggingface") is False

    def test_huggingface_humpback_is_not_resident(self):
        """HuggingFace 'humpback' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("humpback", "huggingface") is False

    def test_huggingface_human_is_not_resident(self):
        """HuggingFace 'human' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("human", "huggingface") is False

    def test_huggingface_vessel_is_not_resident(self):
        """HuggingFace 'vessel' prediction maps to non-resident."""
        from compare_models import is_resident_prediction
        assert is_resident_prediction("vessel", "huggingface") is False


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

    def _make_wav_files(self, tmp_path: Path, samples) -> Path:
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
        """A resident sample predicted as 'whale' (fastai) counts as correct."""
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

        mock_result = {"global_prediction_label": "whale", "global_confidence": 0.8}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.skipped == 0

    def test_false_positive_counted(self, tmp_path):
        """A non-resident sample predicted as 'whale' (fastai) counts as false positive."""
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

        mock_result = {"global_prediction_label": "whale", "global_confidence": 0.7}
        with patch("compare_models.run_inference", return_value=mock_result):
            result = evaluate_model("fastai", "./model", [sample], wav_dir)

        assert result.correct == 0
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_false_negative_counted(self, tmp_path):
        """A resident sample predicted as 'other' (fastai) counts as false negative."""
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

    def test_huggingface_resident_prediction_correct(self, tmp_path):
        """HuggingFace 'resident' prediction for a resident sample counts as correct."""
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
            result = evaluate_model("huggingface", "/path/to/model", [sample], wav_dir)

        assert result.correct == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_huggingface_water_prediction_correct_for_non_resident(self, tmp_path):
        """HuggingFace 'water' prediction for a non-resident sample counts as correct."""
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
            result = evaluate_model("huggingface", "/path/to/model", [sample], wav_dir)

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
        """print_summary prints the 'Model Comparison Summary' header."""
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

    def test_returns_1_for_missing_testing_csv(self, tmp_path):
        """main() returns 1 when the testing CSV file does not exist."""
        from compare_models import main
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()
        test_args = [
            "compare_models.py",
            "--testing-csv", str(tmp_path / "nonexistent.csv"),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_missing_wav_dir(self, tmp_path):
        """main() returns 1 when the WAV directory does not exist."""
        from compare_models import main

        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, _make_sample_rows())

        test_args = [
            "compare_models.py",
            "--testing-csv", str(csv_path),
            "--wav-dir", str(tmp_path / "nonexistent-wav-dir"),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_unknown_model(self, tmp_path):
        """main() returns 1 when an unrecognised model type is specified."""
        from compare_models import main

        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, _make_sample_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        test_args = [
            "compare_models.py",
            "--testing-csv", str(csv_path),
            "--wav-dir", str(wav_dir),
            "--models", "unknown_model",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_1_for_huggingface_without_model_path(self, tmp_path):
        """main() returns 1 when 'huggingface' is in --models but --huggingface-model-path is absent."""
        from compare_models import main

        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, _make_sample_rows())
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        test_args = [
            "compare_models.py",
            "--testing-csv", str(csv_path),
            "--wav-dir", str(wav_dir),
            "--models", "huggingface",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1

    def test_returns_0_on_success_with_fastai(self, tmp_path):
        """main() returns 0 when it successfully evaluates fastai on a test set."""
        from compare_models import main, TestSample

        rows = [_make_sample_rows()[0]]  # One resident sample.
        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, rows)

        wav_dir = tmp_path / "testing-wav"
        sample = TestSample(
            category=rows[0]["Category"],
            node_name=rows[0]["NodeName"],
            timestamp=rows[0]["Timestamp"],
            uri="",
            description="",
            notes=rows[0]["Notes"],
        )
        node_name_in_filename = sample.node_name.replace("_", "-")
        wav_file = wav_dir / sample.category / f"{node_name_in_filename}_{sample.timestamp}.wav"
        wav_file.parent.mkdir(parents=True)
        wav_file.touch()

        mock_result = {"global_prediction_label": "whale", "global_confidence": 0.8}
        test_args = [
            "compare_models.py",
            "--testing-csv", str(csv_path),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
            "--fastai-model-path", "./model",
        ]
        with patch.object(sys, "argv", test_args):
            with patch("compare_models.run_inference", return_value=mock_result):
                result = main()
        assert result == 0

    def test_returns_1_for_empty_csv(self, tmp_path):
        """main() returns 1 when testing_samples.csv has no data rows."""
        from compare_models import main

        csv_path = tmp_path / "testing_samples.csv"
        _write_testing_csv(csv_path, [])
        wav_dir = tmp_path / "testing-wav"
        wav_dir.mkdir()

        test_args = [
            "compare_models.py",
            "--testing-csv", str(csv_path),
            "--wav-dir", str(wav_dir),
            "--models", "fastai",
        ]
        with patch.object(sys, "argv", test_args):
            result = main()
        assert result == 1
