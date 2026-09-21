# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for process_testing_set_mispredictions.py."""

import csv
from unittest.mock import Mock, patch

from process_testing_set_mispredictions import (
    build_training_row,
    find_top_non_adjacent_segments,
    triage_testing_set_mispredictions,
)


def test_find_top_non_adjacent_segments_uses_best_pair():
    """Top-confidence adjacent segments should be skipped for non-adjacent pairing."""
    segments = find_top_non_adjacent_segments(
        local_prediction_labels=["resident", "resident", "water", "resident"],
        local_confidences=[0.96, 0.95, 0.10, 0.91],
        target_label="resident",
        min_confidence=0.80,
    )

    assert segments == [(0, 0.96), (3, 0.91)]


def test_build_training_row_updates_timestamp_uri_and_confidence():
    """Training row should use segment offset for StartTimestamp and URI time parameter."""
    testing_row = {
        "Category": "bird",
        "NodeName": "rpi_test",
        "StartTimestamp": "2026_01_01_00_00_00_PST",
        "URI": "https://live.orcasound.net/bouts/new/test-slug?time=2026-01-01T08%3A00%3A00.000Z",
        "Description": "desc",
        "Notes": "fp_machine_only",
        "Confidence": "100",
    }

    row = build_training_row(
        testing_row=testing_row,
        target_label="resident",
        segment_index=2,
        segment_confidence=0.9234,
        hop_duration=2.0,
    )

    assert row["Category"] == "resident"
    assert row["StartTimestamp"] == "2026_01_01_00_00_04_PST"
    assert "time=2026-01-01T08%3A00%3A04.000Z" in row["URI"]
    assert row["Confidence"] == "92.3"


def test_triage_proposes_training_rows_and_testing_removal(tmp_path):
    """Rows with resident predictions and two qualifying non-adjacent segments are proposed."""
    testing_csv = tmp_path / "testing_60s_samples.csv"
    wav_dir = tmp_path / "testing-wav"
    category_dir = wav_dir / "bird"
    category_dir.mkdir(parents=True, exist_ok=True)

    with open(testing_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Category",
                "NodeName",
                "StartTimestamp",
                "URI",
                "Description",
                "Notes",
                "Confidence",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "Category": "bird",
                "NodeName": "rpi_test",
                "StartTimestamp": "2026_02_02_01_02_03_PST",
                "URI": "https://live.orcasound.net/bouts/new/test-slug?time=2026-02-02T09%3A02%3A03.000Z",
                "Description": "desc",
                "Notes": "fp_machine_only",
                "Confidence": "100",
            }
        )

    wav_path = category_dir / "rpi-test_2026_02_02_01_02_03_PST.wav"
    wav_path.write_bytes(b"wav")

    mock_model = Mock()
    mock_model.id2label = {0: "water", 1: "resident"}
    mock_model.predict.return_value = {
        "global_prediction_labels": ["srkw"],
        "local_predictions": [1, 1, 1],
        "local_confidences": [0.91, 0.89, 0.92],
        "hop_duration": 2.0,
    }

    with patch("model_inference.get_model_inference", return_value=mock_model):
        training_rows, testing_rows_to_remove, summary = triage_testing_set_mispredictions(
            testing_csv=testing_csv,
            wav_dir=wav_dir,
            model_path="dummy",
            min_confidence=0.80,
        )

    assert len(training_rows) == 2
    assert all(row["Category"] == "resident" for row in training_rows)
    assert {row["StartTimestamp"] for row in training_rows} == {
        "2026_02_02_01_02_07_PST",
        "2026_02_02_01_02_03_PST",
    }
    assert len(testing_rows_to_remove) == 1
    assert testing_rows_to_remove[0]["Category"] == "bird"
    assert summary["proposed_training_rows"] == 2
    assert summary["proposed_testing_removals"] == 1


def test_triage_skips_row_on_inference_exception(tmp_path, capsys):
    """Inference exceptions should be counted and reported, not crash processing."""
    testing_csv = tmp_path / "testing_60s_samples.csv"
    wav_dir = tmp_path / "testing-wav"
    category_dir = wav_dir / "bird"
    category_dir.mkdir(parents=True, exist_ok=True)

    with open(testing_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Category",
                "NodeName",
                "StartTimestamp",
                "URI",
                "Description",
                "Notes",
                "Confidence",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "Category": "bird",
                "NodeName": "rpi_test",
                "StartTimestamp": "2026_02_02_01_02_03_PST",
                "URI": "https://live.orcasound.net/bouts/new/test-slug?time=2026-02-02T09%3A02%3A03.000Z",
                "Description": "desc",
                "Notes": "fp_machine_only",
                "Confidence": "100",
            }
        )

    wav_path = category_dir / "rpi-test_2026_02_02_01_02_03_PST.wav"
    wav_path.write_bytes(b"wav")

    mock_model = Mock()
    mock_model.id2label = {0: "water", 1: "resident"}
    mock_model.predict.side_effect = RuntimeError("model failed")

    with patch("model_inference.get_model_inference", return_value=mock_model):
        training_rows, testing_rows_to_remove, summary = triage_testing_set_mispredictions(
            testing_csv=testing_csv,
            wav_dir=wav_dir,
            model_path="dummy",
            min_confidence=0.80,
        )

    captured = capsys.readouterr()
    assert "inference failed for" in captured.err
    assert summary["inference_errors"] == 1
    assert training_rows == []
    assert testing_rows_to_remove == []


def test_triage_skips_row_on_mismatched_inference_lengths(tmp_path, capsys):
    """Mismatched local prediction/confidence lengths should be treated as an error."""
    testing_csv = tmp_path / "testing_60s_samples.csv"
    wav_dir = tmp_path / "testing-wav"
    category_dir = wav_dir / "bird"
    category_dir.mkdir(parents=True, exist_ok=True)

    with open(testing_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Category",
                "NodeName",
                "StartTimestamp",
                "URI",
                "Description",
                "Notes",
                "Confidence",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "Category": "bird",
                "NodeName": "rpi_test",
                "StartTimestamp": "2026_02_02_01_02_03_PST",
                "URI": "https://live.orcasound.net/bouts/new/test-slug?time=2026-02-02T09%3A02%3A03.000Z",
                "Description": "desc",
                "Notes": "fp_machine_only",
                "Confidence": "100",
            }
        )

    wav_path = category_dir / "rpi-test_2026_02_02_01_02_03_PST.wav"
    wav_path.write_bytes(b"wav")

    mock_model = Mock()
    mock_model.id2label = {0: "water", 1: "resident"}
    mock_model.predict.return_value = {
        "global_prediction_labels": ["resident"],
        "local_predictions": [1, 1, 1],
        "local_confidences": [0.91, 0.89],
        "hop_duration": 2.0,
    }

    with patch("model_inference.get_model_inference", return_value=mock_model):
        training_rows, testing_rows_to_remove, summary = triage_testing_set_mispredictions(
            testing_csv=testing_csv,
            wav_dir=wav_dir,
            model_path="dummy",
            min_confidence=0.80,
        )

    captured = capsys.readouterr()
    assert "mismatched inference lengths" in captured.err
    assert summary["inference_errors"] == 1
    assert training_rows == []
    assert testing_rows_to_remove == []
