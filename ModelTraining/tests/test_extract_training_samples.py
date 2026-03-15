# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for the write_training_samples pipeline in extract_training_samples.py.

Tests focus on the process_sample helper, which contains the core logic for
adjusting timestamps, URIs, and confidence values for each training sample.
"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from extract_training_samples import process_sample, write_training_samples


# ---------------------------------------------------------------------------
# Test fixtures: raw input rows (as they appear in detections.csv)
# ---------------------------------------------------------------------------

TP_HUMAN_ONLY_INPUT = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_12_12_01_14_55_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-12-12T09%3A14%3A55.000Z',
    'Description': 'Humpback!',
    'Notes': 'tp_human_only',
    'Confidence': '',
}

TP_HUMAN_ONLY_EXPECTED = {
    'Category': 'humpback',
    'NodeName': 'rpi_andrews_bay',
    'Timestamp': '2025_12_12_01_14_17_PST',
    'URI': 'https://live.orcasound.net/bouts/new/andrews-bay?time=2025-12-12T09%3A14%3A17.000Z',
    'Description': 'Humpback!',
    'Notes': 'tp_human_only',
    'Confidence': '43.2',
}

TP_MACHINE_ONLY_INPUT = {
    'Category': 'humpback',
    'NodeName': 'rpi_orcasound_lab',
    'Timestamp': '2025_12_17_22_33_50_PST',
    'URI': 'https://live.orcasound.net/bouts/new/orcasound-lab?time=2025-12-18T06%3A33%3A50.000Z',
    'Description': 'Humpback whale song',
    'Notes': 'tp_machine_only',
    'Confidence': '58.3503',
}

TP_MACHINE_ONLY_EXPECTED = {
    'Category': 'humpback',
    'NodeName': 'rpi_orcasound_lab',
    'Timestamp': '2025_12_17_22_33_47_PST',
    'URI': 'https://live.orcasound.net/bouts/new/orcasound-lab?time=2025-12-18T06%3A33%3A47.000Z',
    'Description': 'Humpback whale song',
    'Notes': 'tp_machine_only',
    'Confidence': '58.4',
}


# ---------------------------------------------------------------------------
# process_sample tests
# ---------------------------------------------------------------------------

class TestProcessSampleTpMachineOnly:
    """Tests for tp_machine_only samples (fixed-offset timestamp correction)."""

    def test_timestamp_subtracted_by_segment_duration(self):
        """Timestamp should be moved back by the default segment duration (3 s)."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result['Timestamp'] == TP_MACHINE_ONLY_EXPECTED['Timestamp']

    def test_uri_updated_to_match_new_timestamp(self):
        """URI time parameter should reflect the corrected timestamp."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result['URI'] == TP_MACHINE_ONLY_EXPECTED['URI']

    def test_confidence_formatted_to_one_decimal(self):
        """Confidence value should be rounded and formatted to one decimal place."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result['Confidence'] == TP_MACHINE_ONLY_EXPECTED['Confidence']

    def test_full_output_row_matches_expected(self):
        """Complete output row should match the expected training_samples.csv row."""
        result = process_sample(TP_MACHINE_ONLY_INPUT, {}, {})
        assert result == TP_MACHINE_ONLY_EXPECTED


class TestProcessSampleTpHumanOnly:
    """Tests for tp_human_only samples (model-based timestamp correction)."""

    def _make_result(self, mock_return=('2025_12_12_01_14_17_PST', 43.2)):
        """Call process_sample with a mocked compute_correct_timestamp_for_tp_human_only."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=mock_return,
        ):
            return process_sample(
                TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=mock_model, tmp_dir='/tmp'
            )

    def test_timestamp_from_model_inference(self):
        """Timestamp should come from the model-based correction."""
        result = self._make_result()
        assert result['Timestamp'] == TP_HUMAN_ONLY_EXPECTED['Timestamp']

    def test_uri_updated_to_match_new_timestamp(self):
        """URI time parameter should reflect the model-corrected timestamp."""
        result = self._make_result()
        assert result['URI'] == TP_HUMAN_ONLY_EXPECTED['URI']

    def test_confidence_formatted_to_one_decimal(self):
        """Model confidence should be formatted to one decimal place."""
        result = self._make_result()
        assert result['Confidence'] == TP_HUMAN_ONLY_EXPECTED['Confidence']

    def test_full_output_row_matches_expected(self):
        """Complete output row should match the expected training_samples.csv row."""
        result = self._make_result()
        assert result == TP_HUMAN_ONLY_EXPECTED

    def test_model_inference_called_with_correct_args(self):
        """compute_correct_timestamp_for_tp_human_only should receive the right arguments."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=('2025_12_12_01_14_17_PST', 43.2),
        ) as mock_compute:
            process_sample(
                TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=mock_model, tmp_dir='/tmp'
            )
        mock_compute.assert_called_once_with(TP_HUMAN_ONLY_INPUT, mock_model, '/tmp', 3)

    def test_falls_back_to_fixed_offset_when_no_model(self):
        """Without model_inference, tp_human_only should use the fixed offset instead."""
        result = process_sample(TP_HUMAN_ONLY_INPUT, {}, {}, model_inference=None)
        # 2025_12_12_01_14_55 - 3 seconds = 2025_12_12_01_14_52
        assert result['Timestamp'] == '2025_12_12_01_14_52_PST'


# ---------------------------------------------------------------------------
# write_training_samples integration test
# ---------------------------------------------------------------------------

class TestWriteTrainingSamples:
    """Integration tests for write_training_samples (CSV output)."""

    def test_tp_machine_only_row_written_correctly(self):
        """write_training_samples should produce the correct CSV row for tp_machine_only."""
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / 'training_samples.csv'
            write_training_samples(
                samples=[TP_MACHINE_ONLY_INPUT],
                output_path=output_path,
                manual_timestamps={},
                manual_confidences={},
            )
            rows = _read_csv(output_path)
        assert len(rows) == 1
        assert rows[0] == TP_MACHINE_ONLY_EXPECTED

    def test_tp_human_only_row_written_correctly(self):
        """write_training_samples should produce the correct CSV row for tp_human_only."""
        mock_model = MagicMock()
        with patch(
            'extract_training_samples.compute_correct_timestamp_for_tp_human_only',
            return_value=('2025_12_12_01_14_17_PST', 43.2),
        ):
            with TemporaryDirectory() as tmp:
                output_path = Path(tmp) / 'training_samples.csv'
                write_training_samples(
                    samples=[TP_HUMAN_ONLY_INPUT],
                    output_path=output_path,
                    manual_timestamps={},
                    manual_confidences={},
                    model_inference=mock_model,
                )
                rows = _read_csv(output_path)
        assert len(rows) == 1
        assert rows[0] == TP_HUMAN_ONLY_EXPECTED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return its rows as a list of dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))
