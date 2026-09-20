# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for testing sample download logic in download_wavs.py."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, Mock, patch
import os

import pytest

from download_wavs import (
    CSVRow,
    add_seconds_to_timestamp_pst,
    download_testing_sample,
    is_external_humpback_training_wav,
    process_csv,
    process_testing_csv,
    run_download_wavs,
    validate_aligned_entries,
    validate_no_overlaps,
)


class TestDownloadTestingSample:
    """Tests for download_testing_sample routing behavior."""

    def test_tp_human_only_downloads_60s_audio_to_testing_directory(self):
        """tp_human_only rows should use download_60s_audio and save output file."""
        row = CSVRow(
            category="resident",
            node_name="rpi_andrews_bay",
            timestamp_pst="2025_01_01_00_00_00_PST",
            uri="https://example.org/sample",
            description="sample",
            notes="tp_human_only",
        )

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "testing-wav"

            def _fake_download_60s_audio(node_name: str, min_end_timestamp_pst_str: str, tmp_dir: str):
                """Create and return a temporary fake 60-second WAV path."""
                wav_path = Path(tmp_dir) / "temp_60s.wav"
                wav_path.write_bytes(b"fake wav content")
                return str(wav_path)

            with patch("download_wavs.download_60s_audio", side_effect=_fake_download_60s_audio) as mock_download_60s:
                download_testing_sample(row, output_root)

            mock_download_60s.assert_called_once_with(
                node_name="rpi_andrews_bay",
                min_end_timestamp_pst_str="2025_01_01_00_01_00_PST",
                tmp_dir=ANY
            )
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert expected.exists()

    def test_tp_machine_only_downloads_60s_clip_from_start_timestamp(self):
        """tp_machine_only rows should use download_60s_audio with a +60s end timestamp."""
        row = CSVRow(
            category="humpback",
            node_name="rpi_orcasound_lab",
            timestamp_pst="2025_01_01_00_00_03_PST",
            uri="https://example.org/sample",
            description="sample",
            notes="tp_machine_only",
        )

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "testing-wav"
            def _fake_download_60s_audio(node_name: str, min_end_timestamp_pst_str: str, tmp_dir: str):
                """Create and return a temporary fake 60-second WAV path."""
                wav_path = Path(tmp_dir) / "temp_60s.wav"
                wav_path.write_bytes(b"fake wav content")
                return str(wav_path)

            with patch("download_wavs.download_60s_audio", side_effect=_fake_download_60s_audio) as mock_download_60s:
                download_testing_sample(row, output_root)

            mock_download_60s.assert_called_once_with(
                node_name="rpi_orcasound_lab",
                min_end_timestamp_pst_str="2025_01_01_00_01_03_PST",
                tmp_dir=ANY
            )
            expected = output_root / "humpback" / "rpi-orcasound-lab_2025_01_01_00_00_03_PST.wav"
            assert expected.exists()


class TestTimestampHelpers:
    """Tests for timestamp conversion helpers."""

    def test_add_seconds_to_timestamp_pst_adds_30_seconds(self):
        """add_seconds_to_timestamp_pst should add requested seconds in PST format."""
        assert add_seconds_to_timestamp_pst("2025_01_01_00_00_03_PST", 30) == "2025_01_01_00_00_33_PST"

    def test_add_seconds_to_timestamp_pst_adds_60_seconds(self):
        """add_seconds_to_timestamp_pst should support adding 60 seconds."""
        assert add_seconds_to_timestamp_pst("2025_01_01_00_00_03_PST", 60) == "2025_01_01_00_01_03_PST"


class TestOverlapValidation:
    def test_validate_no_overlaps_allows_non_overlapping_rows(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_00_00_PST", "", "", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_00_03_PST", "", "", ""),
        ]
        testing_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_59_00_PST", "", "", "tp_human_only"),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_01_01_06_PST", "", "", "tp_human_only"),
        ]
        validate_no_overlaps(training_rows, testing_rows)

    def test_validate_no_overlaps_rejects_training_overlap(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_00_PST", "", "", ""),
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_02_PST", "", "", ""),
        ]
        with pytest.raises(ValueError, match="training overlap"):
            validate_no_overlaps(training_rows, [])

    def test_validate_no_overlaps_rejects_cross_file_overlap(self):
        training_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2025_01_01_00_00_00_PST", "", "", ""),
        ]
        testing_rows = [
            CSVRow("resident", "rpi_andrews_bay", "2024_12_31_23_59_58_PST", "", "", "tp_machine_only"),
        ]
        with pytest.raises(ValueError, match="cross-file overlap"):
            validate_no_overlaps(training_rows, testing_rows)


class TestAlignedEntryValidation:
    @staticmethod
    def _mock_detection_response(payload):
        response = Mock()
        response.text = "[]"
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        if payload:
            response.text = "[{\"id\":\"1\"}]"
        return response

    def test_validate_aligned_entries_handles_paginated_items_payload(self):
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2025_12_01_00_00_00_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-12-01T08%3A00%3A00.000Z",
                "Radio",
                "fp_machine_only",
            ),
        ]
        first_page_payload = {
            "items": [
                {
                    "timestamp": "2025-12-01T07:00:00Z",
                    "comments": "not a match",
                    "found": "No",
                    "reviewed": False,
                }
                for _ in range(50)
            ],
        }
        second_page_payload = [
            {
                "timestamp": "2025-12-01T08:00:00Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
        ]

        with patch(
            "download_wavs.requests.get",
            side_effect=[
                self._mock_detection_response(first_page_payload),
                self._mock_detection_response(second_page_payload),
            ],
        ) as mock_get, \
                patch("download_wavs.get_cached_folders", side_effect=AssertionError("should not query S3 for current epoch")):
            validate_aligned_entries(testing_rows)

        assert mock_get.call_count == 2

    def test_validate_aligned_entries_rejects_old_epoch_misalignment(self):
        testing_rows = [
            CSVRow(
                "human",
                "rpi_sunset_bay",
                "2025_01_01_00_11_05_PST",
                "https://live.orcasound.net/bouts/new/sunset-bay?time=2025-01-01T08%3A11%3A05.000Z",
                "Radio",
                "fp_machine_only",
            ),
        ]
        detections = [
            {
                "timestamp": "2025-01-01T08:11:05Z",
                "comments": "Radio",
                "found": "No",
                "reviewed": True,
            },
        ]
        folder_time = int(datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp())

        with patch("download_wavs.requests.get", return_value=self._mock_detection_response(detections)), \
                patch("download_wavs.get_cached_folders", return_value=[str(folder_time)]):
            with pytest.raises(ValueError, match="corrected testing_row: CSVRow\\(category='human', node_name='rpi_sunset_bay', timestamp_pst='2025_01_01_00_10_02_PST'"):
                validate_aligned_entries(testing_rows)


class TestCacheAndCleanup:
    def test_process_csv_copies_from_cache_without_downloading(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            cache_root = tmp_path / "cache-wav"
            cached_file = cache_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"cached")

            with patch("download_wavs.get_cached_folders", side_effect=AssertionError("should not download")):
                process_csv(csv_path, output_root, cache_root=cache_root)

            downloaded_file = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert downloaded_file.exists()
            assert downloaded_file.read_bytes() == b"cached"

    def test_process_csv_deletes_stale_wavs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            stale = output_root / "resident" / "old.wav"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"keep")
            stale.write_bytes(b"remove")

            process_csv(csv_path, output_root)

            assert expected.exists()
            assert not stale.exists()

    def test_process_csv_keeps_humpback_signal_wavs_and_rejects_noise(self):
        """signals-humpback segments stay in training; unlabeled noise is deleted."""
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "training_3s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-wav"
            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            humpback_signal = (
                output_root / "humpback" / "signals-humpback_song_0000s.wav"
            )
            noise = output_root / "humpback" / "random_noise.wav"
            expected.parent.mkdir(parents=True, exist_ok=True)
            humpback_signal.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"keep")
            humpback_signal.write_bytes(b"humpback")
            noise.write_bytes(b"noise")

            process_csv(csv_path, output_root)

            assert expected.exists()
            assert humpback_signal.exists()
            assert not noise.exists()

    def test_is_external_humpback_training_wav_matches_signal_segments(self):
        assert is_external_humpback_training_wav(
            Path("humpback") / "signals-humpback_song_0000s.wav"
        )
        assert not is_external_humpback_training_wav(
            Path("humpback") / "rpi-orcasound-lab_2025_01_01_00_00_00_PST.wav"
        )
        assert not is_external_humpback_training_wav(
            Path("water") / "signals-humpback_song_0000s.wav"
        )

    def test_process_testing_csv_copies_from_cache_and_deletes_stale(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "testing_60s_samples.csv"
            csv_path.write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_00_00_00_PST,uri,desc,tp_human_only\n",
                encoding="utf-8",
            )

            output_root = tmp_path / "output-testing-wav"
            stale = output_root / "resident" / "old.wav"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_bytes(b"remove")

            cache_root = tmp_path / "cache-testing-wav"
            cached_file = cache_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            cached_file.write_bytes(b"cached")

            with patch("download_wavs.download_60s_audio", side_effect=AssertionError("should not download")):
                process_testing_csv(csv_path, output_root, cache_root=cache_root)

            expected = output_root / "resident" / "rpi-andrews-bay_2025_01_01_00_00_00_PST.wav"
            assert expected.exists()
            assert expected.read_bytes() == b"cached"
            assert not stale.exists()


class TestValidateOnly:
    def test_run_download_wavs_validate_only_skips_download_processing(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_dir = tmp_path / "output" / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            (csv_dir / "training_3s_samples.csv").write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_01_00_00_PST,uri,desc,note\n",
                encoding="utf-8",
            )
            (csv_dir / "testing_60s_samples.csv").write_text(
                "category,node_name,timestamp_pst,uri,description,notes\n"
                "resident,rpi_andrews_bay,2025_01_01_01_01_06_PST,uri,desc,tp_human_only\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                with patch("download_wavs.process_csv") as mock_process_csv, \
                        patch("download_wavs.process_testing_csv") as mock_process_testing_csv, \
                        patch("download_wavs.validate_aligned_entries") as mock_validate_aligned_entries:
                    run_download_wavs(validate_only=True)
                mock_process_csv.assert_not_called()
                mock_process_testing_csv.assert_not_called()
                mock_validate_aligned_entries.assert_called_once()
            finally:
                os.chdir(original_cwd)
