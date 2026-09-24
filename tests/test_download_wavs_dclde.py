# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Regression tests for DCLDE integration in download_wavs.py."""

from pathlib import Path

import pytest

from download_wavs import (
    CSVRow,
    DEFAULT_DCLDE_WAV_ROOT,
    DEFAULT_TESTING_WAV_ROOT,
    process_dclde_csv,
    process_testing_csv,
    validate_no_overlaps,
)


CSV_HEADER = "category,node_name,timestamp_pst,uri,description,notes,confidence\n"


def test_dclde_uses_testing_wav_directory_by_default():
    assert DEFAULT_DCLDE_WAV_ROOT == Path("output/testing-wav")
    assert DEFAULT_DCLDE_WAV_ROOT == DEFAULT_TESTING_WAV_ROOT


def test_overlap_validation_includes_training_and_dclde():
    training_rows = [
        CSVRow(
            "resident", "rpi_orcasound_lab", "2025_01_01_00_00_30_PST",
            "", "", "",
        )
    ]
    dclde_rows = [
        CSVRow(
            "resident", "rpi_orcasound_lab", "2025_01_01_00_00_00_PST",
            "https://example.org/recording.wav", "", "dclde_orcasound_full_recording",
        )
    ]

    with pytest.raises(ValueError, match="training .* overlaps DCLDE"):
        validate_no_overlaps(training_rows, [], dclde_rows)


def test_shared_cleanup_preserves_files_from_both_manifests(tmp_path):
    testing_csv = tmp_path / "testing_60s_samples.csv"
    testing_csv.write_text(
        CSV_HEADER
        + "resident,rpi_orcasound_lab,2025_01_01_01_00_00_PST,uri,test,tp_human_only,100\n",
        encoding="utf-8",
    )
    dclde_csv = tmp_path / "dclde_60s_samples.csv"
    dclde_csv.write_text(
        CSV_HEADER
        + "abiotic,rpi_bush_point,2017_01_01_01_00_00_PST,"
        "https://example.org/dclde.wav,dclde,dclde_orcasound_full_recording,100\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "testing-wav"
    testing_path = (
        output_root / "resident" /
        "rpi-orcasound-lab_2025_01_01_01_00_00_PST.wav"
    )
    dclde_path = (
        output_root / "abiotic" /
        "rpi-bush-point_2017_01_01_01_00_00_PST.wav"
    )
    stale_path = output_root / "resident" / "stale.wav"
    for path in (testing_path, dclde_path, stale_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")

    combined_expected = {
        testing_path.relative_to(output_root),
        dclde_path.relative_to(output_root),
    }
    process_testing_csv(
        testing_csv,
        output_root,
        cleanup_expected_paths=combined_expected,
    )
    process_dclde_csv(
        dclde_csv,
        output_root,
        cleanup_expected_paths=combined_expected,
    )

    assert testing_path.exists()
    assert dclde_path.exists()
    assert not stale_path.exists()
