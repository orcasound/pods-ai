import json
import subprocess

import pytest

from testing_dataset import load_testing_dataset_metadata


def test_loads_semver_and_optional_provenance_from_json(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_version": "1.2.3",
                "source_id": ["archive-a", "archive-b", "archive-a"],
                "hydrophone_site": "rpi_orcasound_lab",
            }
        ),
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v1.2.3"
    assert metadata.source_ids == ("archive-a", "archive-b")
    assert metadata.hydrophone_sites == ("rpi_orcasound_lab",)
    assert metadata.git_tag is None
    assert metadata.warnings == ()


@pytest.mark.parametrize("version", ["1", "1.2", "01.2.3", "v1.02.3", "latest"])
def test_invalid_json_version_falls_back_without_raising(tmp_path, version):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(json.dumps({"version": version}), encoding="utf-8")

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "unversioned"
    assert any("invalid dataset version" in warning for warning in metadata.warnings)


def test_version_alias_is_used_when_preferred_field_is_invalid(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(
        json.dumps({"dataset_version": "latest", "version": "1.2.3"}),
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v1.2.3"
    assert any("invalid dataset version" in warning for warning in metadata.warnings)


def test_reads_version_and_optional_site_columns_from_csv(tmp_path):
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,NodeName,StartTimestamp,Source ID,Hydrophone Site,Version\n"
        "resident,rpi_orcasound_lab,2026_01_02_03_04_05_PST,archive-a,lab,2.1.0\n"
        "transient,rpi_mast_center,2026_01_03_03_04_05_PST,archive-b,mast,2.1.0\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v2.1.0"
    assert metadata.source_ids == ("archive-a", "archive-b")
    assert metadata.hydrophone_sites == ("lab", "mast")


def test_blank_hydrophone_site_falls_back_to_site_alias(tmp_path):
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,Hydrophone Site,Site\nresident,   ,Orcasound Lab\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.hydrophone_sites == ("Orcasound Lab",)


def test_nonblank_hydrophone_site_precedes_site_alias(tmp_path):
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,Hydrophone Site,Site\nresident,Primary Site,Alias Site\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.hydrophone_sites == ("Primary Site",)


def test_legacy_node_name_column_is_not_reported_as_hydrophone_site(tmp_path):
    manifest_path = tmp_path / "legacy-testing.csv"
    manifest_path.write_text(
        "Category,NodeName,StartTimestamp\n"
        "resident,rpi_orcasound_lab,2026_01_02_03_04_05_PST\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.hydrophone_sites == ()


def test_invalid_utf8_json_metadata_is_nonfatal(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_bytes(b'{"dataset_version":"\xff"}')

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "unversioned"
    assert any(
        "could not read dataset metadata" in warning for warning in metadata.warnings
    )


def test_invalid_utf8_csv_manifest_is_nonfatal(tmp_path):
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_bytes(b"Category,Source ID\nresident,\xff\n")

    metadata = load_testing_dataset_metadata(
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "unversioned"
    assert metadata.source_ids == ()
    assert any(
        "could not read dataset manifest" in warning for warning in metadata.warnings
    )


def test_tag_at_checked_out_commit_supplies_version(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        assert command[-3:] == ["tag", "--points-at", "HEAD"]
        return subprocess.CompletedProcess(
            command, 0, stdout="TestingDataset.v3.4.5\nother-tag\n", stderr=""
        )

    monkeypatch.setattr("testing_dataset.subprocess.run", fake_run)

    metadata = load_testing_dataset_metadata(repo_path=tmp_path)

    assert metadata.dataset_version == "v3.4.5"
    assert metadata.git_tag == "TestingDataset.v3.4.5"


def test_sidecar_tag_overrides_inconsistent_version(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_version": "1.0.0",
                "git_tag": "TestingDataset.v2.0.0",
            }
        ),
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v2.0.0"
    assert metadata.git_tag == "TestingDataset.v2.0.0"
    assert any("does not match" in warning for warning in metadata.warnings)


def test_sidecar_version_has_precedence_over_manifest_tag(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(json.dumps({"dataset_version": "2.0.0"}), encoding="utf-8")
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,git_tag\nresident,TestingDataset.v1.0.0\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v2.0.0"
    assert metadata.git_tag is None
    assert any("ignoring manifest git_tag" in warning for warning in metadata.warnings)


def test_sidecar_version_has_precedence_over_manifest_version(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(
        json.dumps({"dataset_version": "2.0.0"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,dataset_version\nresident,1.0.0\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v2.0.0"


def test_invalid_sidecar_tag_falls_back_to_manifest_tag(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(
        json.dumps({"git_tag": "not-a-testing-dataset-tag"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,git_tag\nresident,TestingDataset.v1.2.3\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v1.2.3"
    assert metadata.git_tag == "TestingDataset.v1.2.3"
    assert any(
        "ignoring invalid dataset git_tag" in warning for warning in metadata.warnings
    )


def test_valid_sidecar_tag_precedes_manifest_tag(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text(
        json.dumps({"git_tag": "TestingDataset.v2.0.0"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text(
        "Category,git_tag\nresident,TestingDataset.v1.0.0\n",
        encoding="utf-8",
    )

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "v2.0.0"
    assert metadata.git_tag == "TestingDataset.v2.0.0"


def test_legacy_dataset_without_metadata_is_unversioned(tmp_path):
    metadata = load_testing_dataset_metadata(
        metadata_path=tmp_path / "missing.json",
        manifest_path=tmp_path / "legacy.csv",
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "unversioned"
    assert metadata.source_ids == ()
    assert metadata.hydrophone_sites == ()
    assert metadata.warnings == ()


def test_invalid_metadata_file_is_nonfatal(tmp_path):
    metadata_path = tmp_path / "dataset.json"
    metadata_path.write_text("not json", encoding="utf-8")

    metadata = load_testing_dataset_metadata(
        metadata_path=metadata_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "unversioned"
    assert any(
        "could not read dataset metadata" in warning for warning in metadata.warnings
    )


def test_malformed_manifest_is_nonfatal(tmp_path):
    manifest_path = tmp_path / "testing.csv"
    manifest_path.write_text('"unterminated quote', encoding="utf-8")

    metadata = load_testing_dataset_metadata(
        manifest_path=manifest_path,
        allow_head_tag=False,
    )

    assert metadata.dataset_version == "unversioned"
