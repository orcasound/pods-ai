"""Tests for OrcaHello inference configuration compatibility."""

from dataclasses import dataclass

from orcahello_inference import _patch_detector_inference_config


def test_from_yaml_uses_defaults_when_file_is_missing(tmp_path) -> None:
    @dataclass
    class Config:
        value: int = 5

    _patch_detector_inference_config(Config)

    assert Config.from_yaml(str(tmp_path / "missing.yml")).value == 5
