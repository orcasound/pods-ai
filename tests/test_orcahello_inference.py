"""Tests for OrcaHello/pods-ai inference configuration compatibility."""

from dataclasses import dataclass

from orcahello_inference import _patch_detector_inference_config


def test_from_yaml_uses_defaults_when_file_is_missing(tmp_path) -> None:
    @dataclass
    class Config:
        value: int = 5

    _patch_detector_inference_config(Config)

    assert Config.from_yaml(str(tmp_path / "missing.yml")).value == 5


def test_from_dict_coerces_nested_dataclass_values() -> None:
    @dataclass
    class InnerConfig:
        threshold: int = 3

    @dataclass
    class OuterConfig:
        enabled: bool = True
        inner: InnerConfig = None

    _patch_detector_inference_config(OuterConfig)

    cfg = OuterConfig.from_dict({"enabled": False, "inner": {"threshold": 7}})

    assert cfg.enabled is False
    assert isinstance(cfg.inner, InnerConfig)
    assert cfg.inner.threshold == 7


def test_from_yaml_reads_yaml_and_preserves_nested_defaults(tmp_path) -> None:
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text("enabled: false\ninner:\n  threshold: 11\n", encoding="utf-8")

    @dataclass
    class InnerConfig:
        threshold: int = 3

    @dataclass
    class OuterConfig:
        enabled: bool = True
        inner: InnerConfig = None

    _patch_detector_inference_config(OuterConfig)

    cfg = OuterConfig.from_yaml(str(yaml_path))

    assert cfg.enabled is False
    assert isinstance(cfg.inner, InnerConfig)
    assert cfg.inner.threshold == 11


def test_patch_is_idempotent_for_repeated_calls() -> None:
    @dataclass
    class InnerConfig:
        value: int = 1

    @dataclass
    class Config:
        inner: InnerConfig = None

    _patch_detector_inference_config(Config)
    _patch_detector_inference_config(Config)

    cfg = Config.from_dict({"inner": {"value": 9}})
    assert cfg.inner.value == 9
    assert Config.from_yaml("/tmp/does-not-exist.yml").inner.value == 1
