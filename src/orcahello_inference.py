# !/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
OrcaHello SRKW Detector inference wrapper.

This module provides an inference wrapper for the OrcaHello SRKW detector model
(orcasound/orcahello-srkw-detector-v1) hosted on HuggingFace Hub.

The model implementation is imported from the orcasound/orcahello submodule at
external/orcahello/InferenceSystem/src/model.  Initialize the submodule before
use with:

    git submodule update --init external/orcahello

Usage:
    from orcahello_inference import OrcaHelloSRKWInference

    model = OrcaHelloSRKWInference("orcasound/orcahello-srkw-detector-v1")
    result = model.predict("sample.wav")
    print(result["global_prediction_label"])  # "whale" or "other"
"""

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, get_args, get_origin

from model_inference import ModelInference

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _coerce_dataclass_value(value: Any, field_type: Any) -> Any:
    """Recursively coerce nested dicts into dataclass instances."""
    if value is None:
        if dataclasses.is_dataclass(field_type):
            return field_type()
        return None

    origin = get_origin(field_type)
    if origin is not None:
        args = get_args(field_type)
        if type(None) in args and value is None:
            return None
        if origin in (list, tuple, set):
            list_type = args[0] if args else None
            if list_type is None:
                return value
            return [
                _coerce_dataclass_value(item, list_type)
                for item in value
            ]
        if origin is dict:
            key_type, val_type = args if len(args) == 2 else (None, None)
            if val_type is not None:
                return {
                    key: _coerce_dataclass_value(val, val_type)
                    for key, val in value.items()
                }
        if origin is not None and not args:
            return value

    if dataclasses.is_dataclass(field_type) and not isinstance(value, field_type):
        if isinstance(value, dict):
            if hasattr(field_type, "from_dict") and callable(getattr(field_type, "from_dict")):
                return field_type.from_dict(value)
            return field_type(**value)
        return value

    return value


def _patch_detector_inference_config(config_cls: Type) -> Type:
    """Patch a config dataclass to support YAML/Dict loading with nested defaults."""
    if getattr(config_cls, "_podsai_config_patched", False):
        return config_cls

    for field in dataclasses.fields(config_cls):
        if dataclasses.is_dataclass(field.type) and not getattr(field.type, "_podsai_config_patched", False):
            _patch_detector_inference_config(field.type)

    def _from_dict(cls: Type, values: Optional[Dict[str, Any]] = None) -> Any:
        payload = values or {}
        if not isinstance(payload, dict):
            raise TypeError(f"{cls.__name__}.from_dict() expects a dict, got {type(payload).__name__}")

        kwargs: Dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            field_value = payload.get(field.name, dataclasses.MISSING)
            if field_value is dataclasses.MISSING:
                if dataclasses.is_dataclass(field.type):
                    kwargs[field.name] = field.type()
                elif hasattr(cls, field.name):
                    kwargs[field.name] = getattr(cls, field.name)
                continue
            if field_value is None and dataclasses.is_dataclass(field.type):
                kwargs[field.name] = field.type()
                continue
            kwargs[field.name] = _coerce_dataclass_value(field_value, field.type)

        return cls(**kwargs)

    def _from_yaml(cls: Type, yaml_path: str) -> Any:
        if not os.path.exists(yaml_path):
            return cls.from_dict({})
        if yaml is None:
            raise ImportError("PyYAML is required to load config YAML files")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()

        if not content:
            return cls.from_dict({})

        lines = content.splitlines()
        nonempty = [line for line in lines if line.strip()]
        if nonempty:
            indents = [len(line) - len(line.lstrip(" \t")) for line in nonempty]
            base_indent = min(indents) if indents else 0
            normalized = "\n".join(
                line[base_indent:] if line.strip() else "" for line in lines
            )
            payload = yaml.safe_load(normalized) or {}
        else:
            payload = {}

        return cls.from_dict(payload)

    config_cls.from_dict = classmethod(_from_dict)
    config_cls.from_yaml = classmethod(_from_yaml)
    config_cls._podsai_config_patched = True
    return config_cls


# Path to the orcahello submodule's model package.
_ORCAHELLO_SRC = (
    Path(__file__).parents[1] / "external" / "orcahello" / "InferenceSystem" / "src"
)

# Cached tuple of (OrcaHelloSRKWDetectorV1, DetectorInferenceConfig) from submodule.
_ORCAHELLO_IMPORTS: Optional[Tuple[Type, Type]] = None


def _get_orcahello_classes() -> Tuple[Type, Type]:
    """Lazily import OrcaHelloSRKWDetectorV1 and DetectorInferenceConfig from the submodule.

    Adds the submodule's src directory to sys.path on first call.

    Returns:
        Tuple of (OrcaHelloSRKWDetectorV1, DetectorInferenceConfig).

    Raises:
        ImportError: If the orcahello submodule is not initialized or dependencies
                     are missing.
    """
    global _ORCAHELLO_IMPORTS
    if _ORCAHELLO_IMPORTS is not None:
        return _ORCAHELLO_IMPORTS

    if not _ORCAHELLO_SRC.exists():
        raise ImportError(
            f"orcahello submodule not found at {_ORCAHELLO_SRC}. "
            "Run: git submodule update --init external/orcahello"
        )

    src_str = str(_ORCAHELLO_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    try:
        from model import OrcaHelloSRKWDetectorV1, DetectorInferenceConfig  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"Failed to import from orcahello submodule at {_ORCAHELLO_SRC}. "
            f"Ensure the submodule is initialized and all dependencies are installed "
            f"(scipy, torchaudio, pydub, librosa, soundfile). "
            f"Original error: {e}"
        ) from e

    _ORCAHELLO_IMPORTS = (OrcaHelloSRKWDetectorV1, DetectorInferenceConfig)
    return _ORCAHELLO_IMPORTS


class OrcaHelloSRKWInference(ModelInference):
    """Inference wrapper for the OrcaHello SRKW detector model.

    Loads OrcaHelloSRKWDetectorV1 from HuggingFace Hub (default:
    orcasound/orcahello-srkw-detector-v1) via the orcahello submodule.

    The model is a binary classifier (0 = no SRKW, 1 = SRKW detected).
    Like the updated PODS-AI default training path, this model uses mel
    spectrogram features instead of raw-audio Wav2Vec2 embeddings.
    """

    DEFAULT_MODEL_PATH = "orcasound/orcahello-srkw-detector-v1"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config: Optional[Dict] = None,
        model_revision: Optional[str] = None,
    ) -> None:
        """Initialize the OrcaHello SRKW inference wrapper.

        Args:
            model_path: HuggingFace Hub model ID or local path to model directory.
                        Defaults to "orcasound/orcahello-srkw-detector-v1".
            config: Optional configuration overrides as a nested dict.
                    If None, uses the default DetectorInferenceConfig values.
            model_revision: Optional Hugging Face revision. Ignored for local paths.

        Raises:
            ImportError: If the orcahello submodule is not initialized.
            RuntimeError: If the model cannot be loaded from model_path.
        """
        super().__init__(model_path=model_path)

        OrcaHelloSRKWDetectorV1, DetectorInferenceConfig = _get_orcahello_classes()
        _patch_detector_inference_config(DetectorInferenceConfig)
        self.inference_config = DetectorInferenceConfig.from_dict(config or {})

        expanded_model_path = Path(model_path).expanduser()
        is_local_model = expanded_model_path.exists()
        load_path = str(expanded_model_path) if is_local_model else model_path
        pretrained_kwargs = {}
        if model_revision and not is_local_model:
            pretrained_kwargs["revision"] = model_revision

        revision_info = f" @ {model_revision}" if "revision" in pretrained_kwargs else ""
        print(f"Loading OrcaHello SRKW detector from {load_path}{revision_info}...")
        try:
            self.model = OrcaHelloSRKWDetectorV1.from_pretrained(
                load_path,
                config=self.inference_config.as_dict(),
                **pretrained_kwargs,
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Error loading OrcaHello model from {load_path}: {type(e).__name__}: {e}"
            ) from e

        print(
            f"OrcaHello model loaded. Device: {self.model._device}, "
            f"Dtype: {self.model._dtype}"
        )

    def predict(self, wav_file_path: str) -> Dict:
        """Run inference on a wav file.

        Uses the OrcaHello audio preprocessing pipeline (mel spectrograms via
        torchaudio, no fastai_audio dependency) and the ResNet50-based detector.

        Args:
            wav_file_path: Path to the wav file (typically 60 seconds long).

        Returns:
            Dictionary with keys:
                - local_predictions: List of binary predictions (0/1) per segment.
                - local_confidences: List of confidence scores (0.0-1.0) per segment.
                - global_prediction: Overall binary prediction (0 or 1).
                - global_prediction_label: "whale" if global_prediction==1 else "other".
                - global_confidence: Aggregated confidence score (0.0-1.0).
                - hop_duration: Hop duration in seconds between segments.
                - segment_duration: Duration of each segment in seconds.
            Returns dict with empty lists and zero scores on error.
        """
        try:
            result = self.model.detect_srkw_from_file(
                wav_file_path, self.inference_config
            )
        except Exception as e:
            error_message = (
                f"{type(e).__name__}: {e}"
            )
            print(f"Error running inference on {wav_file_path}: {error_message}")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": 0,
                "global_prediction_label": "other",
                "global_confidence": 0.0,
                "error": error_message,
                "hop_duration": float(self.inference_config.inference.window_hop_s),
                "segment_duration": float(self.inference_config.inference.window_s),
            }

        return {
            "local_predictions": result.local_predictions,
            "local_confidences": result.local_confidences,
            "global_prediction": result.global_prediction,
            "global_prediction_label": "whale" if result.global_prediction == 1 else "other",
            "global_confidence": result.global_confidence,
            "hop_duration": float(self.inference_config.inference.window_hop_s),
            "segment_duration": float(self.inference_config.inference.window_s),
        }


def get_orcahello_srkw_inference(
    model_path: str = OrcaHelloSRKWInference.DEFAULT_MODEL_PATH,
    model_revision: Optional[str] = None,
    **kwargs,
) -> OrcaHelloSRKWInference:
    """Factory function to create an OrcaHelloSRKWInference instance.

    Args:
        model_path: HuggingFace Hub model ID or local path.
        model_revision: Optional Hugging Face revision for Hub models.
        **kwargs: Additional arguments passed to OrcaHelloSRKWInference.

    Returns:
        OrcaHelloSRKWInference instance.
    """
    return OrcaHelloSRKWInference(
        model_path, model_revision=model_revision, **kwargs
    )