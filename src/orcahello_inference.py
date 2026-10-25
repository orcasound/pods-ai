#!/usr/bin/env python3
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
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union, get_args, get_origin

from model_inference import ModelInference


def _resolve_nested_dataclass_type(field_type: Any) -> Optional[Type]:
    """Resolve the dataclass type for a field, including Optional[T]."""
    if isinstance(field_type, type):
        return field_type if dataclasses.is_dataclass(field_type) else None

    origin = get_origin(field_type)
    union_types = {getattr(types, "UnionType", None), Union}
    if origin in union_types:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return _resolve_nested_dataclass_type(args[0])
    return None


def _patch_detector_inference_config(detector_config_cls: Type) -> None:
    """Patch the vendored DetectorInferenceConfig to support stable round-trips.

    We intentionally do this in the wrapper layer instead of editing the external
    submodule so the patch remains local to this repo while keeping the upstream
    file unchanged.
    """
    if getattr(detector_config_cls, "_podsai_roundtrip_patched", False):
        return

    module = sys.modules.get(detector_config_cls.__module__)
    nested_types = {
        field.name: (
            getattr(module, field_type.__name__, field_type)
            if module is not None and hasattr(field_type, "__name__")
            else field_type
        )
        for field in dataclasses.fields(detector_config_cls)
        for field_type in [
            _resolve_nested_dataclass_type(field.type)
        ]
        if field_type is not None
    }

    def _coerce(value: Any, target_type: Optional[Type]) -> Any:
        if target_type is None:
            return value
        if value is None:
            return target_type() if dataclasses.is_dataclass(target_type) else value
        if isinstance(value, target_type):
            return value
        if dataclasses.is_dataclass(value):
            return value
        if isinstance(value, dict):
            if not dataclasses.is_dataclass(target_type):
                raise TypeError(
                    f"Expected dataclass type, got {target_type!r} for {type(value).__name__}"
                )
            nested_target_types = {
                nested_field.name: _resolve_nested_dataclass_type(nested_field.type)
                for nested_field in dataclasses.fields(target_type)
            }
            resolved = {}
            for field in dataclasses.fields(target_type):
                if field.name not in value:
                    continue
                nested_target_type = nested_target_types.get(field.name)
                resolved[field.name] = _coerce(value[field.name], nested_target_type)
            return target_type(**resolved)
        raise TypeError(
            f"Expected {target_type.__name__} or dict, got {type(value).__name__}"
        )

    def _patched_post_init(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            target_type = nested_types.get(field.name)
            if target_type is not None:
                setattr(self, field.name, _coerce(value, target_type))

    def _patched_from_dict(cls, d: Optional[Dict[str, Any]]):
        if d is None:
            return cls()
        if isinstance(d, cls):
            return d
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or {cls.__name__}, got {type(d).__name__}")

        resolved = {}
        for field in dataclasses.fields(cls):
            if field.name in d:
                value = d[field.name]
            elif field.default is not dataclasses.MISSING:
                value = field.default
            elif field.default_factory is not dataclasses.MISSING:
                value = field.default_factory()
            else:
                value = getattr(cls, field.name, None)

            target_type = nested_types.get(field.name)
            if target_type is not None:
                resolved[field.name] = _coerce(value, target_type)
            else:
                resolved[field.name] = value
        return cls(**resolved)

    def _patched_from_yaml(cls, yaml_path: str):
        import yaml

        try:
            with open(yaml_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            data = {}
        return cls.from_dict(data)

    detector_config_cls.__post_init__ = _patched_post_init
    detector_config_cls.from_dict = classmethod(_patched_from_dict)
    detector_config_cls.from_yaml = classmethod(_patched_from_yaml)
    detector_config_cls._podsai_roundtrip_patched = True


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

    _patch_detector_inference_config(DetectorInferenceConfig)
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
