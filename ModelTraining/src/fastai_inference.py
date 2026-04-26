#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
FastAI legacy inference wrapper for the orca detection model.

This module reduces code duplication by importing FastAIModel from the orcahello
submodule's legacy directory (external/orcahello/InferenceSystem/src/legacy/) rather
than maintaining a second copy of the same logic.  The legacy module also adds GPU
device placement at initialization, which is a performance benefit over the built-in
FastAIModel fallback.

The legacy FastAI model uses 2-second segments with 1-second hop.  All segments are
padded/snipped to 4 seconds by the fastai_audio AudioConfig, so the model inputs are
equivalent to the 3-second segment variant in model_inference.py.

Initialize the submodule before use with:

    git submodule update --init external/orcahello

Usage:
    from fastai_inference import get_legacy_fastai_inference

    model = get_legacy_fastai_inference("./model", use_gpu=True)
    result = model.predict("sample.wav")
    print(result["global_prediction"])  # 0 or 1
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Type

from model_inference import ModelInference, SEGMENT_GROUP_SIZE


# Path to the orcahello submodule's legacy directory.
_LEGACY_SRC = (
    Path(__file__).parent.parent.parent
    / "external"
    / "orcahello"
    / "InferenceSystem"
    / "src"
    / "legacy"
)

# Cached FastAIModel class from the legacy module (populated on first use).
_LEGACY_FASTAI_MODEL_CLASS: Optional[Type] = None


def _get_legacy_fastai_model_class() -> Type:
    """Lazily import FastAIModel from the orcahello legacy module.

    Adds the legacy directory to sys.path on the first call, then imports and
    caches the class for subsequent calls.

    Returns:
        FastAIModel class from the orcahello legacy module.

    Raises:
        ImportError: If the orcahello submodule is not initialized or its
                     dependencies (fastai, fastai_audio) are missing.
    """
    global _LEGACY_FASTAI_MODEL_CLASS
    if _LEGACY_FASTAI_MODEL_CLASS is not None:
        return _LEGACY_FASTAI_MODEL_CLASS

    if not _LEGACY_SRC.exists():
        raise ImportError(
            f"orcahello legacy module not found at {_LEGACY_SRC}. "
            "Run: git submodule update --init external/orcahello"
        )

    legacy_str = str(_LEGACY_SRC)
    if legacy_str not in sys.path:
        sys.path.insert(0, legacy_str)

    try:
        # Import FastAIModel from the orcahello legacy module, not from this file.
        # The legacy path is at sys.path[0] so it takes priority over src/fastai_inference.py.
        from fastai_inference import FastAIModel as _LegacyFastAIModel  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"Failed to import FastAIModel from orcahello legacy module at {_LEGACY_SRC}. "
            "Ensure the submodule is initialized and all dependencies are installed "
            "(fastai==1.0.61, fastai_audio). "
            f"Original error: {e}"
        ) from e

    _LEGACY_FASTAI_MODEL_CLASS = _LegacyFastAIModel
    return _LEGACY_FASTAI_MODEL_CLASS


class LegacyFastAIInference(ModelInference):
    """Inference wrapper that delegates to the orcahello legacy FastAIModel.

    Imports FastAIModel from external/orcahello/InferenceSystem/src/legacy/ and
    adapts its output to the ModelInference interface.  Compared with the built-in
    FastAIModel fallback in model_inference.py this class adds:

    - GPU device placement at initialization (``use_gpu=True`` moves the model to
      CUDA once, avoiding repeated host-to-device transfers during inference).
    - Configurable rolling-average smoothing (``smooth_predictions``).
    - Explicit memory cleanup (``gc.collect`` + ``torch.cuda.empty_cache``) after
      each inference call.

    The legacy model uses 2-second audio segments with 1-second hop.  Since
    fastai_audio pads/snips every segment to 4 seconds (``config.duration=4000``),
    the model inputs are equivalent to the 3-second variant in the built-in fallback.
    """

    def __init__(
        self,
        model_path: str = "./model",
        model_name: str = "model.pkl",
        threshold: float = 0.5,
        min_num_positive_calls_threshold: int = 3,
        use_gpu: bool = False,
        smooth_predictions: bool = True,
        batch_size: int = 32,
    ) -> None:
        """Initialize the legacy FastAI inference wrapper.

        Args:
            model_path: Path to directory containing the model file.
            model_name: Name of the model file (default: "model.pkl").
            threshold: Confidence threshold for positive predictions (default: 0.5).
            min_num_positive_calls_threshold: Minimum number of positive-window
                predictions required for a global positive result.  Automatically
                scaled down for short clips using SEGMENT_GROUP_SIZE.  Default: 3.
            use_gpu: If True, move the model to CUDA at initialization when a
                GPU is available.  Provides significant throughput improvement for
                batch inference on GPU-capable machines.  Default: False.
            smooth_predictions: If True, apply a rolling-average window to smooth
                per-segment predictions before thresholding.  Default: True.
            batch_size: Number of audio segments per DataLoader batch.  Larger
                values improve GPU throughput; reduce if you run out of memory.
                Default: 32.

        Raises:
            ImportError: If the orcahello submodule is not initialized or its
                         dependencies are missing.
        """
        super().__init__(model_path=model_path)
        _LegacyFastAIModel = _get_legacy_fastai_model_class()
        self._legacy = _LegacyFastAIModel(
            model_path,
            model_name=model_name,
            threshold=threshold,
            min_num_positive_calls_threshold=min_num_positive_calls_threshold,
            use_gpu=use_gpu,
            smooth_predictions=smooth_predictions,
            batch_size=batch_size,
        )
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold

    def predict(self, wav_file_path: str) -> Dict:
        """Run inference via the legacy FastAI model and adapt the output format.

        Args:
            wav_file_path: Path to the wav file to score.

        Returns:
            Dictionary with:
                - local_predictions: list of per-hop binary predictions (0 or 1).
                - local_confidences: list of per-hop confidence scores (0.0-1.0).
                - global_prediction: overall binary prediction (0 or 1), scaled by
                  SEGMENT_GROUP_SIZE.
                - global_confidence: mean confidence of positive windows (0.0-1.0).
                - hop_duration: 1.0 (seconds between hop positions).
                - segment_duration: 2.0 (seconds per audio segment in legacy model).
        """
        result = self._legacy.predict(wav_file_path)

        # The legacy model returns global_confidence as a percentage (0-100); convert
        # to 0-1 range to match the ModelInference interface.
        global_confidence = float(result.get("global_confidence", 0.0))
        if global_confidence > 1.0:
            global_confidence = round(global_confidence / 100.0, 4)

        # Apply SEGMENT_GROUP_SIZE threshold scaling (not done in the legacy model).
        # For every SEGMENT_GROUP_SIZE segments, require at least 1 positive prediction.
        local_predictions = result.get("local_predictions", [])
        total_segments = len(local_predictions)
        scaled_threshold = max(1, (total_segments + SEGMENT_GROUP_SIZE - 1) // SEGMENT_GROUP_SIZE)
        effective_threshold = min(scaled_threshold, self.min_num_positive_calls_threshold)

        return {
            "local_predictions": local_predictions,
            "local_confidences": result.get("local_confidences", []),
            "global_prediction": int(sum(local_predictions) >= effective_threshold),
            "global_confidence": global_confidence,
            "hop_duration": 1.0,
            "segment_duration": 2.0,
        }


def get_legacy_fastai_inference(
    model_path: str = "./model",
    model_name: str = "model.pkl",
    threshold: float = 0.5,
    min_num_positive_calls_threshold: int = 3,
    use_gpu: bool = False,
    smooth_predictions: bool = True,
    batch_size: int = 32,
) -> LegacyFastAIInference:
    """Create a LegacyFastAIInference instance.

    Preferred over constructing LegacyFastAIInference directly because it mirrors
    the factory-function pattern used by get_orcahello_srkw_inference and
    get_huggingface_inference.

    Args:
        model_path: Path to directory containing the model file.
        model_name: Name of the model file (default: "model.pkl").
        threshold: Confidence threshold for positive predictions (default: 0.5).
        min_num_positive_calls_threshold: Minimum positive predictions for a global
            positive result.  Scaled by SEGMENT_GROUP_SIZE for short clips.
            Default: 3.
        use_gpu: If True, place the model on CUDA at initialization when available.
            Provides a significant performance benefit.  Default: False.
        smooth_predictions: If True, apply rolling-average smoothing.  Default: True.
        batch_size: Number of audio segments per DataLoader batch.  Larger values
            improve GPU throughput; reduce if you run out of memory.  Default: 32.

    Returns:
        LegacyFastAIInference instance.

    Raises:
        ImportError: If the orcahello submodule is not initialized or its
                     dependencies are missing.
    """
    return LegacyFastAIInference(
        model_path=model_path,
        model_name=model_name,
        threshold=threshold,
        min_num_positive_calls_threshold=min_num_positive_calls_threshold,
        use_gpu=use_gpu,
        smooth_predictions=smooth_predictions,
        batch_size=batch_size,
    )
