#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for run_inference.py.

Tests cover:
- run_inference() with mocked HuggingFace model
- run_inference() with mocked FastAI model
- Per-class probability output format and values
- CLI argument validation (invalid model type)
- Missing wav file error handling
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(duration_s: int = 5, sr: int = 16000) -> str:
    """Write a synthetic wav file and return its path (caller must clean up)."""
    samples = np.zeros(duration_s * sr, dtype=np.float32)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, samples, sr)
    f.close()
    return f.name


def _make_fastai_model_mock(global_prediction: int = 1, global_confidence: float = 0.75,
                             num_local: int = 5) -> MagicMock:
    """Return a mock FastAI model whose predict() returns a binary result."""
    mock_model = MagicMock()
    local_confidences = [0.8] * num_local
    mock_model.predict.return_value = {
        "local_predictions": [global_prediction] * num_local,
        "local_confidences": local_confidences,
        "global_prediction": global_prediction,
        "global_confidence": global_confidence,
        "hop_duration": 1.0,
        "segment_duration": 3.0,
    }
    return mock_model


def _make_huggingface_model_mock(num_local: int = 10) -> MagicMock:
    """Return a mock HuggingFace model whose predict() returns a 7-class result."""
    mock_model = MagicMock()

    # 7-class label mapping matching the standard schema.
    mock_model.id2label = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
        5: "jingle",
        6: "human",
    }
    mock_model.label2id = {v: k for k, v in mock_model.id2label.items()}
    mock_model.threshold = 0.5

    # Simulate most windows predicting "resident" (1) with confidence above threshold,
    # and a few predicting "water" (0) with confidence below threshold.
    local_predictions = [1] * (num_local - 2) + [0] * 2
    local_confidences = [0.7] * (num_local - 2) + [0.1] * 2

    mock_model.predict.return_value = {
        "local_predictions": local_predictions,
        "local_confidences": local_confidences,
        "global_prediction": 1,
        "global_prediction_label": "resident",
        "global_confidence": 0.7,
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }
    return mock_model


# ---------------------------------------------------------------------------
# Tests for run_inference()
# ---------------------------------------------------------------------------

class TestRunInferenceHuggingFace:
    """Tests for run_inference() with a mocked HuggingFace model."""

    def test_returns_expected_keys(self):
        """run_inference returns a dict with probabilities, global_prediction_label, global_confidence."""
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            assert "probabilities" in result
            assert "global_prediction_label" in result
            assert "global_confidence" in result
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_probabilities_cover_all_seven_classes(self):
        """All 7 HuggingFace class labels are present in the probabilities output."""
        expected_labels = {"water", "resident", "transient", "humpback", "vessel", "jingle", "human"}
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            assert set(result["probabilities"].keys()) == expected_labels
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_predicted_class_probability_equals_global_confidence(self):
        """The globally predicted class should have a probability equal to global_confidence."""
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock(num_local=10)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            # "resident" windows (local_predictions==1) all have confidence 0.7 > threshold 0.5.
            # So resident probability = mean([0.7]*8) = 0.7 = global_confidence.
            assert abs(result["probabilities"]["resident"] - 0.7) < 1e-4
            assert abs(result["probabilities"]["resident"] - result["global_confidence"]) < 1e-4
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_classes_below_threshold_have_zero_probability(self):
        """Classes whose windows are all below the confidence threshold should have probability 0.0."""
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock(num_local=10)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            # "water" windows (local_predictions==0) have confidence 0.1 < threshold 0.5 → 0.0.
            assert result["probabilities"]["water"] == 0.0
            # Classes never predicted also get 0.0.
            for label in ["transient", "humpback", "vessel", "jingle", "human"]:
                assert result["probabilities"][label] == 0.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_probabilities_in_valid_range(self):
        """All per-class probability values should be in [0.0, 1.0]."""
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            for label, prob in result["probabilities"].items():
                assert 0.0 <= prob <= 1.0, f"Probability for {label!r} out of range: {prob}"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_and_confidence(self):
        """global_prediction_label and global_confidence match the mock model output."""
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            assert result["global_prediction_label"] == "resident"
            assert abs(result["global_confidence"] - 0.7) < 1e-6
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_empty_predictions_returns_all_zero_probabilities(self):
        """When local_predictions is empty, all class probabilities should be 0.0."""
        wav_path = _make_wav()
        try:
            mock_model = _make_huggingface_model_mock()
            mock_model.predict.return_value = {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": 0,
                "global_prediction_label": "water",
                "global_confidence": 0.0,
            }
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="huggingface", model_path="fake-path")

            for prob in result["probabilities"].values():
                assert prob == 0.0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_raises_value_error_without_model_path(self):
        """run_inference raises ValueError when model_path is None for huggingface."""
        wav_path = _make_wav()
        try:
            from run_inference import run_inference
            with pytest.raises(ValueError, match="model_path is required"):
                run_inference(wav_path, model_type="huggingface", model_path=None)
        finally:
            Path(wav_path).unlink(missing_ok=True)


class TestRunInferenceFastAI:
    """Tests for run_inference() with a mocked FastAI model."""

    def test_returns_expected_keys(self):
        """run_inference with fastai returns a dict with the required keys."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert "probabilities" in result
            assert "global_prediction_label" in result
            assert "global_confidence" in result
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_probabilities_contain_two_classes(self):
        """FastAI output should contain exactly 'other' and 'whale' classes."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert set(result["probabilities"].keys()) == {"other", "whale"}
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_fastai_probabilities_sum_to_one(self):
        """'other' and 'whale' probabilities should sum to 1.0."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock(global_confidence=0.75, num_local=5)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-3
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_whale_when_positive(self):
        """When global_prediction=1, global_prediction_label should be 'whale'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock(global_prediction=1, global_confidence=0.8)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert result["global_prediction_label"] == "whale"
            assert abs(result["global_confidence"] - 0.8) < 1e-6
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_global_prediction_other_when_negative(self):
        """When global_prediction=0, global_prediction_label should be 'other'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock(global_prediction=0, global_confidence=0.0)
            with patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import run_inference
                result = run_inference(wav_path, model_type="fastai", model_path="./model")

            assert result["global_prediction_label"] == "other"
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_defaults_model_path_to_dot_model(self):
        """When model_path is None for fastai, get_model_inference is called with './model'."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("run_inference.get_model_inference", return_value=mock_model) as mock_factory:
                from run_inference import run_inference
                run_inference(wav_path, model_type="fastai", model_path=None)

            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args
            assert call_kwargs.kwargs.get("model_path") == "./model" or \
                   (call_kwargs.args and "./model" in call_kwargs.args)
        finally:
            Path(wav_path).unlink(missing_ok=True)


class TestRunInferenceErrors:
    """Tests for error handling in run_inference()."""

    def test_raises_on_unknown_model_type(self):
        """run_inference raises ValueError for an unknown model type."""
        wav_path = _make_wav()
        try:
            from run_inference import run_inference
            with pytest.raises(ValueError, match="Unknown model type"):
                run_inference(wav_path, model_type="unknown_model", model_path="./model")
        finally:
            Path(wav_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Tests for the main() entry point."""

    def test_main_returns_one_when_wav_not_found(self, tmp_path):
        """main() returns exit code 1 when the wav file does not exist."""
        nonexistent = str(tmp_path / "nonexistent.wav")
        with patch("sys.argv", ["run_inference.py", nonexistent, "--model", "fastai", "--model-path", "./model"]):
            from run_inference import main
            assert main() == 1

    def test_main_returns_zero_on_success(self):
        """main() returns exit code 0 on successful inference."""
        wav_path = _make_wav()
        try:
            mock_model = _make_fastai_model_mock()
            with patch("sys.argv", ["run_inference.py", wav_path, "--model", "fastai", "--model-path", "./model"]), \
                 patch("run_inference.get_model_inference", return_value=mock_model):
                from run_inference import main
                assert main() == 0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_main_returns_one_on_value_error(self):
        """main() returns exit code 1 when run_inference raises ValueError."""
        wav_path = _make_wav()
        try:
            # Calling with huggingface and no --model-path should raise ValueError.
            with patch("sys.argv", ["run_inference.py", wav_path, "--model", "huggingface"]):
                from run_inference import main
                assert main() == 1
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_print_results_output_contains_class_names(self, capsys):
        """print_results() writes class names to stdout."""
        from run_inference import print_results
        results = {
            "probabilities": {"other": 0.3, "whale": 0.7},
            "global_prediction_label": "whale",
            "global_confidence": 0.7,
        }
        print_results(results, "fastai")
        captured = capsys.readouterr()
        assert "other" in captured.out
        assert "whale" in captured.out
        assert "fastai" in captured.out
        assert "0.7000" in captured.out
