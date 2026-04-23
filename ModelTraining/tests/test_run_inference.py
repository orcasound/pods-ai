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
- Integration tests with real models (if available)
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional
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


def _verify_fastai_result_structure(result: dict) -> None:
    """Verify FastAI result has expected structure and valid values."""
    assert "probabilities" in result
    assert "global_prediction_label" in result
    assert "global_confidence" in result

    # Verify binary classes.
    assert set(result["probabilities"].keys()) == {"other", "whale"}

    # Verify probabilities sum to 1.0.
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-3

    # Verify all values in valid range.
    for prob in result["probabilities"].values():
        assert 0.0 <= prob <= 1.0

    assert 0.0 <= result["global_confidence"] <= 1.0
    assert result["global_prediction_label"] in {"other", "whale"}


def _verify_huggingface_result_structure(result: dict) -> None:
    """Verify HuggingFace result has expected structure and valid values."""
    expected_classes = {"water", "resident", "transient", "humpback", "vessel", "jingle", "human"}

    assert "probabilities" in result
    assert "global_prediction_label" in result
    assert "global_confidence" in result

    # Verify all 7 classes are present.
    assert set(result["probabilities"].keys()) == expected_classes

    # Verify all values in valid range.
    for prob in result["probabilities"].values():
        assert 0.0 <= prob <= 1.0

    assert 0.0 <= result["global_confidence"] <= 1.0
    assert result["global_prediction_label"] in expected_classes


def _print_fastai_result(result: dict, label: str = "") -> None:
    """Print FastAI inference results for debugging."""
    prefix = f"FastAI inference results{f' on {label}' if label else ''}:"
    print(f"\n{prefix}")
    print(f"  Global prediction: {result['global_prediction_label']}")
    print(f"  Global confidence: {result['global_confidence']:.4f}")
    print(f"  Probabilities: {result['probabilities']}")


def _print_huggingface_result(result: dict, label: str = "") -> None:
    """Print HuggingFace inference results for debugging."""
    prefix = f"HuggingFace inference results{f' on {label}' if label else ''}:"
    print(f"\n{prefix}")
    print(f"  Global prediction: {result['global_prediction_label']}")
    print(f"  Global confidence: {result['global_confidence']:.4f}")
    print("  Probabilities:")
    for class_label, prob in sorted(result["probabilities"].items()):
        print(f"    {class_label}: {prob:.4f}")


def _verify_fastai_prediction(result: dict, audio_type: str) -> None:
    """
    Verify FastAI model predicted the correct class for the audio type.

    For whale audio (resident, transient, humpback), expect "whale".
    For non-whale audio (water, vessel, human, jingle), expect "other".
    """
    whale_classes = {"resident", "transient", "humpback"}
    expected = "whale" if audio_type in whale_classes else "other"

    actual = result["global_prediction_label"]
    assert actual == expected, (
        f"FastAI model predicted '{actual}' for {audio_type} audio, "
        f"but expected '{expected}'"
    )


def _verify_huggingface_prediction(result: dict, audio_type: str, allow_category_match: bool = False) -> None:
    """
    Verify HuggingFace model predicted the correct class for the audio type.

    Args:
        result: Inference result dictionary.
        audio_type: Expected audio type (resident, transient, humpback, water, vessel, human, jingle).
        allow_category_match: If True, accept category match (whale vs non-whale) instead of exact match.
                              Defaults to False, requiring exact match for all classes.

    For all classes:
        - By default (allow_category_match=False), requires exact match.
        - If allow_category_match=True, accepts category match:
          - For whale classes: accepts any whale class.
          - For non-whale classes: accepts any non-whale class.
    """
    actual = result["global_prediction_label"]

    whale_classes = {"resident", "transient", "humpback"}
    non_whale_classes = {"water", "vessel", "human", "jingle"}

    if allow_category_match:
        # Category match mode.
        if audio_type in whale_classes:
            # Accept any whale class.
            assert actual in whale_classes, (
                f"HuggingFace model predicted '{actual}' for {audio_type} audio, "
                f"but expected one of {whale_classes}"
            )
        elif audio_type in non_whale_classes:
            # Accept any non-whale class.
            assert actual in non_whale_classes, (
                f"HuggingFace model predicted '{actual}' for {audio_type} audio, "
                f"but expected one of {non_whale_classes}"
            )
    else:
        # Exact match required.
        assert actual == audio_type, (
            f"HuggingFace model predicted '{actual}' for {audio_type} audio, "
            f"but expected exact match '{audio_type}'"
        )


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


# ---------------------------------------------------------------------------
# Integration tests with real models (if available)
# ---------------------------------------------------------------------------

class TestIntegrationWithRealModels:
    """
    Integration tests that run inference on real wav files with real models.

    These tests are skipped if the required models or wav files are not present.
    They verify end-to-end functionality with actual model weights.
    """

    # Shared fixtures for model paths.
    @pytest.fixture
    def fastai_model_path(self) -> str:
        """Path to the FastAI model directory."""
        path = Path("model")
        if not path.exists():
            pytest.skip(f"FastAI model directory not found: {path}")
        return str(path)

    @pytest.fixture
    def huggingface_model_path(self) -> str:
        """Path to the HuggingFace multiclass model directory."""
        path = Path("model/multiclass")
        if not path.exists():
            pytest.skip(f"HuggingFace model directory not found: {path}")
        return str(path)

    # Fixtures for test wav files (one per audio type).
    def _get_testing_wav_path(self, category: str) -> str:
        """Return one testing wav path for the given category, or skip if unavailable."""
        candidates = sorted(Path(f"output/testing-wav/{category}").glob("*.wav"))
        if not candidates:
            pytest.skip(f"No testing {category} wav file found in output/testing-wav/{category}")
        return str(candidates[0])

    @pytest.fixture
    def resident_wav_path(self) -> str:
        """Path to a real resident orca wav file for testing."""
        path = Path("output/wav/resident/rpi-andrews-bay_2026_02_16_22_52_59_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def transient_wav_path(self) -> str:
        """Path to a real transient orca wav file for testing."""
        path = Path("output/wav/transient/rpi-sunset-bay_2024_12_19_12_39_03_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def humpback_wav_path(self) -> str:
        """Path to a real humpback whale wav file for testing."""
        path = Path("output/wav/humpback/rpi-orcasound-lab_2025_12_19_04_55_55_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def vessel_wav_path(self) -> str:
        """Path to a real vessel noise wav file for testing."""
        path = Path("output/wav/vessel/rpi-mast-center_2026_01_26_19_01_25_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def water_wav_path(self) -> str:
        """Path to a real water/ambient noise wav file for testing."""
        path = Path("output/wav/water/rpi-bush-point_2025_06_29_04_19_09_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def human_wav_path(self) -> str:
        """Path to a real human voice wav file for testing."""
        path = Path("output/wav/human/rpi-sunset-bay_2024_07_23_11_32_48_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def jingle_wav_path(self) -> str:
        """Path to a real jingle/signal wav file for testing."""
        path = Path("output/wav/jingle/rpi-north-sjc_2024_11_01_00_47_53_PST.wav")
        if not path.exists():
            pytest.skip(f"Test wav file not found: {path}")
        return str(path)

    @pytest.fixture
    def testing_resident_wav_path(self) -> str:
        """Path to a 60-second testing resident wav file."""
        return self._get_testing_wav_path("resident")

    @pytest.fixture
    def testing_transient_wav_path(self) -> str:
        """Path to a 60-second testing transient wav file."""
        return self._get_testing_wav_path("transient")

    @pytest.fixture
    def testing_humpback_wav_path(self) -> str:
        """Path to a 60-second testing humpback wav file."""
        return self._get_testing_wav_path("humpback")

    @pytest.fixture
    def testing_vessel_wav_path(self) -> str:
        """Path to a 60-second testing vessel wav file."""
        return self._get_testing_wav_path("vessel")

    @pytest.fixture
    def testing_water_wav_path(self) -> str:
        """Path to a 60-second testing water wav file."""
        return self._get_testing_wav_path("water")

    @pytest.fixture
    def testing_human_wav_path(self) -> str:
        """Path to a 60-second testing human wav file."""
        return self._get_testing_wav_path("human")

    @pytest.fixture
    def testing_jingle_wav_path(self) -> str:
        """Path to a 60-second testing jingle wav file."""
        return self._get_testing_wav_path("jingle")

    # Parametrized tests for FastAI model on different audio types.
    @pytest.mark.parametrize("wav_fixture,label", [
        ("resident_wav_path", "resident"),
        ("transient_wav_path", "transient"),
        ("humpback_wav_path", "humpback"),
        ("vessel_wav_path", "vessel"),
        ("water_wav_path", "water"),
        ("human_wav_path", "human"),
        ("jingle_wav_path", "jingle"),
    ])
    def test_fastai_model_inference(
        self,
        wav_fixture: str,
        label: str,
        fastai_model_path: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test FastAI model inference on various audio types."""
        from run_inference import run_inference

        wav_path = request.getfixturevalue(wav_fixture)
        result = run_inference(wav_path, model_type="fastai", model_path=fastai_model_path)

        _verify_fastai_result_structure(result)
        _verify_fastai_prediction(result, label)
        _print_fastai_result(result, label)

    # Parametrized tests for HuggingFace model on different audio types.
    @pytest.mark.parametrize("wav_fixture,label,xfail_reason", [
        ("resident_wav_path", "resident", None),
        ("transient_wav_path", "transient", None),
        ("humpback_wav_path", "humpback", None),
        ("vessel_wav_path", "vessel", None),
        ("water_wav_path", "water", None),
        ("human_wav_path", "human", None),
        ("jingle_wav_path", "jingle", None),
    ])
    def test_huggingface_model_inference(
        self,
        wav_fixture: str,
        label: str,
        xfail_reason: Optional[str],
        huggingface_model_path: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test HuggingFace model inference on various audio types."""
        from run_inference import run_inference
        
        # Apply xfail marker if this test case is expected to fail.
        if xfail_reason:
            request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))

        wav_path = request.getfixturevalue(wav_fixture)
        result = run_inference(wav_path, model_type="huggingface", model_path=huggingface_model_path)

        _verify_huggingface_result_structure(result)
        # Always require exact match - no category matching allowed.
        _verify_huggingface_prediction(result, label, allow_category_match=False)
        _print_huggingface_result(result, label)

    # Parametrized CLI integration tests.
    @pytest.mark.parametrize("wav_fixture,model_type,model_path_fixture", [
        ("resident_wav_path", "fastai", "fastai_model_path"),
        ("resident_wav_path", "huggingface", "huggingface_model_path"),
        ("transient_wav_path", "fastai", "fastai_model_path"),
        ("transient_wav_path", "huggingface", "huggingface_model_path"),
        ("humpback_wav_path", "fastai", "fastai_model_path"),
        ("humpback_wav_path", "huggingface", "huggingface_model_path"),
        ("vessel_wav_path", "fastai", "fastai_model_path"),
        ("vessel_wav_path", "huggingface", "huggingface_model_path"),
        ("water_wav_path", "fastai", "fastai_model_path"),
        ("water_wav_path", "huggingface", "huggingface_model_path"),
        ("human_wav_path", "fastai", "fastai_model_path"),
        ("human_wav_path", "huggingface", "huggingface_model_path"),
        ("jingle_wav_path", "fastai", "fastai_model_path"),
        ("jingle_wav_path", "huggingface", "huggingface_model_path"),
    ])
    def test_cli_integration(
        self,
        wav_fixture: str,
        model_type: str,
        model_path_fixture: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test CLI integration with various audio types and models."""
        from run_inference import main

        wav_path = request.getfixturevalue(wav_fixture)
        model_path = request.getfixturevalue(model_path_fixture)

        with patch("sys.argv", [
            "run_inference.py",
            wav_path,
            "--model", model_type,
            "--model-path", model_path
        ]):
            exit_code = main()

        assert exit_code == 0

    @pytest.mark.parametrize("wav_fixture,label", [
        ("testing_resident_wav_path", "resident"),
        ("testing_transient_wav_path", "transient"),
        ("testing_humpback_wav_path", "humpback"),
        ("testing_vessel_wav_path", "vessel"),
        ("testing_water_wav_path", "water"),
        ("testing_human_wav_path", "human"),
        ("testing_jingle_wav_path", "jingle"),
    ])
    def test_huggingface_model_inference_on_testing_wavs(
        self,
        wav_fixture: str,
        label: str,
        huggingface_model_path: str,
        request: pytest.FixtureRequest
    ) -> None:
        """Test HuggingFace inference on one 60-second testing wav per category."""
        from run_inference import run_inference

        wav_path = request.getfixturevalue(wav_fixture)
        result = run_inference(wav_path, model_type="huggingface", model_path=huggingface_model_path)

        _verify_huggingface_result_structure(result)
        # Require exact label match for per-category testing WAV fixtures.
        _verify_huggingface_prediction(result, label, allow_category_match=False)
        _print_huggingface_result(result, f"testing-{label}")
