import sys
from types import SimpleNamespace

import numpy as np

from podsai_onnx_inference import PodsAIOnnxInference


def _inference_for_logits(monkeypatch, logits):
    audio = np.zeros(((len(logits) - 1) * 2 + 3) * 16000)
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(load=lambda *args, **kwargs: (audio, 16000)),
    )

    class FakeSession:
        def run(self, output_names, inputs):
            return [np.asarray(logits)]

    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference.threshold = 0.5
    inference.min_num_positive_calls_threshold = 2
    inference.inference_batch_size = 32
    inference.sess = FakeSession()
    inference._input_name = "input_values"
    inference.id2label = {0: "water", 1: "resident", 2: "other"}
    inference.label2id = {"water": 0, "resident": 1, "other": 2}
    inference.negative_class_ids = {0, 2}
    inference._compute_input_values = lambda *args: np.zeros(
        (len(logits), 3, 2), dtype=np.float32
    )
    return inference


def test_per_class_probabilities_average_matching_local_confidences(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(5 * 16000), 16000)),
    )

    class FakeSession:
        def run(self, output_names, inputs):
            return [np.array([[4.0, 3.0, 0.0], [3.0, 4.0, 0.0]])]

    inference = PodsAIOnnxInference.__new__(PodsAIOnnxInference)
    inference.threshold = 0.5
    inference.min_num_positive_calls_threshold = 2
    inference.inference_batch_size = 8
    inference.sess = FakeSession()
    inference._input_name = "input_values"
    inference.id2label = {0: "water", 1: "resident", 2: "other"}
    inference.label2id = {"water": 0, "resident": 1, "other": 2}
    inference.negative_class_ids = {0, 2}
    inference._compute_input_values = lambda *args: np.zeros((2, 3, 2), dtype=np.float32)

    result = inference.predict("sample.wav")

    assert result["local_predictions"] == [0, 1]
    assert result["per_class_probabilities"] == {
        "water": result["local_confidences"][0],
        "resident": result["local_confidences"][1],
        "other": 0.0,
    }


def test_global_prediction_labels_exclude_positive_classes_below_threshold(monkeypatch):
    logits = np.log([[0.35, 0.4, 0.25]] * 2)
    inference = _inference_for_logits(monkeypatch, logits)

    result = inference.predict("sample.wav")

    assert result["global_prediction_labels"] == []
    assert result["global_prediction_label"] == "other"


def test_global_prediction_labels_require_enough_non_adjacent_events(monkeypatch):
    logits = np.log(
        [[0.1, 0.8, 0.1]] * 2 + [[0.8, 0.1, 0.1]] * 9
    )
    inference = _inference_for_logits(monkeypatch, logits)

    result = inference.predict("sample.wav")

    assert result["global_prediction_labels"] == ["water"]
    assert result["global_prediction_label"] == "water"
