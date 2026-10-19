import sys
from types import SimpleNamespace

import numpy as np

from podsai_onnx_inference import PodsAIOnnxInference


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
    inference._compute_input_values = lambda *args: np.zeros((2, 3, 2), dtype=np.float32)

    result = inference.predict("sample.wav")

    assert result["local_predictions"] == [0, 1]
    assert result["per_class_probabilities"] == {
        "water": result["local_confidences"][0],
        "resident": result["local_confidences"][1],
        "other": 0.0,
    }
