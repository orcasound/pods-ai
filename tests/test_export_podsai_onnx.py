import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import export_podsai_onnx


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def float(self):
        return self

    def to(self, _device):
        return self


def test_export_uses_cli_model_revision_and_batch_size(monkeypatch):
    model = Mock()
    model.audio_spectrogram_transformer = None
    feature_extractor = SimpleNamespace(
        max_length=2,
        num_mel_bins=4,
        hop_length=2,
    )
    inference = SimpleNamespace(
        model=model,
        feature_extractor=feature_extractor,
        _ensure_ast_position_embeddings=Mock(return_value=False),
        _compute_input_values=Mock(
            side_effect=lambda _audio, _sr, count, *_args: FakeTensor(
                np.zeros((count, 20, 4), dtype=np.float32)
            )
        ),
    )
    inference_factory = Mock(return_value=inference)
    monkeypatch.setitem(
        sys.modules,
        "podsai_inference",
        SimpleNamespace(PodsAIInference=inference_factory),
    )

    class FakeModule:
        def __init__(self):
            pass

        def to(self, _device):
            return self

    export = Mock()
    fake_torch = SimpleNamespace(
        device=lambda value: value,
        nn=SimpleNamespace(Module=FakeModule),
        from_numpy=FakeTensor,
        onnx=SimpleNamespace(export=export),
    )
    monkeypatch.setattr(export_podsai_onnx, "torch", fake_torch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_podsai_onnx.py",
            "--model",
            "example/model",
            "--revision",
            "0123456789abcdef",
            "--out",
            "model.onnx",
            "--batch-size",
            "3",
        ],
    )

    export_podsai_onnx.main()

    inference_factory.assert_called_once_with(
        "example/model", device="cpu", model_revision="0123456789abcdef"
    )
    assert inference._compute_input_values.call_args.args[2] == 3
    assert export.call_args.args[1][0].array.shape == (3, 20, 4)
    assert export.call_args.args[2] == "model.onnx"
    assert export.call_args.kwargs["dynamic_axes"] == {
        "input_values": {0: "batch"},
        "logits": {0: "batch"},
    }
