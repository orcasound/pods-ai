#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from model_inference import FastAIModel


def test_fastai_predict_uses_filtered_dataset_paths(tmp_path):
    """FastAI prediction should use dataset-filtered paths, not raw directory listings."""
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"wav")

    model = FastAIModel.__new__(FastAIModel)
    model.model = MagicMock()
    model.model.predict.return_value = (None, None, [0.1, 0.9])
    model.threshold = 0.5
    model.min_num_positive_calls_threshold = 3
    model.use_gpu = False
    model.smooth_predictions = False
    model.batch_size = 32

    kept_segment = tmp_path / "kept_0_3.wav"
    dropped_segment = tmp_path / "dropped_1_4.wav"

    def fake_extract_segments(_audio_path, _sample_dict, destn_path, _suffix):
        Path(destn_path, kept_segment.name).write_bytes(b"segment")
        Path(destn_path, dropped_segment.name).write_bytes(b"")

    dataset = SimpleNamespace(
        x=SimpleNamespace(items=[kept_segment]),
    )
    dataset.split_none = lambda: dataset
    dataset.label_empty = lambda: dataset
    dataset.transform = lambda _tfms: dataset
    dataset.databunch = lambda bs: SimpleNamespace(x=["item0"])

    with patch("model_inference.get_duration", return_value=4.0), \
            patch("model_inference.extract_segments", side_effect=fake_extract_segments), \
            patch("audio.data.AudioList.from_folder", return_value=dataset), \
            patch("model_inference.gc.collect"), \
            patch("model_inference.torch.cuda.is_available", return_value=False):
        result = model.predict(str(wav_path))

    assert len(result["local_confidences"]) == 1
    assert result["local_confidences"] == [0.9]
    assert result["local_predictions"] == [1]
