#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import wave


def test_fastai_predict_uses_train_dataset_items_when_valid_loader_is_empty(tmp_path):
    """split_none() keeps generated clips in train_ds.x even when testdb.x is empty."""
    from model_inference import FastAIModel

    positive_confidences = [0.25, 0.75]

    wav_path = tmp_path / "example.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16)

    def fake_extract_segments(_audio_path, sample_dict, dest_dir, _suffix):
        for wav_name, segments in sample_dict.items():
            stem = Path(wav_name).stem.lower()
            for begin_time, end_time in segments:
                (Path(dest_dir) / f"{stem}_{begin_time}_{end_time}.wav").write_bytes(b"")

    mock_loaded_model = MagicMock()
    mock_loaded_model.model = MagicMock()
    mock_loaded_model.predict.side_effect = [
        (None, None, [0.0, positive_confidences[0]]),
        (None, None, [0.0, positive_confidences[1]]),
    ]

    score_items = MagicMock()
    score_items.items = [
        Path(tmp_path / "example_0_3.wav"),
        Path(tmp_path / "example_1_4.wav"),
    ]
    score_items.__iter__.return_value = iter(["segment-a", "segment-b"])

    fake_testdb = SimpleNamespace(
        x=[],
        train_ds=SimpleNamespace(x=score_items),
    )

    fake_test = MagicMock()
    fake_test.transform.return_value.databunch.return_value = fake_testdb

    with patch("model_inference.load_model", return_value=mock_loaded_model), \
         patch("model_inference.get_duration", return_value=4.0), \
         patch("model_inference.extract_segments", side_effect=fake_extract_segments) as mock_extract_segments, \
         patch("model_inference.gc.collect"), \
         patch("model_inference.torch.cuda.is_available", return_value=False), \
         patch("audio.data.AudioList.from_folder") as mock_from_folder:
        mock_from_folder.return_value.split_none.return_value.label_empty.return_value = fake_test

        model = FastAIModel(
            model_path="model",
            smooth_predictions=False,
            min_num_positive_calls_threshold=1,
        )
        result = model.predict(str(wav_path))

    assert result["local_confidences"] == positive_confidences
    assert result["local_predictions"] == [0, 1]
    assert result["global_prediction"] == 1
    assert result["submission"]["start_time_s"].tolist() == [0, 1]
    assert mock_loaded_model.predict.call_count == 2
    mock_loaded_model.predict.assert_any_call("segment-a")
    mock_loaded_model.predict.assert_any_call("segment-b")
    mock_extract_segments.assert_called_once()
