#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import wave


def test_fastai_predict_with_empty_valid_loader(tmp_path):
    """FastAI model prediction works when split_none() is used."""
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
                segment_path = Path(dest_dir) / f"{stem}_{begin_time}_{end_time}.wav"
                with wave.open(str(segment_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b"\x00\x00" * 16)

    mock_loaded_model = MagicMock()
    mock_loaded_model.model = MagicMock()
    mock_loaded_model.predict.side_effect = [
        (None, None, [0.0, positive_confidences[0]]),
        (None, None, [0.0, positive_confidences[1]]),
    ]

    score_paths = [
        Path(tmp_path / "example_0_3.wav"),
        Path(tmp_path / "example_1_4.wav"),
    ]
    score_items = MagicMock()
    score_items.items = score_paths
    score_items.__iter__.return_value = iter(score_paths)

    # When split_none() is used, items are in testdb.x (not in a separate validation set)
    fake_testdb = SimpleNamespace(
        x=score_items,  # Items to score are here
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
    mock_loaded_model.predict.assert_any_call(score_paths[0])
    mock_loaded_model.predict.assert_any_call(score_paths[1])
    mock_extract_segments.assert_called_once()


def test_fastai_predict_prefers_populated_testdb_x(tmp_path):
    """Use testdb.x when it already exposes generated clips."""
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
                segment_path = Path(dest_dir) / f"{stem}_{begin_time}_{end_time}.wav"
                with wave.open(str(segment_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b"\x00\x00" * 16)

    mock_loaded_model = MagicMock()
    mock_loaded_model.model = MagicMock()
    mock_loaded_model.predict.side_effect = [
        (None, None, [0.0, positive_confidences[0]]),
        (None, None, [0.0, positive_confidences[1]]),
    ]

    score_paths = [
        Path(tmp_path / "example_0_3.wav"),
        Path(tmp_path / "example_1_4.wav"),
    ]
    score_items = MagicMock()
    score_items.items = score_paths
    score_items.__iter__.return_value = iter(score_paths)
    score_items.__len__.return_value = len(score_paths)

    empty_score_items = MagicMock()
    empty_score_items.items = []
    empty_score_items.__iter__.return_value = iter([])
    empty_score_items.__len__.return_value = 0

    fake_testdb = SimpleNamespace(
        x=score_items,
        train_ds=SimpleNamespace(x=empty_score_items),
    )

    fake_test = MagicMock()
    fake_test.transform.return_value.databunch.return_value = fake_testdb

    with patch("model_inference.load_model", return_value=mock_loaded_model), \
         patch("model_inference.get_duration", return_value=4.0), \
         patch("model_inference.extract_segments", side_effect=fake_extract_segments), \
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
    mock_loaded_model.predict.assert_any_call(score_paths[0])
    mock_loaded_model.predict.assert_any_call(score_paths[1])
