# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for PODS-AI LiveInferenceOrchestrator helpers."""

from unittest.mock import Mock, patch

import LiveInferenceOrchestrator as orchestrator


@patch("LiveInferenceOrchestrator.requests.get")
def test_orcasound_attributes_pulled_from_feed(mock_get) -> None:
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "data": [
            {
                "id": "feed_02u8r4EPgmlYQmh6gzlGIL",
                "attributes": {
                    "name": "Andrews Bay",
                    "node_name": "rpi_andrews_bay",
                    "location_point": {
                        "coordinates": [-123.166408, 48.546653]
                    }
                }
            }
        ]
    }
    mock_get.return_value = mock_response

    result_dict = orchestrator.get_hydro_attributes_from_feed()
    assert result_dict["rpi_andrews_bay"]["name"] == "Andrews Bay"


def test_is_positive_label_uses_multiclass_negative_set() -> None:
    """Negative classes should be non-detections, whale classes should be detections."""
    assert orchestrator.is_positive_label("resident")
    assert orchestrator.is_positive_label("transient")
    assert orchestrator.is_positive_label("humpback")

    assert not orchestrator.is_positive_label("other")
    assert not orchestrator.is_positive_label("abiotic")
    assert not orchestrator.is_positive_label("water")
    assert not orchestrator.is_positive_label("vessel")
    assert not orchestrator.is_positive_label("jingle")
    assert not orchestrator.is_positive_label("human")
    assert not orchestrator.is_positive_label("bird")


def test_build_prediction_list_filters_negative_labels() -> None:
    """Only positive local segment labels should be included in prediction rows."""
    result = {
        "local_predictions": [1, 0, 3, 4],
        "local_confidences": [0.95, 0.10, 0.80, 0.05],
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }
    id2label = {
        0: "water",
        1: "resident",
        2: "transient",
        3: "humpback",
        4: "vessel",
    }

    predictions = orchestrator.build_prediction_list(result, id2label=id2label)

    assert len(predictions) == 2
    assert predictions[0]["label"] == "resident"
    assert predictions[0]["startTime"] == 0.0
    assert predictions[0]["duration"] == 3.0
    assert predictions[1]["label"] == "humpback"
    assert predictions[1]["startTime"] == 4.0


def test_apply_model_config_overrides_deep_merges_nested_values() -> None:
    """Nested model config overrides should override only the targeted keys."""
    base = {
        "threshold": 0.5,
        "inference": {"max_batch_size": 16, "window_hop_s": 2.0},
        "global_prediction": {"pred_global_threshold": 0.5},
    }
    overrides = {
        "inference": {"max_batch_size": 8},
        "global_prediction": {"pred_global_threshold": 0.8},
    }

    result = orchestrator.apply_model_config_overrides(base, overrides)

    assert result["threshold"] == 0.5
    assert result["inference"]["max_batch_size"] == 8
    assert result["inference"]["window_hop_s"] == 2.0
    assert result["global_prediction"]["pred_global_threshold"] == 0.8


def test_get_effective_model_config_resolves_model_id_from_effective_config() -> None:
    """Metadata model IDs should follow the resolved config, while explicit model_id wins."""
    config_with_override = {
        "model_hf_repo_id": "custom/model",
        "model_config_overrides": {"inference": {"max_batch_size": 8}},
    }
    effective = orchestrator.get_effective_model_config(config_with_override, Mock())
    assert effective["model_hf_repo_id"] == "custom/model"
    assert effective["inference"]["max_batch_size"] == 8
    assert (
        effective.get(
            "model_id",
            effective.get("model_hf_repo_id", orchestrator.PODSAI_MODEL_ID),
        )
        == "custom/model"
    )

    explicit_config = {
        "model_id": "explicit/model",
        "model_hf_repo_id": "custom/model",
    }
    explicit_effective = orchestrator.get_effective_model_config(explicit_config, Mock())
    assert explicit_effective.get(
        "model_id",
        explicit_effective.get("model_hf_repo_id", orchestrator.PODSAI_MODEL_ID),
    ) == "explicit/model"


def test_process_segment_applies_inference_overrides() -> None:
    segment = Mock(
        folder_epoch=1,
        start_index=0,
        end_index=1,
        start_iso="2026-01-01T00:00:00Z",
        duration_s=60.0,
    )
    segment.download_as_wav.return_value = "clip.wav"
    model = Mock()
    model.predict.return_value = {
        "global_prediction_label": "resident",
        "global_confidence": 0.7,
        "local_predictions": [1],
    }
    config = {
        "hls_hydrophone_id": "test",
        "delete_local_wavs": True,
        "upload_to_azure": False,
        "model_config_overrides": {
            "inference": {"window_hop_s": 1.5},
            "global_prediction": {"pred_global_threshold": 0.8},
        },
    }
    logger = Mock()

    with patch("LiveInferenceOrchestrator.os.remove"):
        orchestrator._process_segment(
            segment, model, config, None, None, logger, "model", "."
        )

    model.predict.assert_called_once_with(
        "clip.wav", segment_duration=3, hop_duration=1.5
    )
    assert not any(
        call.args[0].startswith("Whale call detected")
        for call in logger.info.call_args_list
    )


def test_build_cosmosdb_metadata_includes_multiclass_fields() -> None:
    """Metadata should include global prediction label, positive segment list, and tags."""
    result = {
        "local_predictions": [2, 0],
        "local_confidences": [0.90, 0.20],
        "global_confidence": 0.9,
        "global_prediction_label": "transient",
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }
    id2label = {0: "water", 2: "transient"}

    metadata = orchestrator.build_cosmosdb_metadata(
        audio_uri="audio-uri",
        image_uri="image-uri",
        result=result,
        timestamp_in_iso="2026-01-01T00:00:00Z",
        source_guid="unknown_feed",
        model_id="podsai-model",
        id2label=id2label,
    )

    assert metadata["modelId"] == "podsai-model"
    assert metadata["globalPredictionLabel"] == "transient"
    assert metadata["whaleFoundConfidence"] == 90.0
    assert metadata["location"]["id"] == "unknown_feed"
    assert len(metadata["predictions"]) == 1
    assert metadata["predictions"][0]["label"] == "transient"
    assert metadata["tags"] == "transient;water"


def test_build_cosmosdb_metadata_comments_appends_dominant_extra_class() -> None:
    """comments field should append the most common vessel/human/jingle/bird label when it differs from the global prediction."""
    result = {
        "local_predictions": ["resident", "vessel", "vessel"],
        "local_confidences": [0.80, 0.70, 0.65],
        "global_confidence": 0.8,
        "global_prediction_label": "resident",
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }

    metadata = orchestrator.build_cosmosdb_metadata(
        audio_uri="audio-uri",
        image_uri="image-uri",
        result=result,
        timestamp_in_iso="2026-01-01T00:00:00Z",
        source_guid="rpi_orcasound_lab",
        model_id="podsai-model",
    )

    # Tags should only include srkw (positive label) and negative local (vessel)
    assert metadata["tags"] == "srkw;vessel"


def test_build_cosmosdb_metadata_tags_include_all_positive_predictions() -> None:
    """tags field should include all unique positive labels from local and global predictions."""
    result = {
        "local_predictions": ["resident", "humpback", "resident"],
        "local_confidences": [0.80, 0.70, 0.65],
        "global_confidence": 0.8,
        "global_prediction_label": "resident",
        "hop_duration": 2.0,
        "segment_duration": 3.0,
    }

    metadata = orchestrator.build_cosmosdb_metadata(
        audio_uri="audio-uri",
        image_uri="image-uri",
        result=result,
        timestamp_in_iso="2026-01-01T00:00:00Z",
        source_guid="rpi_orcasound_lab",
        model_id="podsai-model",
    )

    assert metadata["tags"] == "srkw"


def test_upload_detection_to_azure_skips_existing_blobs(tmp_path) -> None:
    """BlobAlreadyExists upload races should be treated as skip/no-op."""
    clip_path = tmp_path / "existing.wav"
    clip_path.write_bytes(b"clip")
    spectrogram_path = tmp_path / "existing.png"
    spectrogram_path.write_bytes(b"spectrogram")

    result = {
        "local_predictions": [],
        "local_confidences": [],
        "global_confidence": 0.9,
        "global_prediction_label": "resident",
    }

    class BlobAlreadyExistsError(Exception):
        error_code = "BlobAlreadyExists"

    audio_blob_client = Mock()
    audio_blob_client.upload_blob.side_effect = BlobAlreadyExistsError()
    spectrogram_blob_client = Mock()
    spectrogram_blob_client.upload_blob.side_effect = BlobAlreadyExistsError()
    blob_service_client = Mock()
    blob_service_client.get_blob_client.side_effect = [
        audio_blob_client,
        spectrogram_blob_client,
    ]

    container = Mock()
    database = Mock()
    database.get_container_client.return_value = container
    cosmos_client = Mock()
    cosmos_client.get_database_client.return_value = database
    logger = Mock()

    orchestrator.upload_detection_to_azure(
        clip_path=str(clip_path),
        spectrogram_path=str(spectrogram_path),
        result=result,
        start_timestamp="2026-01-01T00:00:00Z",
        hls_hydrophone_id="rpi_orcasound_lab",
        model_id="podsai-model",
        blob_service_client=blob_service_client,
        cosmos_client=cosmos_client,
        logger=logger,
    )

    audio_blob_client.exists.assert_not_called()
    spectrogram_blob_client.exists.assert_not_called()
    audio_blob_client.upload_blob.assert_called_once()
    spectrogram_blob_client.upload_blob.assert_called_once()
    logger.info.assert_any_call("Blob already exists, skipping upload: existing.wav")
    logger.info.assert_any_call("Blob already exists, skipping upload: existing.png")
