# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Unit tests for PODS-AI LiveInferenceOrchestrator helpers."""

from unittest.mock import Mock, patch

import pytest

import LiveInferenceOrchestrator as orchestrator


def test_get_model_id_appends_release_version() -> None:
    """The logged model ID should identify the deployed container release."""
    model_id = orchestrator.get_model_id(
        {"model_id": "davethaler/whale-call-detector"},
        release_version="v1.6.0",
    )

    assert model_id == "davethaler/whale-call-detector-v1.6.0"


def test_get_model_id_without_release_version_is_unchanged() -> None:
    """Local and test runs without release metadata should keep the configured ID."""
    model_id = orchestrator.get_model_id(
        {"model_id": "davethaler/whale-call-detector"}
    )

    assert model_id == "davethaler/whale-call-detector"


def test_legacy_thresholds_and_count_keep_strict_gate_disabled() -> None:
    config = orchestrator.resolve_detection_config(
        {"threshold": 0.5, "min_num_positive_calls_threshold": 3}
    )

    assert config["pred_local_threshold"] == 0.5
    assert config["pred_global_threshold"] == 0.5
    assert config["require_n_positive_segments"] == 3
    assert config["use_strict_require_n"] is False
    assert orchestrator.evaluate_global_detection([0.95, 0.2, 0.2], config)[
        "global_detection"
    ]


def test_split_thresholds_have_independent_effects() -> None:
    config = orchestrator.resolve_detection_config(
        {
            "pred_local_threshold": 0.7,
            "pred_global_threshold": 0.65,
            "aggregation_strategy": "mean_top_k",
            "mean_top_k": 2,
        }
    )

    assert orchestrator.evaluate_global_detection([0.8, 0.6], config) == {
        "global_confidence": 0.7,
        "num_local_positives": 1,
        "global_detection": True,
    }
    config["pred_local_threshold"] = 0.9
    assert orchestrator.evaluate_global_detection([0.8, 0.6], config)[
        "global_detection"
    ]
    config["pred_global_threshold"] = 0.71
    assert not orchestrator.evaluate_global_detection([0.8, 0.6], config)[
        "global_detection"
    ]


def test_thresholded_aggregation_averages_continuous_scores() -> None:
    config = orchestrator.resolve_detection_config(
        {"pred_local_threshold": 0.5, "aggregation_strategy": "mean_thresholded"}
    )

    assert orchestrator.aggregate_local_scores([0.95, 0.8, 0.2], config) == 0.875


def test_strict_require_n_rejects_single_high_confidence_segment() -> None:
    config = orchestrator.resolve_detection_config(
        {
            "pred_local_threshold": 0.5,
            "pred_global_threshold": 0.5,
            "require_n_positive_segments": 2,
            "use_strict_require_n": True,
        }
    )

    assert not orchestrator.evaluate_global_detection(
        [0.95, 0.2, 0.2], config
    )["global_detection"]


def test_explicit_require_n_overrides_legacy_count() -> None:
    config = orchestrator.resolve_detection_config(
        {
            "pred_threshold": 0.4,
            "pred_local_threshold": 0.3,
            "min_num_positive_calls_threshold": 4,
            "require_n_positive_segments": 1,
            "use_strict_require_n": True,
        }
    )

    assert config["pred_local_threshold"] == 0.3
    assert config["pred_global_threshold"] == 0.4
    assert config["require_n_positive_segments"] == 1
    assert orchestrator.evaluate_global_detection([0.4], config)["global_detection"]


def test_empty_scores_zero_requirement_and_threshold_equality() -> None:
    config = orchestrator.resolve_detection_config(
        {
            "pred_local_threshold": 0.5,
            "pred_global_threshold": 0.5,
            "require_n_positive_segments": 1,
            "use_strict_require_n": True,
        }
    )
    assert orchestrator.evaluate_global_detection([0.5, 0.4], config) == {
        "global_confidence": 0.5,
        "num_local_positives": 1,
        "global_detection": True,
    }

    config["require_n_positive_segments"] = 0
    assert orchestrator.evaluate_global_detection([0.5], config)["global_detection"]
    assert orchestrator.evaluate_global_detection([], config) == {
        "global_confidence": 0.0,
        "num_local_positives": 0,
        "global_detection": False,
    }


def test_negative_require_n_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        orchestrator.resolve_detection_config({"require_n_positive_segments": -1})


@patch("LiveInferenceOrchestrator.get_model_inference")
def test_load_model_maps_split_and_legacy_config(mock_get_model) -> None:
    orchestrator.load_model(
        {
            "threshold": 0.4,
            "pred_local_threshold": 0.3,
            "pred_global_threshold": 0.8,
            "min_num_positive_calls_threshold": 5,
            "require_n_positive_segments": 1,
        },
        Mock(),
    )

    kwargs = mock_get_model.call_args.kwargs
    assert kwargs["threshold"] == 0.3
    assert kwargs["min_num_positive_calls_threshold"] == 5


@pytest.mark.parametrize(("strict", "expected_detection"), [(False, True), (True, False)])
def test_process_segment_reconciles_uploaded_label_and_tags(
    tmp_path, strict: bool, expected_detection: bool
) -> None:
    local_confidences = [0.95] + [0.2] * 28
    result = {
        "local_predictions": [1] + [0] * 28,
        "local_confidences": local_confidences,
        "global_prediction_label": "resident" if strict else "water",
        "global_prediction_labels": ["resident"] if strict else [],
    }
    model = Mock()
    model.id2label = {0: "water", 1: "resident"}
    model.predict.return_value = result
    segment = Mock()
    segment.folder_epoch = 0
    segment.start_index = 0
    segment.end_index = 30
    segment.start_iso = "2026-01-01T00:00:00Z"
    segment.duration_s = 60.0
    segment.download_as_wav.return_value = str(tmp_path / "clip.wav")
    spectrogram_module = Mock()
    spectrogram_module.write_spectrogram.return_value = str(tmp_path / "clip.png")
    config = {
        "hls_hydrophone_id": "rpi_orcasound_lab",
        "pred_local_threshold": 0.5,
        "pred_global_threshold": 0.5,
        "min_num_positive_calls_threshold": 3,
        "require_n_positive_segments": 3,
        "use_strict_require_n": strict,
        "delete_local_wavs": False,
        "upload_to_azure": True,
    }

    with patch.dict("sys.modules", {"spectrogram_visualizer": spectrogram_module}), patch.object(
        orchestrator, "upload_detection_to_azure"
    ) as upload:
        orchestrator._process_segment(
            segment,
            model,
            config,
            Mock(),
            Mock(),
            Mock(),
            "podsai-model",
            str(tmp_path),
        )

    assert result["global_detection"] is expected_detection
    if not expected_detection:
        upload.assert_not_called()
        assert result["global_prediction_label"] == "resident"
        return

    upload.assert_called_once()
    uploaded_result = upload.call_args.args[2]
    metadata = orchestrator.build_cosmosdb_metadata(
        audio_uri="audio-uri",
        image_uri="image-uri",
        result=uploaded_result,
        timestamp_in_iso="2026-01-01T00:00:00Z",
        source_guid="rpi_orcasound_lab",
        model_id="podsai-model",
        id2label=model.id2label,
    )
    assert metadata["globalPredictionLabel"] == "resident"
    assert metadata["modelGlobalPredictionLabel"] == "water"
    assert metadata["tags"] == "srkw"


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


def test_build_cosmosdb_metadata_includes_multiclass_fields() -> None:
    """Metadata should include global prediction label, positive segment list, and tags."""
    result = {
        "local_predictions": [2, 0],
        "local_confidences": [0.90, 0.20],
        "global_confidence": 0.9,
        "global_prediction_label": "transient",
        "num_local_positives": 1,
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
    assert metadata["numLocalPositives"] == 1
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
