#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
HuggingFace model inference wrapper for orca call detection.

This module provides a wrapper around HuggingFace audio classification models
that implements the interface expected by PODS-AI's model_inference system.
"""

import torch
import librosa
import numpy as np
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class HuggingFaceInference:
    """
    Inference wrapper for HuggingFace audio classification models.
    
    This class implements the interface expected by model_inference.py,
    providing predictions in the format required by extract_training_samples.py.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None,
                 threshold: float = 0.5, min_num_positive_calls_threshold: int = 3) -> None:
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            threshold: Confidence threshold for positive predictions (default: 0.5)
            min_num_positive_calls_threshold: Minimum positive predictions for global positive (default: 3)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold
        
        # Auto-detect device. Default to GPU if available, otherwise CPU. Allow override via argument.
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading HuggingFace model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load feature extractor and model.
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        except Exception as e:
            error_msg = f"Error loading feature extractor from {model_path}: {type(e).__name__}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {type(e).__name__}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Get label mapping. This assumes the model was trained with a config that includes id2label and label2id.
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print(f"Model loaded successfully. Labels: {list(self.label2id.keys())}")
        
        # Validate that model has required labels for orca call detection.
        # We require an "other" class to distinguish positive calls from background noise.
        if "other" not in self.label2id:
            raise ValueError(
                f"Model must include an 'other' label for background/negative class. "
                f"Found labels: {list(self.label2id.keys())}. "
                f"Please train the model with label mapping that includes 'other' as the negative class. "
                f"Expected labels: resident, transient, humpback, other"
            )
        
        # Warn if model doesn't have expected positive classes (but don't fail - could be binary model)
        expected_positive_labels = {"resident", "transient", "humpback"}
        found_positive_labels = expected_positive_labels & set(self.label2id.keys())
        if not found_positive_labels:
            print(
                f"Warning: Model has 'other' but no expected positive classes (resident, transient, humpback). "
                f"This may work but confidence scores may not be meaningful. "
                f"Found labels: {list(self.label2id.keys())}"
            )
        elif found_positive_labels != expected_positive_labels:
            missing = expected_positive_labels - found_positive_labels
            print(
                f"Warning: Model is missing some expected positive classes: {missing}. "
                f"Confidence will be computed from available classes: {found_positive_labels}"
            )
    
    def predict(self, wav_path: str, segment_duration: int = 3, hop_duration: int = 1,
                threshold: Optional[float] = None, min_num_positive_calls_threshold: Optional[int] = None) -> dict[str, object]:
        """
        Run inference on a wav file using sliding window.
        
        Uses a sliding window approach with configurable segment and hop duration to match
        the FastAI model behavior. Each element in local_confidences corresponds to a
        1-second position from the start of the audio file.
        
        Args:
            wav_path: Path to wav file (typically 60 seconds long)
            segment_duration: Duration of each segment in seconds (default: 3)
            hop_duration: Hop size in seconds between segments (default: 1)
            threshold: Confidence threshold for positive predictions (default: use instance value)
            min_num_positive_calls_threshold: Minimum positive predictions for global positive (default: use instance value)
            
        Returns:
            Dictionary with keys:
                - local_predictions: List of binary predictions (0 or 1) for each second
                - local_confidences: List of confidence scores (0.0-1.0), where index i represents second i
                - global_prediction: Overall binary prediction for the entire audio
                - global_confidence: Overall confidence score (0.0-1.0)
            Returns dict with empty lists and error values if audio loading fails.
        """
        # Use instance values if not overridden
        threshold = threshold if threshold is not None else self.threshold
        min_num_positive_calls_threshold = (
            min_num_positive_calls_threshold
            if min_num_positive_calls_threshold is not None
            else self.min_num_positive_calls_threshold
        )
        
        # Load audio. Resample to 16kHz and convert to mono. Handle exceptions gracefully.
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            error_msg = f"Error loading audio file {wav_path}: {type(e).__name__}: {e}"
            print(error_msg)
            return {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": "error",
                "global_confidence": 0.0,
            }
        
        # Handle empty audio. This can happen if the file is corrupted or has no valid audio data.
        if len(audio) == 0:
            print(f"Warning: Audio file {wav_path} is empty")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": "error",
                "global_confidence": 0.0,
            }
        
        # Calculate segment and hop sizes in samples.
        segment_samples = segment_duration * sr
        hop_samples = hop_duration * sr
        
        # Calculate total audio duration and number of segments with sliding window.
        audio_duration = len(audio) / sr
        
        # Generate segment predictions using sliding window with 1-second hop.
        # This ensures local_confidences[i] corresponds to second i from the start.
        segment_confidences: list[float] = []
        
        # For each starting position (in seconds), extract a segment_duration window.
        num_positions = int(np.floor(audio_duration)) - (segment_duration - 1)
        
        # Handle very short audio (shorter than segment_duration).
        # In this case, we will just process one segment starting at 0 seconds.
        if num_positions < 1:
            num_positions = 1
        
        print(f"  Processing {num_positions} positions with {segment_duration}s window, {hop_duration}s hop...")
        
        with torch.no_grad():
            for pos_idx in range(num_positions):
                start = pos_idx * hop_samples
                end = min(start + segment_samples, len(audio))
                segment = audio[start:end]
                
                # Pad if necessary (for short audio or last segment).
                if len(segment) < segment_samples:
                    padding = segment_samples - len(segment)
                    segment = np.pad(segment, (0, padding), mode='constant')
                
                # Extract features. This will handle padding internally if needed.
                inputs = self.feature_extractor(
                    segment,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True,
                )
                
                # Move to device. This is important for performance, especially if using GPU.
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions.
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Compute confidence for positive classes only (exclude "other").
                # Sum probabilities of all positive classes (resident, transient, humpback).
                positive_confidence = 0.0
                for label, idx in self.label2id.items():
                    if label != "other":
                        positive_confidence += probs[0, idx].item()
                
                segment_confidences.append(positive_confidence)
        
        # Guard against empty confidences list.
        # This can happen if the audio is too short or if there was an error during processing.
        if not segment_confidences:
            print(f"Warning: No segments processed for {wav_path}")
            return {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": "error",
                "global_confidence": 0.0,
            }
        
        # Apply rolling average to smooth predictions (matching FastAI behavior).
        # FastAI uses a 2-position rolling window, averaging overlapping segments.
        # Build local_confidences with exactly n entries for n segments.
        n = len(segment_confidences)
        local_confidences: list[float] = []
        
        for i in range(n):
            if i == 0:
                # First position: use first segment confidence directly
                local_confidences.append(segment_confidences[0])
            elif i == n - 1:
                # Last position: use last segment confidence directly
                local_confidences.append(segment_confidences[-1])
            else:
                # Middle positions: average previous and current segment
                avg_confidence = (segment_confidences[i - 1] + segment_confidences[i]) / 2.0
                local_confidences.append(avg_confidence)
        
        # Determine local predictions based on the threshold
        local_predictions = [1 if conf >= threshold else 0 for conf in local_confidences]
        
        # Global prediction is based on the sum of local predictions.
        # If the sum exceeds the threshold, predict positive (1), else negative (0).
        global_prediction = 1 if sum(local_predictions) >= min_num_positive_calls_threshold else 0
        
        # Global confidence is the average confidence of the segments contributing to the global positive prediction.
        # If predicting negative, set confidence to 0.
        global_confidence = (
            sum(np.array(local_confidences)[np.array(local_predictions) == 1]) / max(1, sum(local_predictions))
        ) if global_prediction == 1 else 0.0
        
        # Convert global prediction to label name (e.g., "other", "resident")
        global_prediction_label = (
            self.id2label[global_prediction] if global_prediction in self.id2label else "other"
        )
        
        return {
            "local_predictions": local_predictions,
            "local_confidences": local_confidences,
            "global_prediction": global_prediction_label,
            "global_confidence": global_confidence,
        }


def get_huggingface_inference(model_path: str, **kwargs) -> HuggingFaceInference:
    """
    Factory function to create HuggingFace inference instance.
    
    Args:
        model_path: Path to model directory or HuggingFace Hub model ID
        **kwargs: Additional arguments passed to HuggingFaceInference
        
    Returns:
        HuggingFaceInference instance
    """
    return HuggingFaceInference(model_path, **kwargs)
