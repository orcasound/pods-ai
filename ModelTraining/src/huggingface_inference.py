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
from pathlib import Path
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class HuggingFaceInference:
    """
    Inference wrapper for HuggingFace audio classification models.
    
    This class implements the interface expected by model_inference.py,
    providing predictions in the format required by extract_training_samples.py.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        
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
    
    def predict(self, wav_path: str, segment_duration: int = 3, hop_duration: int = 1) -> dict[str, object]:
        """
        Run inference on a wav file using sliding window.
        
        Uses a sliding window approach with configurable segment and hop duration to match
        the FastAI model behavior. Each element in local_confidences corresponds to a 
        1-second position from the start of the audio file.
        
        Args:
            wav_path: Path to wav file (typically 60 seconds long)
            segment_duration: Duration of each segment in seconds (default: 3)
            hop_duration: Hop size in seconds between segments (default: 1)
            
        Returns:
            Dictionary with keys:
                - local_confidences: List of confidence scores, where index i represents second i
                - prediction: Overall prediction label
                - confidence: Overall confidence score
            Returns empty dict with empty list if audio loading fails.
        """
        # Load audio. Resample to 16kHz and convert to mono. Handle exceptions gracefully.
        try:
            audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            error_msg = f"Error loading audio file {wav_path}: {type(e).__name__}: {e}"
            print(error_msg)
            return {
                "local_confidences": [],
                "prediction": "error",
                "confidence": 0.0,
            }
        
        # Handle empty audio. This can happen if the file is corrupted or has no valid audio data.
        if len(audio) == 0:
            print(f"Warning: Audio file {wav_path} is empty")
            return {
                "local_confidences": [],
                "prediction": "error",
                "confidence": 0.0,
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
                "local_confidences": [],
                "prediction": "error",
                "confidence": 0.0,
            }
        
        # Apply rolling average to smooth predictions (matching FastAI behavior).
        # FastAI uses a 2-position rolling window, averaging overlapping segments.
        local_confidences: list[float] = []
        
        # First position: use first segment confidence directly.
        local_confidences.append(segment_confidences[0])
        
        # Middle positions: average current and next segment.
        for i in range(len(segment_confidences) - 1):
            avg_confidence = (segment_confidences[i] + segment_confidences[i + 1]) / 2.0
            local_confidences.append(avg_confidence)
        
        # Last position: use last segment confidence directly.
        if len(segment_confidences) > 1:
            local_confidences.append(segment_confidences[-1])
        
        # Overall prediction is based on the segment with highest confidence.
        max_idx = np.argmax(local_confidences)
        max_confidence = local_confidences[max_idx]
        
        # Get the predicted class for the highest confidence segment.
        start = max_idx * hop_samples
        end = min(start + segment_samples, len(audio))
        best_segment = audio[start:end]
        
        if len(best_segment) < segment_samples:
            padding = segment_samples - len(best_segment)
            best_segment = np.pad(best_segment, (0, padding), mode='constant')
        
        with torch.no_grad():
            inputs = self.feature_extractor(
                best_segment,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_id = logits.argmax(-1).item()
            prediction = self.id2label[pred_id]
        
        return {
            "local_confidences": local_confidences,
            "prediction": prediction,
            "confidence": max_confidence,
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