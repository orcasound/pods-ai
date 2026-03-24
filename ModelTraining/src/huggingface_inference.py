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
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class HuggingFaceInference:
    """
    Inference wrapper for HuggingFace audio classification models.
    
    This class implements the interface expected by model_inference.py,
    providing predictions in the format required by extract_training_samples.py.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the inference model.
        
        Args:
            model_path: Path to model directory or HuggingFace Hub model ID
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading HuggingFace model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print(f"Model loaded successfully. Labels: {list(self.label2id.keys())}")
    
    def predict(self, wav_path: str, segment_duration: int = 3) -> dict:
        """
        Run inference on a wav file.
        
        Args:
            wav_path: Path to wav file (typically 60 seconds long)
            segment_duration: Duration of each segment in seconds (default: 3)
            
        Returns:
            Dictionary with keys:
                - local_confidences: List of confidence scores for each segment
                - prediction: Overall prediction label
                - confidence: Overall confidence score
        """
        # Load audio
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        
        # Split into segments
        segment_samples = segment_duration * sr
        num_segments = int(len(audio) / segment_samples)
        
        local_confidences = []
        
        print(f"  Processing {num_segments} segments...")
        
        with torch.no_grad():
            for i in range(num_segments):
                start = i * segment_samples
                end = start + segment_samples
                segment = audio[start:end]
                
                # Pad if necessary
                if len(segment) < segment_samples:
                    padding = segment_samples - len(segment)
                    segment = np.pad(segment, (0, padding), mode='constant')
                
                # Extract features
                inputs = self.feature_extractor(
                    segment,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True,
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get max probability (any positive class)
                max_prob = probs.max().item()
                local_confidences.append(max_prob)
        
        # Overall prediction is based on the segment with highest confidence
        max_idx = np.argmax(local_confidences)
        max_confidence = local_confidences[max_idx]
        
        # Get the predicted class for that segment
        start = max_idx * segment_samples
        end = start + segment_samples
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


def get_huggingface_inference(model_path: str, **kwargs):
    """
    Factory function to create HuggingFace inference instance.
    
    Args:
        model_path: Path to model directory or HuggingFace Hub model ID
        **kwargs: Additional arguments passed to HuggingFaceInference
        
    Returns:
        HuggingFaceInference instance
    """
    return HuggingFaceInference(model_path, **kwargs)