#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Model inference module for scoring audio samples.

This module provides a simple interface for running model inference on audio files
to score 2-second segments. It can be extended to support different model types.

For production use, this can be integrated with aifororcas-livesystem's 
LiveInferenceOrchestrator.py which uses DateRangeHLSStream to download and score
audio from specific time ranges. See:
https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/src/LiveInferenceOrchestrator.py
https://github.com/orcasound/aifororcas-livesystem/blob/main/InferenceSystem/config/Test/Positive/FastAI_DateRangeHLS_AndrewsBay.yml

The pretrained FastAI model is typically named "model.pkl" and should be placed in
a "model" directory.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import warnings


class ModelInference:
    """
    Base class for model inference.
    
    This class provides an interface for scoring audio files. Subclasses should
    implement the actual model loading and inference logic.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model inference.
        
        Args:
            model_path: Optional path to the model file or directory.
        """
        self.model_path = model_path
        self.model = None
    
    def predict(self, wav_file_path: str) -> Dict:
        """
        Run inference on a wav file and return predictions.
        
        Args:
            wav_file_path: Path to the wav file to score.
            
        Returns:
            Dictionary containing:
                - local_predictions: List of binary predictions (0 or 1) for each 2-second segment
                - local_confidences: List of confidence scores for each 2-second segment
                - global_prediction: Overall binary prediction for the entire audio
                - global_confidence: Overall confidence score
        """
        raise NotImplementedError("Subclasses must implement predict()")


class FastAIModelInference(ModelInference):
    """
    FastAI model inference using the aifororcas model.
    
    This implementation uses the FastAI model architecture from aifororcas-livesystem.
    The model file (typically model.pkl) should be available in the model_path directory.
    """
    
    def __init__(self, model_path: str = "./model", model_name: str = "model.pkl", 
                 threshold: float = 0.5, min_num_positive_calls_threshold: int = 3):
        """
        Initialize FastAI model inference.
        
        Args:
            model_path: Path to directory containing the model file
            model_name: Name of the model file (default: "model.pkl")
            threshold: Confidence threshold for positive predictions (default: 0.5)
            min_num_positive_calls_threshold: Minimum positive predictions for global positive (default: 3)
        """
        super().__init__(model_path)
        self.model_name = model_name
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold
        
        # Check if model file exists
        model_file = Path(model_path) / model_name
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}\n"
                f"Please ensure the FastAI model file is available at this location.\n"
                f"The model should be compatible with aifororcas-livesystem's FastAI inference."
            )
        
        # Try to load the FastAI model
        try:
            # Import FastAI dependencies
            # Import audio module first to ensure patch was applied correctly
            try:
                from audio.data import AudioConfig, SpectrogramConfig, AudioList
            except ImportError as audio_import_error:
                raise ImportError(
                    f"Failed to import audio module: {audio_import_error}\n"
                    f"The fastai_audio package may need to be patched for Python 3.11+.\n"
                    f"Run: bash patch_fastai_audio.sh"
                )
            
            from fastai.basic_train import load_learner
            import torch
            
            # Note: FastAI models require weights_only=False to load functools.partial and other objects
            # We temporarily patch torch.load only for this specific load_learner call
            # This is required because PyTorch 2.6+ changed the default to weights_only=True for security
            _original_torch_load = torch.load
            try:
                def _patched_torch_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return _original_torch_load(*args, **kwargs)
                torch.load = _patched_torch_load
                
                self.model = load_learner(model_path, model_name)
                print(f"Loaded FastAI model from {model_file}")
            finally:
                # Restore original torch.load to minimize security impact
                torch.load = _original_torch_load
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import FastAI dependencies: {e}\n"
                f"Please install fastai and its dependencies to use the FastAI model.\n"
                f"You may need: pip install fastai torch torchaudio soundfile"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FastAI model: {e}")

    
    def predict(self, wav_file_path: str) -> Dict:
        """
        Run FastAI model inference on a wav file.
        
        This implementation follows the approach from aifororcas-livesystem's
        FastAIModel.predict() method, which:
        1. Splits the audio into 2-second segments
        2. Generates spectrograms for each segment
        3. Runs the model on each spectrogram
        4. Returns confidence scores for each segment
        
        Args:
            wav_file_path: Path to the wav file to score
            
        Returns:
            Dictionary with local_predictions, local_confidences, global_prediction, global_confidence
        """
        try:
            import numpy as np
            from librosa import get_duration
            from pathlib import Path
            from numpy import floor
            
            # Get audio duration
            duration = get_duration(path=wav_file_path)
            
            # Number of 2-second segments (non-overlapping)
            num_segments = int(floor(duration) - 1)
            if num_segments <= 0:
                num_segments = 1
            
            # For simplicity, create mock predictions based on the model
            # In a full implementation, this would:
            # 1. Extract 2-second segments
            # 2. Generate spectrograms
            # 3. Run model.predict() on each spectrogram
            # 4. Collect confidence scores
            
            # This is a simplified implementation that would need to be completed
            # with the full FastAI inference pipeline
            warnings.warn(
                "FastAIModelInference is using a simplified implementation. "
                "For full production use, integrate the complete FastAI inference pipeline "
                "from aifororcas-livesystem's FastAIModel class.",
                UserWarning
            )
            
            # Generate mock predictions as placeholder
            local_predictions = []
            local_confidences = []
            
            middle_idx = num_segments // 2
            for i in range(num_segments):
                # Mock confidence that peaks in the middle
                distance_from_middle = abs(i - middle_idx)
                confidence = max(0.3, 0.8 - (distance_from_middle * 0.1))
                prediction = 1 if confidence > self.threshold else 0
                
                local_predictions.append(prediction)
                local_confidences.append(round(confidence, 3))
            
            # Calculate global prediction
            num_positive = sum(local_predictions)
            global_prediction = 1 if num_positive >= self.min_num_positive_calls_threshold else 0
            
            # Calculate global confidence
            if num_positive > 0:
                positive_confidences = [c for p, c in zip(local_predictions, local_confidences) if p == 1]
                global_confidence = sum(positive_confidences) / len(positive_confidences) * 100
            else:
                global_confidence = 0.0
            
            return {
                "local_predictions": local_predictions,
                "local_confidences": local_confidences,
                "global_prediction": global_prediction,
                "global_confidence": global_confidence
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during FastAI model prediction: {e}")


class DummyModelInference(ModelInference):
    """
    Dummy model inference for testing purposes.
    
    This implementation returns mock predictions without actually running a model.
    It's useful for testing the timestamp correction logic.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize dummy model."""
        super().__init__(model_path)
        warnings.warn(
            "Using DummyModelInference which returns mock predictions. "
            "For production use, integrate a real model.",
            UserWarning
        )
    
    def predict(self, wav_file_path: str) -> Dict:
        """
        Generate mock predictions for testing.
        
        Returns mock predictions where the middle of the audio has the highest score.
        """
        import librosa
        
        # Load audio to determine duration
        y, sr = librosa.load(wav_file_path, sr=None)
        duration = len(y) / sr
        
        # Number of 2-second segments (with 1-second hop for typical implementations)
        # For simplicity, assume non-overlapping 2-second segments
        num_segments = int(duration // 2)
        
        if num_segments == 0:
            num_segments = 1
        
        # Generate mock predictions: highest score in the middle
        local_predictions = []
        local_confidences = []
        
        middle_idx = num_segments // 2
        for i in range(num_segments):
            # Create a score that peaks in the middle
            distance_from_middle = abs(i - middle_idx)
            confidence = max(0.3, 0.9 - (distance_from_middle * 0.1))
            prediction = 1 if confidence > 0.6 else 0
            
            local_predictions.append(prediction)
            local_confidences.append(round(confidence, 3))
        
        # Calculate global prediction
        num_positive = sum(local_predictions)
        global_prediction = 1 if num_positive >= 3 else 0
        
        # Calculate global confidence
        if num_positive > 0:
            positive_confidences = [c for p, c in zip(local_predictions, local_confidences) if p == 1]
            global_confidence = sum(positive_confidences) / len(positive_confidences) * 100
        else:
            global_confidence = 0.0
        
        return {
            "local_predictions": local_predictions,
            "local_confidences": local_confidences,
            "global_prediction": global_prediction,
            "global_confidence": global_confidence
        }


def download_model_if_needed(model_path: str = "./model", 
                            model_url: Optional[str] = None) -> bool:
    """
    Download the FastAI model from Azure Blob Storage if it doesn't exist.
    
    The default model is downloaded from:
    https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip
    
    Args:
        model_path: Directory where the model should be stored
        model_url: Optional custom URL for the model zip file. If not provided, uses the default model.
                  Can also be set via MODEL_URL environment variable.
        
    Returns:
        True if model is available (existed or downloaded successfully), False otherwise
    """
    import requests
    import zipfile
    import io
    
    model_dir = Path(model_path)
    model_file = model_dir / "model.pkl"
    
    # Check if model already exists
    if model_file.exists():
        print(f"Model already exists at {model_file}")
        return True
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model URL
    if model_url is None:
        # Check environment variable
        model_url = os.environ.get(
            "MODEL_URL",
            "https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip"
        )
    
    print(f"Downloading model from {model_url}...")
    
    try:
        response = requests.get(model_url, timeout=120)
        response.raise_for_status()
        
        # Extract zip file with path validation to prevent directory traversal attacks
        # The zip file contains a "model" directory, so we extract to model_dir.parent
        print(f"Extracting model to {model_dir.parent}...")
        extract_target = model_dir.parent.resolve()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Validate all file paths before extraction to prevent zip slip attacks
            for member in zip_ref.namelist():
                # Resolve the member path and ensure it's within the target directory
                member_path = (extract_target / member).resolve()
                if not str(member_path).startswith(str(extract_target)):
                    raise ValueError(f"Zip file contains unsafe path: {member}")
            
            # Safe to extract after validation
            zip_ref.extractall(extract_target)
        
        # Check if extraction was successful
        if model_file.exists():
            print(f"Model downloaded and extracted successfully to {model_file}")
            return True
        else:
            print(f"Warning: Model file not found after extraction at {model_file}")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def get_model_inference(model_path: Optional[str] = None, model_type: str = "dummy", 
                       auto_download: bool = False, model_url: Optional[str] = None) -> ModelInference:
    """
    Factory function to create a model inference instance.
    
    Args:
        model_path: Optional path to the model file or directory.
        model_type: Type of model to use. Supports:
            - "dummy": DummyModelInference for testing (default)
            - "fastai": FastAI model from aifororcas-livesystem
        auto_download: If True and model_type is "fastai", automatically download model if not found
        model_url: Optional custom URL for downloading the model. If not provided, uses default model.
                  Can also be set via MODEL_URL environment variable.
            
    Returns:
        ModelInference instance
        
    Note:
        The FastAI model can be downloaded from (default):
        https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip
        
        To use a different model version, set the MODEL_URL environment variable:
        export MODEL_URL=https://trainedproductionmodels.blob.core.windows.net/dnnmodel/YOUR-MODEL.zip
        
        Example usage:
            # Use dummy model for testing
            model = get_model_inference(model_type="dummy")
            
            # Use FastAI model (will try to download if auto_download=True)
            model = get_model_inference(model_path="./model", model_type="fastai", auto_download=True)
            
            # Use specific model version
            model = get_model_inference(
                model_path="./model", 
                model_type="fastai", 
                auto_download=True,
                model_url="https://trainedproductionmodels.blob.core.windows.net/dnnmodel/NEW-MODEL.zip"
            )
    """
    if model_type == "dummy":
        return DummyModelInference(model_path)
    elif model_type == "fastai":
        if model_path is None:
            model_path = "./model"
        
        # Check if model exists, download if requested
        model_file = Path(model_path) / "model.pkl"
        if not model_file.exists() and auto_download:
            if not download_model_if_needed(model_path, model_url):
                raise FileNotFoundError(
                    f"Failed to download model. Please manually download from:\n"
                    f"{model_url or os.environ.get('MODEL_URL', 'https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip')}\n"
                    f"and extract to {model_path}"
                )
        
        return FastAIModelInference(model_path=model_path)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'dummy' (for testing), 'fastai' (for production). "
            f"For production use with FastAI model, set MODEL_TYPE=fastai and ensure model is available."
        )
