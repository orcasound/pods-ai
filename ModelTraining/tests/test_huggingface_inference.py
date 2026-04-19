#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Unit tests for HuggingFaceInference timestamp correction semantics.

These tests verify that HuggingFaceInference.predict() produces output
with correct length and indexing semantics for use in timestamp correction.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path


class TestHuggingFaceInferenceIndexing:
    """Test indexing semantics for timestamp correction."""
    
    @pytest.fixture
    def synthetic_audio_60s(self):
        """Create a 60-second synthetic audio file with a tone at second 30."""
        sr = 16000
        duration = 60
        samples = sr * duration
        
        # Create silence with a 1-second tone at second 30
        audio = np.zeros(samples, dtype=np.float32)
        tone_start = 30 * sr
        tone_end = 31 * sr
        t = np.linspace(0, 1, sr)
        audio[tone_start:tone_end] = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    def test_output_length_matches_audio_duration(self, synthetic_audio_60s):
        """Test that local_confidences length matches audio duration in seconds."""
        # This test will FAIL with current code (returns 29 instead of 60)
        from huggingface_inference import HuggingFaceInference
        
        # Use a dummy model path (will fail to load, but we can test with mocking)
        # For now, this is a placeholder showing what should be tested
        pytest.skip("Requires mock model or test model for HuggingFace")
        
        model = HuggingFaceInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=2)
        
        # EXPECTED: 60 confidences for 60-second audio (one per second)
        assert len(result["local_confidences"]) == 60, \
            f"Expected 60 confidences for 60s audio, got {len(result['local_confidences'])}"
    
    def test_index_corresponds_to_seconds(self, synthetic_audio_60s):
        """Test that local_confidences[i] represents confidence at second i."""
        pytest.skip("Requires mock model or test model for HuggingFace")
        
        from huggingface_inference import HuggingFaceInference
        
        model = HuggingFaceInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=2)
        
        # EXPECTED: Peak confidence around index 30 (where the tone is)
        max_idx = result["local_confidences"].index(max(result["local_confidences"]))
        
        # Should be within ±2 seconds of actual tone position (second 30)
        assert 28 <= max_idx <= 32, \
            f"Peak confidence at index {max_idx}, expected near 30"
    
    def test_hop_duration_1_second(self, synthetic_audio_60s):
        """Test with hop_duration=1 to match FastAI behavior."""
        pytest.skip("Requires mock model or test model for HuggingFace")
        
        from huggingface_inference import HuggingFaceInference
        
        model = HuggingFaceInference("test-model-path")
        result = model.predict(synthetic_audio_60s, segment_duration=3, hop_duration=1)
        
        # With 1-second hop, should get one prediction per second
        assert len(result["local_confidences"]) == 60
    
    def test_short_audio_handling(self):
        """Test handling of audio shorter than segment_duration."""
        # Create 2-second audio (shorter than 3-second segment)
        sr = 16000
        audio = np.random.randn(2 * sr).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            audio_path = f.name
        
        try:
            pytest.skip("Requires mock model or test model for HuggingFace")
            
            from huggingface_inference import HuggingFaceInference
            
            model = HuggingFaceInference("test-model-path")
            result = model.predict(audio_path, segment_duration=3, hop_duration=2)
            
            # Should return at least 1 confidence value (padded segment)
            assert len(result["local_confidences"]) >= 1
        finally:
            Path(audio_path).unlink(missing_ok=True)
    
    def test_exact_multiple_of_hop_duration(self):
        """Test audio length that's exact multiple of hop_duration."""
        # 60 seconds is exact multiple of hop_duration=2
        # Should produce consistent results
        pytest.skip("Requires mock model or test model for HuggingFace")
        
        sr = 16000
        audio = np.random.randn(60 * sr).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            audio_path = f.name
        
        try:
            from huggingface_inference import HuggingFaceInference
            
            model = HuggingFaceInference("test-model-path")
            result = model.predict(audio_path, segment_duration=3, hop_duration=2)
            
            # Should handle exact multiples cleanly
            assert len(result["local_confidences"]) == 60
        finally:
            Path(audio_path).unlink(missing_ok=True)


class TestHuggingFaceInferenceVsFastAI:
    """Compare HuggingFace and FastAI output format compatibility."""
    
    def test_output_format_compatibility(self):
        """Test that both models return compatible output formats."""
        pytest.skip("Requires both models")
        
        # Both should return:
        # - local_predictions: list
        # - local_confidences: list (same length as local_predictions)
        # - global_prediction: int (FastAI) or int (HuggingFace)
        # - global_confidence: float (0.0-1.0)
        
        # HuggingFace additionally returns:
        # - global_prediction_label: str