#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
OrcaHello SRKW Detector inference wrapper.

This module provides an inference wrapper for the OrcaHello SRKW detector model
(orcasound/orcahello-srkw-detector-v1) hosted on HuggingFace Hub.

The model code (OrcaHelloSRKWDetectorV1, AudioPreprocessor, etc.) is adapted from
https://github.com/orcasound/orcahello/tree/main/InferenceSystem/src/model
which is part of the OrcaHello project by the Orcasound community.

Usage:
    from orcahello_inference import OrcaHelloSRKWInference

    model = OrcaHelloSRKWInference("orcasound/orcahello-srkw-detector-v1")
    result = model.predict("sample.wav")
    print(result["global_prediction_label"])  # "whale" or "other"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library imports
# ──────────────────────────────────────────────────────────────────────────────
import dataclasses
import math
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

# ──────────────────────────────────────────────────────────────────────────────
# Third-party imports
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa import get_duration
from pydub import AudioSegment
from scipy.signal import resample_poly
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.models import resnet50

from huggingface_hub import PyTorchModelHubMixin

# Import base class.
from model_inference import ModelInference


# ──────────────────────────────────────────────────────────────────────────────
# Data classes (adapted from orcahello/InferenceSystem/src/model/types.py)
# ──────────────────────────────────────────────────────────────────────────────

def _from_dict(cls, d: Dict):
    """Create dataclass from dict, ignoring unknown keys."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in field_names})


@dataclass
class AudioConfig:
    """Audio loading and preprocessing configuration."""
    downmix_mono: bool = True
    resample_rate: int = 20000
    normalize: bool = False


@dataclass
class SpectrogramConfig:
    """Mel spectrogram computation configuration."""
    sample_rate: int = 16000
    n_fft: int = 2560
    hop_length: int = 256
    mel_n_filters: int = 256
    mel_f_min: float = 0.0
    mel_f_max: float = 10000.0
    mel_f_pad: int = 0
    convert_to_db: bool = True
    top_db: int = 100


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "orcahello-srkw-detector-v1"
    input_pad_s: float = 4.0
    num_classes: int = 2
    call_class_index: int = 1
    device: str = "auto"
    precision: str = "auto"


@dataclass
class InferenceConfig:
    """Inference sliding-window configuration."""
    window_s: float = 2.0
    window_hop_s: float = 1.0
    max_batch_size: int = 8
    strict_segments: bool = True


@dataclass
class GlobalPredictionConfig:
    """Configuration for aggregating segment predictions into a global prediction.

    Attributes:
        aggregation_strategy: "mean_thresholded" or "mean_top_k".
        mean_top_k: Number of top segments to average for global_confidence (mean_top_k strategy).
        pred_local_threshold: Threshold for local binary predictions and segment selection.
        pred_global_threshold: Applied to global_confidence for binary global_prediction.
    """
    aggregation_strategy: str = "mean_top_k"
    mean_top_k: int = 2
    pred_local_threshold: float = 0.5
    pred_global_threshold: float = 0.6


@dataclass
class DetectorInferenceConfig:
    """Full configuration for the SRKW detector inference pipeline."""
    audio: AudioConfig = None
    spectrogram: SpectrogramConfig = None
    model: ModelConfig = None
    inference: InferenceConfig = None
    global_prediction: GlobalPredictionConfig = None

    def __post_init__(self):
        self.audio = self.audio or AudioConfig()
        self.spectrogram = self.spectrogram or SpectrogramConfig()
        self.model = self.model or ModelConfig()
        self.inference = self.inference or InferenceConfig()
        self.global_prediction = self.global_prediction or GlobalPredictionConfig()

    @classmethod
    def from_dict(cls, d: Dict) -> "DetectorInferenceConfig":
        """Create config from nested dict."""
        return cls(
            audio=_from_dict(AudioConfig, d.get("audio", {})),
            spectrogram=_from_dict(SpectrogramConfig, d.get("spectrogram", {})),
            model=_from_dict(ModelConfig, d.get("model", {})),
            inference=_from_dict(InferenceConfig, d.get("inference", {})),
            global_prediction=_from_dict(GlobalPredictionConfig, d.get("global_prediction", {})),
        )

    def as_dict(self) -> Dict:
        """Convert to nested dict."""
        return asdict(self)


@dataclass
class SegmentPrediction:
    """Prediction for a single audio segment."""
    start_time_s: float
    duration_s: float
    confidence: float


@dataclass
class DetectionMetadata:
    """Metadata for a detection result."""
    wav_file_path: str
    file_duration_s: float
    processing_time_s: float

    @property
    def realtime_factor(self) -> float:
        """Ratio of file duration to processing time (>1 means faster than realtime)."""
        if self.processing_time_s > 0:
            return self.file_duration_s / self.processing_time_s
        return float('inf')


@dataclass
class DetectionResult:
    """Detection result from the SRKW detector."""
    local_predictions: List[int]
    local_confidences: List[float]
    segment_predictions: List[SegmentPrediction]
    global_prediction: int
    global_confidence: float
    metadata: DetectionMetadata


# ──────────────────────────────────────────────────────────────────────────────
# Audio frontend (adapted from orcahello/InferenceSystem/src/model/audio_frontend.py)
# ──────────────────────────────────────────────────────────────────────────────

def _downmix_to_mono(waveform: Tensor) -> Tensor:
    """Downmix multi-channel audio to mono.

    Args:
        waveform: Audio tensor of shape (channels, samples).

    Returns:
        Mono audio tensor of shape (1, samples).
    """
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def _resample_audio(waveform: Tensor, orig_sr: int, target_sr: int) -> Tensor:
    """Resample audio to target sample rate using scipy's polyphase resampling.

    Args:
        waveform: Audio tensor of shape (channels, samples).
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio tensor.
    """
    if orig_sr == target_sr:
        return waveform
    sr_gcd = math.gcd(orig_sr, target_sr)
    sig_np = waveform.numpy()
    resampled = resample_poly(sig_np, int(target_sr / sr_gcd), int(orig_sr / sr_gcd), axis=-1)
    return torch.from_numpy(resampled.astype(np.float32))


def _compute_mel_spectrogram(waveform: Tensor, spectrogram_config: Dict) -> Tensor:
    """Compute mel spectrogram from waveform.

    NOTE: Uses sample_rate=16000 for the MelSpectrogram transform even though
    audio is resampled to 20kHz before segmenting. This is an intentional quirk
    of how the model was trained: fastai_audio passes resample_rate=16000 to
    torchaudio.MelSpectrogram regardless of the actual audio sample rate.
    This must be preserved for exact parity with the trained model weights.

    Args:
        waveform: Audio tensor of shape (channels, samples).
        spectrogram_config: Dict with mel spectrogram parameters.

    Returns:
        Mel spectrogram tensor in dB scale.
    """
    mel_transform = MelSpectrogram(
        sample_rate=spectrogram_config["sample_rate"],
        n_fft=spectrogram_config["n_fft"],
        hop_length=spectrogram_config["hop_length"],
        n_mels=spectrogram_config["mel_n_filters"],
        f_min=spectrogram_config["mel_f_min"],
        f_max=spectrogram_config["mel_f_max"],
        pad=spectrogram_config["mel_f_pad"],
    )
    mel_spec = mel_transform(waveform)
    if spectrogram_config.get("convert_to_db", True):
        amplitude_to_db = AmplitudeToDB(top_db=spectrogram_config["top_db"])
        mel_spec = amplitude_to_db(mel_spec)
    return mel_spec.detach()


def _load_processed_waveform(file_path: str, audio_config: Dict) -> Tuple[Tensor, int]:
    """Load audio file and apply waveform-level preprocessing (downmix, resample).

    Args:
        file_path: Path to audio file.
        audio_config: Dict with audio preprocessing parameters.

    Returns:
        Tuple of (waveform tensor [1, samples], sample_rate).
    """
    data, orig_sr = sf.read(str(file_path), dtype="float32")
    if data.ndim == 1:
        waveform = torch.from_numpy(data.reshape(1, -1))
    else:
        waveform = torch.from_numpy(data.T)
        waveform = _downmix_to_mono(waveform)
    target_sr = audio_config["resample_rate"]
    waveform = _resample_audio(waveform, orig_sr, target_sr)
    if audio_config.get("normalize", False):
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    return waveform, target_sr


def _prepare_waveform(waveform: Tensor, sample_rate: int, config: Dict) -> Tensor:
    """Featurize and standardize a pre-loaded waveform into a mel spectrogram.

    Args:
        waveform: Pre-processed audio tensor of shape (1, samples).
        sample_rate: Sample rate of the waveform.
        config: Full config dict with audio/spectrogram/model sections.

    Returns:
        Mel spectrogram tensor ready for model inference, shape (1, n_mels, target_frames).
    """
    audio_config = config["audio"]
    spectrogram_config = config["spectrogram"]
    model_config = config["model"]

    # Compute mel spectrogram.
    features = _compute_mel_spectrogram(waveform, spectrogram_config)

    # Pad or crop to target frame count.
    input_pad_s = model_config["input_pad_s"]
    hop_length = spectrogram_config["hop_length"]
    resample_rate = model_config.get("resample_rate", audio_config["resample_rate"])
    target_frames = int(input_pad_s * resample_rate / hop_length)
    current_frames = features.shape[-1]

    if current_frames > target_frames:
        features = features[..., :target_frames]
    elif current_frames < target_frames:
        padding = target_frames - current_frames
        features = torch.nn.functional.pad(features, (0, padding), mode="constant", value=0)

    return features


@contextmanager
def _temp_segment_dir(output_dir: Optional[str] = None):
    """Context manager for temporary segment directory."""
    if output_dir is not None:
        yield output_dir
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir


def _audio_segment_generator(
    audio_file_path: str,
    segment_duration_s: float,
    segment_hop_s: float,
    output_dir: Optional[str] = None,
    max_segments: Optional[int] = None,
    start_time_s: float = 0.0,
    strict_segments: bool = True,
    process_waveform_config: Optional[Dict] = None,
) -> Generator[Tuple[str, float, float], None, None]:
    """Generate overlapping audio segments from an audio file.

    Yields segment file paths as they are created. Uses a temporary directory
    unless output_dir is explicitly provided.

    Args:
        audio_file_path: Path to input audio file.
        segment_duration_s: Duration of each segment in seconds.
        segment_hop_s: Hop/stride between segment starts in seconds.
        output_dir: Optional directory to save segments.
        max_segments: Optional maximum number of segments.
        start_time_s: Start time offset in seconds.
        strict_segments: If True, only generate complete segments.
        process_waveform_config: Optional dict with downmix_mono and resample_rate keys.
                                 When provided, audio is pre-processed before segmenting.

    Yields:
        Tuple of (segment_path, start_s, end_s).
    """
    with _temp_segment_dir(output_dir) as segment_dir:
        wav_name = Path(audio_file_path).stem

        if process_waveform_config is not None:
            # Pre-process the full audio once, then segment.
            waveform, target_sr = _load_processed_waveform(audio_file_path, process_waveform_config)
            processed_wav_path = f"{segment_dir}/_processed_{wav_name}.wav"
            sf.write(processed_wav_path, waveform.squeeze(0).numpy(), target_sr)
            source_path = processed_wav_path
        else:
            source_path = audio_file_path

        audio_duration = get_duration(path=source_path)
        audio = AudioSegment.from_file(source_path)

        effective_duration = audio_duration - start_time_s
        if strict_segments:
            num_segments = int(np.floor(effective_duration / segment_hop_s))
        else:
            num_segments = int(np.ceil(effective_duration / segment_hop_s))
        if max_segments is not None:
            num_segments = min(max_segments, num_segments)

        for i in range(num_segments):
            start_s = i * segment_hop_s + start_time_s
            end_s = start_s + segment_duration_s

            if strict_segments and end_s > audio_duration:
                break

            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            segment_path = f"{segment_dir}/{wav_name}_{start_ms:06d}_{end_ms:06d}.wav"
            segment = audio[start_ms:end_ms]
            segment.export(segment_path, format="wav")

            yield segment_path, start_s, end_s


class AudioPreprocessor:
    """Config-driven wrapper around segment generation and mel spectrogram computation.

    Args:
        config: DetectorInferenceConfig or dict with audio/spectrogram/model/inference sections.
    """

    def __init__(self, config: Union[DetectorInferenceConfig, Dict]) -> None:
        """Initialize AudioPreprocessor.

        Args:
            config: DetectorInferenceConfig or dict with model configuration.
        """
        if isinstance(config, DetectorInferenceConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = DetectorInferenceConfig.from_dict(config)
        else:
            raise TypeError(
                f"config must be DetectorInferenceConfig or dict, got {type(config)}"
            )

    def process_segments(
        self,
        audio_file_path: str,
    ) -> Generator[Tuple[Tensor, float, float], None, None]:
        """Generate preprocessed mel spectrogram segments from an audio file.

        Args:
            audio_file_path: Path to input audio file.

        Yields:
            Tuple of (mel_spectrogram, start_time_s, duration_s):
                - mel_spectrogram: tensor of shape (1, n_mels, target_frames)
                - start_time_s: start time of this segment in seconds
                - duration_s: duration of this segment in seconds
        """
        inf = self.config.inference
        config_dict = self.config.as_dict()
        waveform_config = dataclasses.asdict(self.config.audio)

        for segment_path, start_s, end_s in _audio_segment_generator(
            audio_file_path,
            segment_duration_s=inf.window_s,
            segment_hop_s=inf.window_hop_s,
            strict_segments=inf.strict_segments,
            process_waveform_config=waveform_config,
        ):
            data, sample_rate = sf.read(segment_path, dtype="float32")
            waveform = torch.from_numpy(data.reshape(1, -1))
            mel_spec = _prepare_waveform(waveform, sample_rate, config_dict)
            yield mel_spec, start_s, inf.window_s


# ──────────────────────────────────────────────────────────────────────────────
# Global prediction aggregation
# (adapted from orcahello/InferenceSystem/src/model/inference.py)
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_predictions(
    segment_preds: List[SegmentPrediction],
    config: GlobalPredictionConfig,
) -> Tuple[List[int], int, float]:
    """Aggregate segment predictions into local predictions and global prediction/confidence.

    Args:
        segment_preds: List of SegmentPrediction objects with confidence scores.
        config: GlobalPredictionConfig with aggregation settings.

    Returns:
        Tuple of (local_predictions, global_prediction, global_confidence).
    """
    if len(segment_preds) == 0:
        return [], 0, 0.0

    local_predictions = [
        1 if seg.confidence > config.pred_local_threshold else 0
        for seg in segment_preds
    ]

    if config.aggregation_strategy == "mean_thresholded":
        num_positive = sum(local_predictions)
        if num_positive > 0:
            positive_confs = [
                seg.confidence for seg, pred in zip(segment_preds, local_predictions)
                if pred == 1
            ]
            global_confidence = sum(positive_confs) / len(positive_confs)
        else:
            global_confidence = 0.0

    elif config.aggregation_strategy == "mean_top_k":
        sorted_segs = sorted(segment_preds, key=lambda s: s.confidence, reverse=True)
        k = max(1, min(config.mean_top_k, len(sorted_segs)))
        top_segs = sorted_segs[:k]
        global_confidence = sum(s.confidence for s in top_segs) / len(top_segs)

    else:
        raise ValueError(f"Unknown aggregation_strategy: {config.aggregation_strategy}")

    global_prediction = 1 if global_confidence >= config.pred_global_threshold else 0
    return local_predictions, global_prediction, global_confidence


# ──────────────────────────────────────────────────────────────────────────────
# OrcaHello SRKW Detector model class
# (adapted from orcahello/InferenceSystem/src/model/inference.py)
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def _resolve_dtype(precision_str: str, device: torch.device) -> torch.dtype:
    """Resolve precision string to torch.dtype."""
    if precision_str == "auto":
        return torch.float16 if device.type in ("cuda", "mps") else torch.float32
    if precision_str == "float16":
        return torch.float16
    return torch.float32


class _AdaptiveConcatPool2d(nn.Module):
    """Adaptive pooling that concatenates max and average pooling (matches fastai).

    Outputs 2x the input channels.
    """

    def __init__(self, output_size: int = 1) -> None:
        """Initialize adaptive concat pool layer."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass concatenating max and avg pools."""
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)


class OrcaHelloSRKWDetectorV1(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="orcahello",
    repo_url="https://github.com/orcasound/aifororcas-livesystem",
    tags=["audio-classification", "bioacoustics", "orca-detection", "srkw"],
    license="other",
):
    """ResNet50-based binary detector for individual SRKW orca calls from mel spectrograms.

    Architecture matches the production fastai model:
    - ResNet50 backbone with single-channel input (grayscale spectrogram)
    - Custom head: AdaptiveConcatPool2d -> Linear(4096, 512) -> Linear(512, 2)

    Adapted from orcahello/InferenceSystem/src/model/inference.py.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the SRKW detector.

        Args:
            config: Model configuration dict (validated via DetectorInferenceConfig).
        """
        super().__init__()
        self.config = DetectorInferenceConfig.from_dict(config)
        self.num_classes = self.config.model.num_classes
        self.call_class_index = self.config.model.call_class_index
        self._device = _resolve_device(self.config.model.device)
        self._dtype = _resolve_dtype(self.config.model.precision, self._device)
        self.model = resnet50()

        # Modify first conv for single-channel input.
        self.model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace pooling/fc with fastai-style head.
        self.model.avgpool = _AdaptiveConcatPool2d(1)
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.25),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )
        self.to(device=self._device, dtype=self._dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        return self.model(x)

    def predict_call(self, x: Tensor) -> Tensor:
        """Get call-class probabilities using softmax.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Probability tensor of shape (batch,) for the call class.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)[:, self.call_class_index]

    def detect_srkw_from_file(
        self,
        wav_file_path: str,
        config: Optional[Union[Dict, DetectorInferenceConfig]] = None,
    ) -> DetectionResult:
        """Detect SRKW calls from a WAV file.

        Segments the audio, computes mel spectrograms, runs inference, and
        aggregates results into local and global predictions.

        Args:
            wav_file_path: Path to the WAV file.
            config: Optional configuration overrides. If None, uses self.config.

        Returns:
            DetectionResult with local predictions, confidences, and global prediction.
        """
        if config is None:
            overrides = self.config
        elif isinstance(config, DetectorInferenceConfig):
            overrides = config
        else:
            overrides = DetectorInferenceConfig.from_dict(config)

        inf = overrides.inference
        max_batch_size = inf.max_batch_size

        start_time = time.perf_counter()
        preprocessor = AudioPreprocessor(overrides)
        spectrograms = []
        segment_info = []

        for mel_spec, start_s, duration_s in preprocessor.process_segments(wav_file_path):
            spectrograms.append(mel_spec)
            segment_info.append({"start_time_s": start_s, "duration_s": duration_s})

        if len(spectrograms) == 0:
            processing_time = time.perf_counter() - start_time
            return DetectionResult(
                local_predictions=[],
                local_confidences=[],
                global_prediction=0,
                global_confidence=0.0,
                segment_predictions=[],
                metadata=DetectionMetadata(
                    wav_file_path=wav_file_path,
                    file_duration_s=0.0,
                    processing_time_s=processing_time,
                ),
            )

        all_confidences = []
        for batch_start in range(0, len(spectrograms), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(spectrograms))
            batch = torch.stack(spectrograms[batch_start:batch_end])
            batch = batch.to(device=self._device, dtype=self._dtype)
            batch_confidences = self.predict_call(batch)
            all_confidences.append(batch_confidences.cpu().float())

        confidences = torch.cat(all_confidences)

        segment_preds = [
            SegmentPrediction(
                start_time_s=seg_info["start_time_s"],
                duration_s=seg_info["duration_s"],
                confidence=conf.item(),
            )
            for seg_info, conf in zip(segment_info, confidences)
        ]

        local_predictions, global_prediction, global_confidence = _aggregate_predictions(
            segment_preds, overrides.global_prediction
        )
        local_confidences = [seg.confidence for seg in segment_preds]

        last_seg = segment_info[-1]
        file_duration_s = last_seg["start_time_s"] + last_seg["duration_s"]
        processing_time_s = time.perf_counter() - start_time

        return DetectionResult(
            local_predictions=local_predictions,
            local_confidences=local_confidences,
            global_prediction=global_prediction,
            global_confidence=global_confidence,
            segment_predictions=segment_preds,
            metadata=DetectionMetadata(
                wav_file_path=wav_file_path,
                file_duration_s=file_duration_s,
                processing_time_s=processing_time_s,
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# ModelInference wrapper
# ──────────────────────────────────────────────────────────────────────────────

class OrcaHelloSRKWInference(ModelInference):
    """Inference wrapper for the OrcaHello SRKW detector model.

    Loads OrcaHelloSRKWDetectorV1 from HuggingFace Hub (default:
    orcasound/orcahello-srkw-detector-v1) and exposes a predict() interface
    compatible with the PODS-AI ModelInference base class.

    The model is a binary classifier (0 = no SRKW, 1 = SRKW detected).
    Unlike the HuggingFaceInference wrapper which uses Wav2Vec2, this model
    uses a ResNet50 backbone with mel spectrograms, matching the architecture
    from orcahello's production inference system.
    """

    DEFAULT_MODEL_PATH = "orcasound/orcahello-srkw-detector-v1"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize the OrcaHello SRKW inference wrapper.

        Args:
            model_path: HuggingFace Hub model ID or local path to model directory.
                        Defaults to "orcasound/orcahello-srkw-detector-v1".
            config: Optional configuration overrides as a nested dict.
                    If None, uses the default DetectorInferenceConfig values.
        """
        super().__init__(model_path=model_path)

        # Build inference configuration.
        self.inference_config = DetectorInferenceConfig.from_dict(config or {})

        print(f"Loading OrcaHello SRKW detector from {model_path}...")

        # Load the model from HuggingFace Hub or local path.
        # from_pretrained() downloads weights and creates OrcaHelloSRKWDetectorV1.
        try:
            self.model = OrcaHelloSRKWDetectorV1.from_pretrained(
                model_path,
                config=self.inference_config.as_dict(),
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Error loading OrcaHello model from {model_path}: {type(e).__name__}: {e}"
            ) from e

        print(
            f"OrcaHello model loaded. Device: {self.model._device}, "
            f"Dtype: {self.model._dtype}"
        )

    def predict(self, wav_file_path: str) -> Dict:
        """Run inference on a wav file.

        Uses the OrcaHello audio preprocessing pipeline (mel spectrograms via
        torchaudio, no fastai_audio dependency) and the ResNet50-based detector.

        Args:
            wav_file_path: Path to the wav file (typically 60 seconds long).

        Returns:
            Dictionary with keys:
                - local_predictions: List of binary predictions (0/1) per segment.
                - local_confidences: List of confidence scores (0.0-1.0) per segment.
                - global_prediction: Overall binary prediction (0 or 1).
                - global_prediction_label: "whale" if global_prediction==1 else "other".
                - global_confidence: Aggregated confidence score (0.0-1.0).
                - hop_duration: Hop duration in seconds between segments.
                - segment_duration: Duration of each segment in seconds.
            Returns dict with empty lists on error.
        """
        try:
            result = self.model.detect_srkw_from_file(
                wav_file_path, self.inference_config
            )
        except Exception as e:
            error_msg = f"Error running inference on {wav_file_path}: {type(e).__name__}: {e}"
            print(error_msg)
            return {
                "local_predictions": [],
                "local_confidences": [],
                "global_prediction": 0,
                "global_prediction_label": "other",
                "global_confidence": 0.0,
                "hop_duration": float(self.inference_config.inference.window_hop_s),
                "segment_duration": float(self.inference_config.inference.window_s),
            }

        global_prediction_label = "whale" if result.global_prediction == 1 else "other"

        return {
            "local_predictions": result.local_predictions,
            "local_confidences": result.local_confidences,
            "global_prediction": result.global_prediction,
            "global_prediction_label": global_prediction_label,
            "global_confidence": result.global_confidence,
            "hop_duration": float(self.inference_config.inference.window_hop_s),
            "segment_duration": float(self.inference_config.inference.window_s),
        }


def get_orcahello_srkw_inference(
    model_path: str = OrcaHelloSRKWInference.DEFAULT_MODEL_PATH,
    **kwargs,
) -> OrcaHelloSRKWInference:
    """Factory function to create an OrcaHelloSRKWInference instance.

    Args:
        model_path: HuggingFace Hub model ID or local path.
        **kwargs: Additional arguments passed to OrcaHelloSRKWInference.

    Returns:
        OrcaHelloSRKWInference instance.
    """
    return OrcaHelloSRKWInference(model_path, **kwargs)
