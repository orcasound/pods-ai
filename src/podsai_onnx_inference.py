#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""ONNX Runtime-backed PODS-AI inference wrapper.

This provides a lightweight replacement for PodsAIInference that runs the
core model forward using an ONNX InferenceSession. It reuses the same
preprocessing (mel fbank windows) as PodsAIInference by loading the
feature-extractor configuration from a HuggingFace model id or local folder.

Usage:
  inf = get_podsai_onnx_inference(onnx_path='podsai.onnx', feature_extractor='davethaler/whale-call-detector')
  result = inf.predict('path/to/sample.wav')
"""
from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from model_inference import ModelInference


class PodsAIOnnxInference(ModelInference):
    def __init__(
        self,
        onnx_path: str,
        feature_extractor: Optional[str] = None,
        threshold: float = 0.5,
        min_num_positive_calls_threshold: int = 2,
        inference_batch_size: int = 8,
        ort_intra_threads: Optional[int] = None,
        ort_inter_threads: Optional[int] = None,
    ) -> None:
        super().__init__(onnx_path)
        try:
            import onnxruntime as ort
            from transformers import AutoFeatureExtractor, AutoConfig
        except Exception as e:
            raise RuntimeError(f"Required packages missing for ONNX inference: {e}")

        self.onnx_path = str(onnx_path)
        self.threshold = float(threshold)
        self.min_num_positive_calls_threshold = int(min_num_positive_calls_threshold)
        self.inference_batch_size = int(inference_batch_size)

        # Load feature extractor/config to obtain spectrogram params and labels.
        if not feature_extractor:
            # Allow override via environment variable for convenience.
            feature_extractor = os.environ.get("PODSAI_FEATURE_EXTRACTOR")
        if not feature_extractor:
            raise ValueError(
                "feature_extractor must be provided (HF model id or local path) via argument or PODSAI_FEATURE_EXTRACTOR env var when using ONNX model"
            )

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor)
        cfg = AutoConfig.from_pretrained(feature_extractor)
        # Build id2label/label2id mapping from config if present.
        raw_id2label = getattr(cfg, "id2label", None) or {}
        raw_label2id = getattr(cfg, "label2id", None) or {}
        # id2label may be either {"0": "water"} or {0: "water"}; normalize to int keys
        if raw_id2label:
            try:
                self.id2label = {int(k): v for k, v in raw_id2label.items()}
            except Exception:
                # Already int-keyed or unexpected format
                self.id2label = {k: v for k, v in raw_id2label.items()}
        else:
            self.id2label = {}

        # label2id may be either {"0": "water"} reversed or {"water": 0}; normalize to name->id mapping
        self.label2id = {}
        if raw_label2id:
            # If keys look like numeric strings mapping to labels, invert
            try:
                # Try to detect numeric-string keys
                if all(str(k).isdigit() for k in raw_label2id.keys()):
                    # raw_label2id: {"0": "water"} -> want {"water": 0}
                    for k, v in raw_label2id.items():
                        self.label2id[v] = int(k)
                else:
                    # raw_label2id: {"water": 0}
                    for k, v in raw_label2id.items():
                        self.label2id[k] = int(v)
            except Exception:
                # Fallback: leave empty
                self.label2id = {}

        self.negative_label_names = {
            "abiotic", "bird", "human", "jingle", "other", "vessel", "water"
        }
        self.negative_class_ids = {
            self.label2id[name]
            for name in self.negative_label_names
            if name in self.label2id
        }

        # Create ONNX Runtime session with CPU provider.
        sess_opts = ort.SessionOptions()
        if ort_intra_threads:
            sess_opts.intra_op_num_threads = int(ort_intra_threads)
        if ort_inter_threads:
            sess_opts.inter_op_num_threads = int(ort_inter_threads)

        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=sess_opts, providers=providers)
        # Input name expected by exported model.
        self._input_name = self.sess.get_inputs()[0].name

    def _compute_input_values(self, audio: np.ndarray, sr: int, num_positions: int, hop_samples: int, segment_samples: int):
        """Compute mel fbank windows replicating PodsAIInference behavior."""
        import torch
        import torchaudio

        num_mel_bins: int = getattr(self.feature_extractor, 'num_mel_bins', 128)
        max_length: int = getattr(self.feature_extractor, 'max_length', 1024)
        fe_mean: float = getattr(self.feature_extractor, 'mean', -4.2677393)
        fe_std: float = getattr(self.feature_extractor, 'std', 4.5689974)
        do_normalize: bool = getattr(self.feature_extractor, 'do_normalize', True)
        frame_shift_ms: float = getattr(self.feature_extractor, 'hop_length', 10)

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        full_fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_shift=frame_shift_ms,
        )

        frames_per_second = 1000.0 / frame_shift_ms
        hop_frames = round(hop_samples / sr * frames_per_second)
        seg_frames = round(segment_samples / sr * frames_per_second)

        target_frames = max(1, min(max_length, seg_frames))

        windows = []
        for pos_idx in range(num_positions):
            start_frame = pos_idx * hop_frames
            end_frame = start_frame + seg_frames
            window = full_fbank[start_frame:end_frame, :]
            window = window - window.mean()
            if window.shape[0] < target_frames:
                pad_len = target_frames - window.shape[0]
                window = torch.nn.functional.pad(window, (0, 0, 0, pad_len))
            else:
                window = window[:target_frames, :]
            windows.append(window)

        input_values = torch.stack(windows, dim=0)
        if do_normalize:
            input_values = (input_values - fe_mean) / (fe_std * 2.0)

        return input_values.detach().cpu().numpy().astype(np.float32)

    def predict(self, wav_path: str, segment_duration: int = 3, hop_duration: int = 2, threshold: Optional[float] = None, min_num_positive_calls_threshold: Optional[int] = None) -> dict:
        threshold = threshold if threshold is not None else self.threshold
        min_num_positive_calls_threshold = min_num_positive_calls_threshold if min_num_positive_calls_threshold is not None else self.min_num_positive_calls_threshold

        # Load audio
        import librosa
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        audio_duration = len(audio) / sr
        segment_samples = int(segment_duration * sr)
        hop_samples = int(hop_duration * sr)
        num_positions = int(math.floor((audio_duration - segment_duration) / hop_duration)) + 1 if audio_duration >= segment_duration else 1

        if num_positions <= 0:
            num_positions = 1

        # Build model input batch
        batch_np = self._compute_input_values(audio, sr, num_positions, hop_samples, segment_samples)
        # batch_np shape: (num_positions, frames, mel_bins)

        # Run ORT in batches
        all_logits = []
        t0_total = time.perf_counter()
        for start in range(0, batch_np.shape[0], self.inference_batch_size):
            end = min(start + self.inference_batch_size, batch_np.shape[0])
            batch = batch_np[start:end]
            res = self.sess.run(None, {self._input_name: batch})
            logits = res[0]
            all_logits.append(logits)
        t1_total = time.perf_counter()
        logits_np = np.vstack(all_logits) if all_logits else np.empty((0, 0), dtype=np.float32)

        # Convert logits to probabilities (softmax)
        def softmax(x):
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)

        if logits_np.size == 0:
            return {
                "local_predictions": [],
                "local_confidences": [],
                "local_probs": [],
                "global_prediction": None,
                "global_prediction_label": "",
                "global_prediction_labels": [],
                "global_confidence": 0.0,
                "per_class_probabilities": {},
                "hop_duration": float(hop_duration),
                "segment_duration": float(segment_duration),
            }

        probs = softmax(logits_np)
        segment_probs = [row.tolist() for row in probs]
        local_pred_ids = [int(np.argmax(row)) for row in probs]
        local_confidences = [float(np.max(row)) for row in probs]

        # Per-class average probability
        num_labels = probs.shape[1]
        per_class_probs = {self.id2label.get(i, str(i)): float(np.mean(probs[:, i])) for i in range(num_labels)}

        # Global prediction: highest mean probability across classes
        class_means = {i: float(np.mean(probs[:, i])) for i in range(num_labels)}
        global_id = max(class_means.keys(), key=lambda k: class_means[k])
        global_label = self.id2label.get(global_id, str(global_id))
        global_confidence = class_means.get(global_id, 0.0)

        return {
            "local_predictions": local_pred_ids,
            "local_confidences": local_confidences,
            "local_probs": segment_probs,
            "global_prediction": int(global_id),
            "global_prediction_label": global_label,
            "global_prediction_labels": [self.id2label.get(i, str(i)) for i in sorted(class_means.keys(), key=lambda k: -class_means[k])],
            "global_confidence": float(global_confidence),
            "per_class_probabilities": per_class_probs,
            "hop_duration": float(hop_duration),
            "segment_duration": float(segment_duration),
            "predict_time": (t1_total - t0_total),
        }


def get_podsai_onnx_inference(onnx_path: str, feature_extractor: Optional[str] = None, **kwargs) -> PodsAIOnnxInference:
    return PodsAIOnnxInference(onnx_path, feature_extractor=feature_extractor, **kwargs)
