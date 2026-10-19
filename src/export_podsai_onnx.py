#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Export PODS-AI HuggingFace model to ONNX for CPU inference.

Usage:
  python src/export_podsai_onnx.py --model davethaler/whale-call-detector --revision <revision> --out podsai.onnx

This script loads the HF model, builds a representative dummy input tensor
matching the AST spectrogram input shape, and exports the model to ONNX with a
dynamic batch axis and fixed spectrogram dimensions.
"""
import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export PODS-AI HF model to ONNX")
    parser.add_argument("--model", default="davethaler/whale-call-detector", help="HF model id or local path")
    parser.add_argument("--revision", default=None, help="HuggingFace model revision")
    parser.add_argument("--out", default="podsai.onnx", help="Output ONNX path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Export device")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=512, help="Representative number of frames (time axis) for export")
    parser.add_argument("--mel-bins", type=int, default=128, help="Representative mel bins for export")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    # Use PodsAIInference to load model + feature extractor and provide helpers
    # for computing input_values and resizing AST positional embeddings.
    from podsai_inference import PodsAIInference

    device = torch.device(args.device)
    inf = PodsAIInference(args.model, device=str(device), model_revision=args.revision)
    model = inf.model
    model.eval()
    model.to(device)
    fe = inf.feature_extractor

    # Attempt to compute a representative frames value from the feature-extractor
    # configuration so that patch/stride arithmetic used by AST-style models
    # is consistent during export. Fall back to the CLI --frames value when
    # attributes are missing or loading fails.
    try:
        fe = inf.feature_extractor
        # Defaults matching podsai_inference / AST expectations
        DEFAULT_AST_PATCH_SIZE = 16
        DEFAULT_AST_TIME_STRIDE = 10
        DEFAULT_AST_FREQUENCY_STRIDE = 10

        patch_size = int(getattr(fe, "patch_size", DEFAULT_AST_PATCH_SIZE))
        # hop_length/hop are often frame shift; prefer hop_length then time_stride
        time_stride = int(getattr(fe, "hop_length", getattr(fe, "time_stride", DEFAULT_AST_TIME_STRIDE)))
        mel_bins = int(getattr(fe, "num_mel_bins", args.mel_bins))

        # Try to infer target frames so that the number of patch tokens matches
        # the model's position embeddings length when possible.
        target_patches = getattr(fe, "max_length", None)

        # If the model exposes positional embeddings (AST), use them to compute
        # a compatible frames value.
        pos_len = None
        try:
            pos_len = getattr(model, "audio_spectrogram_transformer", None)
            if pos_len is not None:
                pos_len = int(model.audio_spectrogram_transformer.embeddings.position_embeddings.shape[1])
        except Exception:
            pos_len = None

        if pos_len:
            NUM_SPECIAL_TOKENS = 2
            freq_stride = int(getattr(fe, "frequency_stride", DEFAULT_AST_FREQUENCY_STRIDE))
            # number of frequency patches (ceil to cover mel bins)
            freq_patches = (mel_bins + freq_stride - 1) // freq_stride
            time_patches = max(1, (pos_len - NUM_SPECIAL_TOKENS) // freq_patches)
            frames = (time_patches - 1) * time_stride + patch_size
            print(f"Inferred frames from position embeddings: frames={frames} (time_patches={time_patches}, freq_patches={freq_patches}, pos_len={pos_len})")
        else:
            if target_patches is None:
                target_patches = max(1, args.frames // int(time_stride))
            frames = int(target_patches) * int(time_stride) + int(patch_size)
            print(f"Computed export frames={frames} (patch={patch_size}, stride={time_stride}, patches={target_patches}), mel_bins={mel_bins}")
    except Exception:
        frames = args.frames
        mel_bins = args.mel_bins
        print(f"Warning: could not read feature extractor config; falling back to frames={frames}, mel_bins={mel_bins}")

    # Build a dummy input: try to use PodsAIInference helper to compute the
    # exact spectrogram-window layout the HF model expects. This produces a
    # shape (num_positions, frames, mel_bins) which we treat as a batch of
    # windows for export. If that fails, fall back to a random tensor.
    try:
        import numpy as _np

        # Ensure AST positional embeddings match our target frames/mel_bins
        try:
            resized = inf._ensure_ast_position_embeddings(int(frames), int(mel_bins))
            print(f"_ensure_ast_position_embeddings returned: {resized}")
        except Exception as e:
            print(f"Warning: _ensure_ast_position_embeddings failed: {e}")

        # Default sliding-window settings matching PodsAIInference defaults.
        segment_duration = 3
        hop_duration = 2
        sr = 16000

        num_positions = max(1, args.batch_size)

        audio_duration_s = segment_duration + hop_duration * (max(1, num_positions) - 1)
        audio_samples = int(audio_duration_s * sr)
        audio = _np.zeros(audio_samples, dtype=_np.float32)

        input_values = inf._compute_input_values(audio, sr, num_positions, int(hop_duration * sr), int(segment_duration * sr))
        # input_values is a torch tensor in PodsAIInference implementation; ensure numpy->torch
        if hasattr(input_values, 'numpy'):
            input_np = input_values.detach().cpu().numpy()
        else:
            input_np = _np.array(input_values)

        dummy = torch.from_numpy(input_np).float().to(device)
        print(f"Prepared dummy from PodsAIInference helper with shape {tuple(dummy.shape)}")
    except Exception as e:
        print(f"Warning: failed to prepare dummy via PodsAIInference helper: {e}; falling back to random tensor")
        dummy = torch.randn(args.batch_size, frames, mel_bins, dtype=torch.float32, device=device)

    out_path = Path(args.out)
    print(f"Exporting to {out_path} (opset=13, dynamic batch axis; fixed spectrogram dimensions)")

    # The HF model expects keyword argument 'input_values' or similar; export by positional input
    # Export with a fixed time dimension (frames) to avoid symbolic-shape
    # inconsistencies during torch.export. We keep batch as a dynamic axis.
    # Wrap the HF model in a minimal nn.Module to provide an explicit I/O contract
    # (input_values -> logits). This keeps the exported graph small and avoids
    # exporting unrelated Python-level code paths.
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_values):
            # Forward the tensor through the HF model and return logits tensor.
            out = self.model(input_values=input_values)
            return out.logits

    wrapper = ExportWrapper(model).to(device)

    # Try the standard export path first; if it fails due to exportdb symbolic
    # shape or fake-tensor runtime errors, fall back to tracing with torch.jit.trace
    # which is often more forgiving for complex model internals.
    try:
        print("Attempting torch.onnx.export (symbolic export)...")
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            opset_version=13,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )
        print("ONNX export complete (symbolic export)")
    except Exception as exc:
        print(f"Symbolic export failed: {type(exc).__name__}: {exc}")
        print("Falling back to tracing (torch.jit.trace) and exporting the traced module...")
        try:
            traced = torch.jit.trace(wrapper, dummy)
            torch.onnx.export(
                traced,
                (dummy,),
                str(out_path),
                opset_version=13,
                input_names=["input_values"],
                output_names=["logits"],
                dynamic_axes={
                    "input_values": {0: "batch"},
                    "logits": {0: "batch"},
                },
                do_constant_folding=True,
            )
            print("ONNX export complete (traced export)")
        except Exception as exc2:
            print(f"Traced export also failed: {type(exc2).__name__}: {exc2}")
            raise
    print("ONNX export complete")


if __name__ == '__main__':
    main()
