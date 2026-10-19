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
    parser.add_argument("--frames", type=int, default=300, help="Representative number of frames (time axis) for export")
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
    if not inf._use_spectrogram_input:
        raise ValueError(
            "ONNX export currently supports only spectrogram-based models; "
            "raw-audio models such as Wav2Vec2 are not supported."
        )

    # Build a dummy input with the dimensions requested on the command line.
    try:
        try:
            resized = inf._ensure_ast_position_embeddings(
                args.frames, args.mel_bins
            )
            print(f"_ensure_ast_position_embeddings returned: {resized}")
        except Exception as e:
            print(f"Warning: _ensure_ast_position_embeddings failed: {e}")

        dummy = torch.randn(
            args.batch_size, args.frames, args.mel_bins,
            dtype=torch.float32, device=device,
        )
        print(f"Prepared dummy with shape {tuple(dummy.shape)}")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare ONNX export input: {e}") from e

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
