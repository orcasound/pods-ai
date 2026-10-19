#!/usr/bin/env python3
"""Validate ONNX export against PyTorch model and benchmark runtimes.

Usage:
  python src/validate_podsai_onnx.py --onnx podsai.onnx --model davethaler/whale-call-detector --wav path/to/sample.wav

This will:
 - Load a sample WAV and compute input_values via PodsAIInference._compute_input_values
 - Run the HF model forward and the ONNX session forward and compare logits
 - Report timings for PyTorch and ONNX runs
"""
import argparse
import time
import numpy as np
import torch


def load_pytorch_model(model_id: str, device: str = "cpu"):
    from transformers import AutoModelForAudioClassification
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return model


def compute_input_values_from_wav(wav_path: str, model_path: str):
    # Reuse PodsAIInference helper by loading the class
    from podsai_inference import PodsAIInference
    inf = PodsAIInference(model_path, device='cpu')
    # Load wav and compute input_values for a single sample using default segment/hop
    import librosa
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    # compute number of positions for default segment/hop
    segment_samples = 3 * sr
    hop_samples = 2 * sr
    audio_duration = len(audio) / sr
    num_positions = int(np.floor((audio_duration - 3) / 2)) + 1 if audio_duration >= 3 else 1
    input_values = inf._compute_input_values(audio, sr, num_positions, hop_samples, segment_samples)
    # Input shape produced: (num_positions, frames, mel_bins)
    # For export/onnx we will use a single batch of all positions flattened as batch dimension
    return input_values


def run_onnx(onnx_path: str, input_np: np.ndarray, runs: int = 3):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # input_np shape should be (batch, frames, mel)
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = sess.run(None, {input_name: input_np.astype(np.float32)})
        t1 = time.perf_counter()
        times.append(t1 - t0)
        out = res[0]
    return out, np.mean(times), np.std(times)


def run_torch(model, input_tensor: torch.Tensor, runs: int = 3):
    times = []
    out = None
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            res = model(input_values=input_tensor)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            out = res.logits.detach().cpu().numpy()
    return out, np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--model", default="davethaler/whale-call-detector")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()

    print("Computing input values from WAV using PodsAIInference helper (may load HF config)...")
    input_values = compute_input_values_from_wav(args.wav, args.model)

    # Collapse positions into batch dimension
    inp = input_values.detach().cpu().numpy()
    batch_np = inp  # shape (num_positions, frames, mel)
    print(f"Prepared input batch shape for ONNX: {batch_np.shape}")

    print("Loading PyTorch model...")
    torch_model = load_pytorch_model(args.model, device='cpu')

    # Run a single forward on torch
    import torch
    input_tensor = torch.from_numpy(batch_np).float()
    print("Running PyTorch forward... (3 runs)")
    torch_out, torch_mean, torch_sd = run_torch(torch_model, input_tensor, runs=3)
    print(f"PyTorch logits shape: {torch_out.shape}, mean time: {torch_mean:.3f}s (sd {torch_sd:.3f})")

    print("Running ONNX forward... (3 runs)")
    onnx_out, onnx_mean, onnx_sd = run_onnx(args.onnx, batch_np, runs=3)
    print(f"ONNX logits shape: {onnx_out.shape}, mean time: {onnx_mean:.3f}s (sd {onnx_sd:.3f})")

    # Compare outputs (per-sample logits) using absolute difference
    import numpy as np
    # Align shapes: logits shape (batch, num_labels)
    if torch_out.shape != onnx_out.shape:
        print(f"Shape mismatch: torch {torch_out.shape} vs onnx {onnx_out.shape}")
    else:
        diff = np.max(np.abs(torch_out - onnx_out))
        print(f"Max abs difference between PyTorch and ONNX logits: {diff:.6f}")


if __name__ == '__main__':
    main()
