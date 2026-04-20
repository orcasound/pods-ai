#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Run inference on a wav file and output per-class probabilities.

Usage:
    python run_inference.py sample.wav
    python run_inference.py sample.wav --model huggingface --model-path /path/to/hf-model
    python run_inference.py sample.wav --model fastai --model-path ../model
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from model_inference import get_model_inference


def run_inference(wav_path: str, model_type: str = "huggingface",
                  model_path: Optional[str] = None) -> dict:
    """
    Run inference on a wav file and return per-class probabilities.

    Args:
        wav_path: Path to the wav file.
        model_type: Type of model to use ('huggingface' or 'fastai').
        model_path: Path to the model directory or HuggingFace Hub model ID.
                    Required for huggingface. Defaults to './model' for fastai.

    Returns:
        Dictionary with:
            - probabilities: dict mapping class label to probability (0.0-1.0)
            - global_prediction_label: predicted class label for the whole file
            - global_confidence: confidence score (0.0-1.0) for the global prediction
    """

    if model_type == "fastai":
        if model_path is None:
            model_path = "./model"
        model = get_model_inference(model_type="fastai", model_path=model_path)
        result = model.predict(wav_path)

        # FastAI is binary: local_confidences are per-second whale-call probabilities.
        local_confidences = result.get("local_confidences", [])
        if local_confidences:
            whale_prob = float(sum(local_confidences) / len(local_confidences))
        else:
            whale_prob = 0.0
        other_prob = 1.0 - whale_prob

        probabilities: dict[str, float] = {
            "other": round(other_prob, 4),
            "whale": round(whale_prob, 4),
        }
        global_prediction = result.get("global_prediction", 0)
        global_prediction_label = "whale" if global_prediction else "other"
        global_confidence = float(result.get("global_confidence", 0.0))

    elif model_type == "huggingface":
        if model_path is None:
            raise ValueError(
                "model_path is required for --model huggingface. "
                "Provide a path to a fine-tuned model directory or a HuggingFace Hub model ID."
            )
        model = get_model_inference(model_type="huggingface", model_path=model_path)
        result = model.predict(wav_path)

        local_predictions = result.get("local_predictions", [])
        id2label: dict = model.id2label

        # Compute per-class probabilities as the fraction of windows that predicted each class.
        if local_predictions:
            counts = Counter(local_predictions)
            total = len(local_predictions)
            probabilities = {
                id2label[class_id]: round(count / total, 4)
                for class_id, count in counts.items()
            }
            # Fill in zero probability for any classes that were never predicted.
            for label in id2label.values():
                if label not in probabilities:
                    probabilities[label] = 0.0
        else:
            probabilities = {label: 0.0 for label in id2label.values()}

        global_prediction_label = result.get("global_prediction_label", "")
        global_confidence = float(result.get("global_confidence", 0.0))

    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Use 'huggingface' or 'fastai'."
        )

    return {
        "probabilities": probabilities,
        "global_prediction_label": global_prediction_label,
        "global_confidence": global_confidence,
    }


def print_results(results: dict, model_type: str) -> None:
    """Print inference results to stdout.

    Args:
        results: Dictionary returned by run_inference().
        model_type: Model type string, printed for context.
    """
    probabilities = results["probabilities"]
    label = results["global_prediction_label"]
    confidence = results["global_confidence"]

    print(f"Model type: {model_type}")
    print(f"Global prediction: {label} (confidence: {confidence:.4f})")
    print()
    print("Per-class probabilities:")
    for class_name, prob in sorted(probabilities.items()):
        print(f"  {class_name}: {prob:.4f}")


def main() -> int:
    """Entry point for the run_inference CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Run model inference on a wav file and output per-class probabilities."
    )
    parser.add_argument(
        "wav_file",
        help="Path to the wav file to score.",
    )
    parser.add_argument(
        "--model",
        choices=["huggingface", "fastai"],
        default="huggingface",
        help=(
            "Model type to use (default: huggingface). "
            "huggingface: 7-class model (water, resident, transient, humpback, vessel, jingle, human). "
            "fastai: 2-class model (other, whale)."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to model directory or HuggingFace Hub model ID. "
            "Required for --model huggingface. "
            "Defaults to ./model for --model fastai."
        ),
    )

    args = parser.parse_args()

    wav_path = args.wav_file
    if not Path(wav_path).exists():
        print(f"Error: wav file not found: {wav_path}", file=sys.stderr)
        return 1

    try:
        results = run_inference(wav_path, model_type=args.model, model_path=args.model_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Inference failed: {e}", file=sys.stderr)
        return 1

    print_results(results, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
