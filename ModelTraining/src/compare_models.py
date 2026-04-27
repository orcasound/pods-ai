#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Compare multiple models on a test set of audio samples.

Usage:
    python compare_models.py [options]

Reads testing_samples.csv, runs each enabled model (fastai, orcahello, huggingface)
on the corresponding 60-second WAV files, and reports correct identifications,
false positives, and false negatives per model.

A "correct" identification means:
  - Model predicted "resident" (whale) when the label is "resident".
  - Model predicted anything other than "resident" when the label is not "resident".

A "false positive" means the model predicted "resident" when the correct label is not "resident".
A "false negative" means the model predicted something other than "resident" when the label is "resident".
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from run_inference import run_inference

RESIDENT_LABEL = "resident"


@dataclass
class TestSample:
    """A single row from testing_samples.csv."""

    category: str
    node_name: str
    timestamp: str
    uri: str
    description: str
    notes: str


@dataclass
class ModelResult:
    """Accumulated results for a single model across all test samples."""

    model_type: str
    total: int = 0
    correct: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    skipped: int = 0

    @property
    def evaluated(self) -> int:
        """Number of samples actually evaluated (not skipped)."""
        return self.total - self.skipped

    @property
    def accuracy(self) -> Optional[float]:
        """Fraction of evaluated samples correctly identified."""
        if self.evaluated == 0:
            return None
        return self.correct / self.evaluated

    @property
    def false_positive_rate(self) -> Optional[float]:
        """Fraction of evaluated samples that are false positives."""
        if self.evaluated == 0:
            return None
        return self.false_positives / self.evaluated

    @property
    def false_negative_rate(self) -> Optional[float]:
        """Fraction of evaluated samples that are false negatives."""
        if self.evaluated == 0:
            return None
        return self.false_negatives / self.evaluated


def load_testing_samples(csv_path: Path) -> list[TestSample]:
    """
    Load testing samples from a CSV file.

    Args:
        csv_path: Path to testing_samples.csv.

    Returns:
        List of TestSample objects, or an empty list on error.
    """
    samples = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(TestSample(
                    category=row.get("Category", ""),
                    node_name=row.get("NodeName", ""),
                    timestamp=row.get("Timestamp", ""),
                    uri=row.get("URI", ""),
                    description=row.get("Description", ""),
                    notes=row.get("Notes", ""),
                ))
    except OSError as e:
        print(f"Error reading {csv_path}: {e}", file=sys.stderr)
    return samples


def find_wav_file(sample: TestSample, wav_dir: Path) -> Optional[Path]:
    """
    Find the 60-second WAV file for a testing sample.

    WAV files are saved by download_wavs.py as:
        <wav_dir>/<category>/<node_name_with_dashes>_<timestamp>.wav

    Args:
        sample: The testing sample.
        wav_dir: Root directory of testing WAV files.

    Returns:
        Path to the WAV file, or None if not found.
    """
    node_name_in_filename = sample.node_name.replace("_", "-")
    wav_filename = f"{node_name_in_filename}_{sample.timestamp}.wav"
    wav_path = wav_dir / sample.category / wav_filename
    if wav_path.exists():
        return wav_path
    return None


def is_resident_prediction(global_prediction_label: str, model_type: str) -> bool:
    """
    Determine whether a model's prediction corresponds to "resident" (whale).

    For binary models (fastai, orcahello), "whale" maps to resident.
    For the HuggingFace 7-class model, "resident" maps directly to resident.

    Args:
        global_prediction_label: The model's predicted class label.
        model_type: The model type ('fastai', 'orcahello', or 'huggingface').

    Returns:
        True if the prediction is "resident"; False otherwise.
    """
    if model_type in ("fastai", "orcahello"):
        # Binary models use "whale" for resident and "other" for everything else.
        return global_prediction_label == "whale"
    # HuggingFace 7-class model uses "resident" directly.
    return global_prediction_label == RESIDENT_LABEL


def evaluate_model(
    model_type: str,
    model_path: Optional[str],
    samples: list[TestSample],
    wav_dir: Path,
) -> ModelResult:
    """
    Run a model against all test samples and accumulate results.

    Args:
        model_type: One of 'fastai', 'orcahello', 'huggingface'.
        model_path: Path to the model (or HuggingFace Hub model ID).
        samples: List of testing samples.
        wav_dir: Root directory containing testing WAV files.

    Returns:
        ModelResult with counts of correct, false positive, and false negative predictions.
    """
    result = ModelResult(model_type=model_type, total=len(samples))

    for sample in samples:
        wav_path = find_wav_file(sample, wav_dir)
        if wav_path is None:
            print(
                f"  [{model_type}] Skipping {sample.category}/{sample.node_name}"
                f"/{sample.timestamp}: WAV not found"
            )
            result.skipped += 1
            continue

        expected_resident = (sample.category == RESIDENT_LABEL)

        try:
            inference_result = run_inference(str(wav_path), model_type=model_type, model_path=model_path)
        except Exception as e:
            print(f"  [{model_type}] Error on {wav_path.name}: {e}")
            result.skipped += 1
            continue

        predicted_label = inference_result.get("global_prediction_label", "")
        predicted_resident = is_resident_prediction(predicted_label, model_type)

        if predicted_resident == expected_resident:
            result.correct += 1
            status = "correct"
        elif predicted_resident and not expected_resident:
            result.false_positives += 1
            status = "false_positive"
        else:
            result.false_negatives += 1
            status = "false_negative"

        print(
            f"  [{model_type}] {sample.category}/{sample.node_name}/{sample.timestamp}: "
            f"predicted={predicted_label!r} → {status}"
        )

    return result


def print_summary(results: list[ModelResult]) -> None:
    """
    Print a formatted comparison table for all model results.

    Args:
        results: List of ModelResult objects, one per model.
    """
    print()
    print("=" * 70)
    print("Model Comparison Summary")
    print("=" * 70)
    header = (
        f"{'Model':<15} {'Evaluated':>9} {'Correct':>9} {'Accuracy':>9}"
        f" {'FP':>6} {'FP%':>7} {'FN':>6} {'FN%':>7}"
    )
    print(header)
    print("-" * 70)

    for r in results:
        evaluated = r.evaluated
        accuracy = f"{r.accuracy:.1%}" if r.accuracy is not None else "N/A"
        fp_rate = f"{r.false_positive_rate:.1%}" if r.false_positive_rate is not None else "N/A"
        fn_rate = f"{r.false_negative_rate:.1%}" if r.false_negative_rate is not None else "N/A"
        print(
            f"{r.model_type:<15} {evaluated:>9} {r.correct:>9} {accuracy:>9}"
            f" {r.false_positives:>6} {fp_rate:>7} {r.false_negatives:>6} {fn_rate:>7}"
        )
        if r.skipped:
            print(f"  ({r.skipped} skipped due to missing WAV or inference error)")

    print("=" * 70)
    print()
    print("Definitions:")
    print("  Correct      = predicted resident when expected, or non-resident when expected")
    print("  FP (false+)  = predicted resident when correct class was non-resident")
    print("  FN (false-)  = predicted non-resident when correct class was resident")


def main() -> int:
    """Entry point for the compare_models CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare model predictions on a test set of 60-second audio samples. "
            "Reads testing_samples.csv, runs each enabled model, and reports "
            "correct identifications, false positives, and false negatives."
        )
    )
    parser.add_argument(
        "--testing-csv",
        default="../output/csv/testing_samples.csv",
        help="Path to testing_samples.csv (default: ../output/csv/testing_samples.csv).",
    )
    parser.add_argument(
        "--wav-dir",
        default="../output/testing-wav",
        help="Root directory containing testing WAV files (default: ../output/testing-wav).",
    )
    parser.add_argument(
        "--models",
        default="fastai,orcahello,huggingface",
        help=(
            "Comma-separated list of models to evaluate "
            "(default: fastai,orcahello,huggingface)."
        ),
    )
    parser.add_argument(
        "--fastai-model-path",
        default=None,
        help=(
            "Path to FastAI model directory. "
            "Defaults to ./model (the run_inference.py default) when not specified."
        ),
    )
    parser.add_argument(
        "--orcahello-model-path",
        default=None,
        help=(
            "Path or HuggingFace Hub ID for the OrcaHello model. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 when not specified."
        ),
    )
    parser.add_argument(
        "--huggingface-model-path",
        default=None,
        help=(
            "Path to HuggingFace model directory or Hub model ID. "
            "Required when 'huggingface' is included in --models."
        ),
    )

    args = parser.parse_args()

    testing_csv = Path(args.testing_csv)
    if not testing_csv.exists():
        print(f"Error: testing CSV not found: {testing_csv}", file=sys.stderr)
        print(
            "Run extract_training_samples.py first to generate testing_samples.csv.",
            file=sys.stderr,
        )
        return 1

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"Error: WAV directory not found: {wav_dir}", file=sys.stderr)
        print(
            "Run download_wavs.py first to download testing WAV files.",
            file=sys.stderr,
        )
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid_models = {"fastai", "orcahello", "huggingface"}
    for model in models:
        if model not in valid_models:
            print(
                f"Error: unknown model type {model!r}. Valid: {sorted(valid_models)}",
                file=sys.stderr,
            )
            return 1

    if "huggingface" in models and args.huggingface_model_path is None:
        print(
            "Error: --huggingface-model-path is required when 'huggingface' is in --models.",
            file=sys.stderr,
        )
        return 1

    model_paths: dict[str, Optional[str]] = {
        "fastai": args.fastai_model_path,
        "orcahello": args.orcahello_model_path,
        "huggingface": args.huggingface_model_path,
    }

    samples = load_testing_samples(testing_csv)
    if not samples:
        print("Error: no testing samples found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(samples)} testing samples from {testing_csv}")
    print(f"WAV directory: {wav_dir}")
    print(f"Models to evaluate: {', '.join(models)}")
    print()

    results = []
    for model_type in models:
        print(f"Evaluating model: {model_type}")
        model_result = evaluate_model(
            model_type=model_type,
            model_path=model_paths[model_type],
            samples=samples,
            wav_dir=wav_dir,
        )
        results.append(model_result)
        print()

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
