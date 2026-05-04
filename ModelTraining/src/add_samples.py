#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Split a WAV file into 3-second segments and run inference on each segment.

Usage:
    python add_samples.py recording.wav --node-name rpi_orcasound_lab \\
        --timestamp 2025_01_15_12_30_00_PST --model-path /path/to/model
    python add_samples.py recording.wav --node-name rpi_sunset_bay \\
        --timestamp 2025_01_15_12_30_00_PST --model fastai

Saves 3-second segments with a 2-second hop to the "new/" output directory
(configurable with --output-dir) using the same filename convention as
output/wav/humpback/ etc.: {node_name_with_hyphens}_{timestamp_pst}.wav.
The timestamp in each filename reflects the actual start time of that sample.

Then runs inference on each saved segment and prints the predicted class label.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ffmpeg
import numpy as np
from pytz import timezone

from model_inference import get_model_inference

SEGMENT_DURATION = 3  # Duration of each segment in seconds.
HOP_DURATION = 2  # Hop size between segments in seconds.
DEFAULT_OUTPUT_DIR = "new"  # Default output directory for segments.
PACIFIC_TZ = timezone("US/Pacific")  # Pacific timezone for timestamp formatting.


def parse_timestamp_pst(timestamp_str: str) -> datetime:
    """
    Parse a PST timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.

    Args:
        timestamp_str: Timestamp string (e.g., "2025_12_24_17_51_23_PST").

    Returns:
        Parsed datetime object localized to the Pacific timezone.
    """
    timestamp_str = timestamp_str.replace("_PST", "")
    dt_naive = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
    return PACIFIC_TZ.localize(dt_naive)


def format_timestamp_pst(dt: datetime) -> str:
    """
    Format a datetime as a PST timestamp string.

    Args:
        dt: Datetime object (should already be localized to Pacific timezone).

    Returns:
        Timestamp string in the format YYYY_MM_DD_HH_MM_SS_PST.
    """
    return dt.strftime("%Y_%m_%d_%H_%M_%S_PST")


def split_wav_into_segments(
    wav_file: str,
    node_name: str,
    base_timestamp: str,
    output_dir: Path,
    segment_duration: int = SEGMENT_DURATION,
    hop_duration: int = HOP_DURATION,
) -> list[tuple[Path, str]]:
    """
    Split a WAV file into fixed-duration segments with a hop and save to output_dir.

    Uses the same filename convention as output/wav/humpback/ etc.:
    {node_name_with_hyphens}_{timestamp_pst}.wav, where the timestamp is the
    actual start time of each sample.

    Args:
        wav_file: Path to the input WAV file.
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        base_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
        output_dir: Directory to save the segment WAV files.
        segment_duration: Duration of each segment in seconds (default: 3).
        hop_duration: Hop size between segment start times in seconds (default: 2).

    Returns:
        List of (segment_path, timestamp_str) tuples for each saved segment,
        in order of increasing start time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe the audio file to get duration.
    try:
        probe = ffmpeg.probe(wav_file)
        duration = float(probe["format"]["duration"])
    except Exception as e:
        print(f"Error: Could not probe {wav_file}: {e}", file=sys.stderr)
        return []

    # Compute number of segment positions (sliding window).
    # Each position starts at pos_idx * hop_duration seconds.
    # The last position must start early enough that the full segment fits.
    num_positions = int(np.floor((duration - segment_duration) / hop_duration)) + 1
    if num_positions < 1:
        num_positions = 1

    # Parse base timestamp and build filename prefix.
    base_time = parse_timestamp_pst(base_timestamp)
    # Replace underscores with hyphens in the node name (matches download_wavs.py convention).
    node_name_in_filename = node_name.replace("_", "-")

    segments: list[tuple[Path, str]] = []
    for pos_idx in range(num_positions):
        start_offset = pos_idx * hop_duration
        seg_time = base_time + timedelta(seconds=start_offset)
        timestamp_str = format_timestamp_pst(seg_time)
        filename = f"{node_name_in_filename}_{timestamp_str}.wav"
        out_path = output_dir / filename

        if out_path.exists():
            print(f"Skipping (already exists): {out_path}")
            segments.append((out_path, timestamp_str))
            continue

        try:
            stream = ffmpeg.input(wav_file, ss=start_offset)
            stream = ffmpeg.output(
                stream,
                str(out_path),
                t=segment_duration,
                acodec="pcm_s16le",
                ar=44100,
                ac=1,
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            print(f"Saved: {out_path}")
            segments.append((out_path, timestamp_str))
        except Exception as e:
            print(
                f"Warning: Failed to extract segment at offset {start_offset}s: {e}",
                file=sys.stderr,
            )

    return segments


def get_segment_prediction(model: object, segment_path: Path, model_type: str) -> str:
    """
    Run inference on a single segment WAV file and return the predicted class label.

    Args:
        model: Loaded model inference object (from get_model_inference).
        segment_path: Path to the segment WAV file to score.
        model_type: Model type string ('podsai', 'fastai', or 'orcahello').

    Returns:
        Predicted class label string (e.g., "resident", "humpback", "other").
        Returns "unknown" if inference fails.
    """
    try:
        result = model.predict(str(segment_path))
    except Exception as e:
        print(f"Warning: Inference failed for {segment_path}: {e}", file=sys.stderr)
        return "unknown"

    if model_type == "podsai":
        label = result.get("global_prediction_label", "unknown")
    else:
        # FastAI and OrcaHello use binary predictions (0 = other, 1 = resident).
        global_prediction = result.get("global_prediction", 0)
        label = "resident" if global_prediction else "other"

    return label


def add_samples(
    wav_file: str,
    node_name: str,
    base_timestamp: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_type: str = "podsai",
    model_path: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Split a WAV file into 3-second segments, save them, and run inference on each.

    Saves segments to output_dir using the filename convention
    {node_name_with_hyphens}_{timestamp_pst}.wav and returns a list of
    (filename, predicted_class) pairs.

    Args:
        wav_file: Path to the input WAV file.
        node_name: Hydrophone node name (e.g., "rpi_orcasound_lab").
        base_timestamp: PST timestamp of the start of the recording
            (e.g., "2025_01_15_12_30_00_PST").
        output_dir: Directory to save segments (default: "new").
        model_type: Model type to use ('podsai', 'fastai', or 'orcahello').
        model_path: Path to model directory or HuggingFace Hub model ID.
            Required for podsai.

    Returns:
        List of (filepath, predicted_class) tuples, one per segment.

    Raises:
        ValueError: If model_path is not provided for the podsai model type.
    """
    out_dir = Path(output_dir)

    # Split the WAV and save segments.
    segments = split_wav_into_segments(wav_file, node_name, base_timestamp, out_dir)
    if not segments:
        return []

    # Resolve default model paths (matching run_inference.py conventions).
    if model_type == "fastai":
        if model_path is None:
            model_path = "./model"
    elif model_type == "orcahello":
        if model_path is None:
            model_path = "orcasound/orcahello-srkw-detector-v1"
    elif model_type == "podsai":
        if model_path is None:
            raise ValueError(
                "model_path is required for --model podsai. "
                "Provide a path to a fine-tuned model directory or a HuggingFace Hub model ID."
            )

    # Load the model once and run inference on each segment.
    print(f"\nLoading {model_type} model from {model_path}...")
    model = get_model_inference(model_type=model_type, model_path=model_path)

    results: list[tuple[str, str]] = []
    print("\nSegment predictions:")
    for seg_path, _timestamp_str in segments:
        label = get_segment_prediction(model, seg_path, model_type)
        results.append((str(seg_path), label))
        print(f"  {seg_path.name}: {label}")

    return results


def main() -> int:
    """Entry point for the add_samples CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Split a WAV file into 3-second segments with 2-second hop, "
            "save to the output directory using the standard filename convention "
            "({node_name_with_hyphens}_{timestamp_pst}.wav), "
            "and run inference on each segment to output its predicted class."
        )
    )
    parser.add_argument(
        "wav_file",
        help="Path to the input WAV file to segment.",
    )
    parser.add_argument(
        "--node-name",
        required=True,
        help=(
            "Hydrophone node name (e.g., 'rpi_orcasound_lab'). "
            "Used in output filenames (underscores are replaced with hyphens)."
        ),
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help=(
            "PST timestamp of the start of the recording "
            "(e.g., '2025_01_15_12_30_00_PST'). "
            "Each segment filename encodes the actual start time of that sample."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save segments (default: {DEFAULT_OUTPUT_DIR!r}).",
    )
    parser.add_argument(
        "--model",
        choices=["podsai", "fastai", "orcahello"],
        default="podsai",
        help=(
            "Model type to use for inference (default: podsai). "
            "podsai: 7-class model (water, resident, transient, humpback, vessel, jingle, human). "
            "fastai: 2-class model (other, resident). "
            "orcahello: 2-class SRKW detector (other, resident)."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to model directory or HuggingFace Hub model ID. "
            "Required for --model podsai. "
            "Defaults to ./model for --model fastai. "
            "Defaults to orcasound/orcahello-srkw-detector-v1 for --model orcahello."
        ),
    )

    args = parser.parse_args()

    if not Path(args.wav_file).exists():
        print(f"Error: WAV file not found: {args.wav_file}", file=sys.stderr)
        return 1

    try:
        results = add_samples(
            wav_file=args.wav_file,
            node_name=args.node_name,
            base_timestamp=args.timestamp,
            output_dir=args.output_dir,
            model_type=args.model,
            model_path=args.model_path,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1

    if not results:
        print("No segments were processed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
