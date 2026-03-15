#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""
Given a node_name and timestamp_str, compute and output the best corrected
timestamp URI by running process_sample().

Usage:
    python get_best_timestamp.py <node_name> <timestamp_str> [--no-model] [--duration N]

Example:
    python get_best_timestamp.py rpi_orcasound_lab 2023_08_18_00_59_53_PST
"""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from extract_training_samples import (
    generate_uri,
    load_manual_corrections,
    process_sample,
    REPO_ROOT,
    SEGMENT_DURATION_SECONDS,
)
from model_inference import get_model_inference


def node_name_to_slug(node_name: str) -> str:
    """
    Convert a node_name to the URL slug used in Orcasound bouts URIs.

    Args:
        node_name: Node name such as 'rpi_orcasound_lab'.

    Returns:
        Slug such as 'orcasound-lab'.
    """
    # Remove the 'rpi_' prefix if present.
    if node_name.startswith('rpi_'):
        node_name = node_name[4:]
    return node_name.replace('_', '-')


def build_sample(node_name: str, timestamp_str: str) -> dict:
    """
    Build a minimal detection sample dict from node_name and timestamp_str.

    Args:
        node_name: Node name (e.g., 'rpi_orcasound_lab').
        timestamp_str: PST timestamp string (e.g., '2023_08_18_00_59_53_PST').

    Returns:
        Sample dict suitable for passing to process_sample().
    """
    slug = node_name_to_slug(node_name)
    base_uri = f"https://live.orcasound.net/bouts/new/{slug}"
    # Encode the original timestamp into the URI so process_sample can use it
    # as a baseline before applying its correction strategy.
    original_uri = generate_uri(base_uri, timestamp_str)
    return {
        'Category': '',
        'NodeName': node_name,
        'Timestamp': timestamp_str,
        'URI': original_uri,
        'Description': '',
        'Notes': 'tp_human_only',
        'Confidence': '',
    }


def main():
    """Compute and print the best corrected timestamp URI for a detection."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute the best corrected timestamp URI for a detection by "
            "running process_sample() with model inference."
        )
    )
    parser.add_argument(
        'node_name',
        type=str,
        help="Node name (e.g., rpi_orcasound_lab)",
    )
    parser.add_argument(
        'timestamp_str',
        type=str,
        help="PST timestamp string (e.g., 2023_08_18_00_59_53_PST)",
    )
    parser.add_argument(
        '--no-model',
        action='store_true',
        help=(
            "Skip model inference and apply a fixed offset correction "
            f"(subtract {SEGMENT_DURATION_SECONDS} seconds) instead."
        ),
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=SEGMENT_DURATION_SECONDS,
        help=f"Duration of each audio segment in seconds (default: {SEGMENT_DURATION_SECONDS})",
    )
    args = parser.parse_args()

    # Construct a sample dict from the provided arguments.
    sample = build_sample(args.node_name, args.timestamp_str)

    # Load manual timestamp corrections.
    manual_corrections_path = REPO_ROOT / 'output' / 'csv' / 'manual_timestamps.csv'
    manual_timestamps, manual_confidences = load_manual_corrections(manual_corrections_path)

    # Initialise model inference unless --no-model was requested.
    model_inference = None
    if not args.no_model:
        model_type = os.environ.get("MODEL_TYPE", "fastai")
        model_path = os.environ.get("MODEL_PATH", "./model")
        model_url = os.environ.get("MODEL_URL", None)
        auto_download_default = "true" if model_type == "fastai" else "false"
        auto_download = os.environ.get("MODEL_AUTO_DOWNLOAD", auto_download_default).lower() == "true"
        try:
            model_inference = get_model_inference(
                model_path=model_path if model_type == "fastai" else None,
                model_type=model_type,
                auto_download=auto_download,
                model_url=model_url,
            )
        except Exception as e:
            print(f"Warning: Failed to initialise model inference: {e}", file=sys.stderr)
            print("Falling back to fixed offset correction.", file=sys.stderr)

    # Run process_sample() and print the corrected URI.
    with TemporaryDirectory() as tmp_dir:
        result = process_sample(
            sample, manual_timestamps, manual_confidences, model_inference, tmp_dir, args.duration
        )

    print(result['URI'])


if __name__ == '__main__':
    main()
